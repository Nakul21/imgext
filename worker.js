// worker.js
importScripts('https://docs.opencv.org/4.5.2/opencv.js');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js');

const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
const TARGET_SIZE = [512, 512];

let detectionModel;
let recognitionModel;

// Worker initialization
async function initializeWorker() {
    await tf.ready();
    await tf.setBackend('cpu'); // Workers always use CPU backend
    
    try {
        detectionModel = await tf.loadGraphModel('models/db_mobilenet_v2/model.json');
        recognitionModel = await tf.loadGraphModel('models/crnn_mobilenet_v2/model.json');
        self.postMessage({ type: 'initialized' });
    } catch (error) {
        self.postMessage({ type: 'error', error: 'Model initialization failed' });
    }
}

// Helper functions for image processing
function preprocessImageForDetection(imageData) {
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(imageData)
            .resizeNearestNeighbor(TARGET_SIZE)
            .toFloat()
            .sub(tf.scalar(255 * DET_MEAN))
            .div(tf.scalar(255 * DET_STD))
            .expandDims();
        return tensor;
    });
}

function preprocessImageForRecognition(crops) {
    const targetSize = [32, 128];
    return tf.tidy(() => {
        const processedTensors = crops.map((crop) => {
            return tf.browser.fromPixels(crop)
                .resizeNearestNeighbor(targetSize)
                .toFloat()
                .sub(tf.scalar(255 * REC_MEAN))
                .div(tf.scalar(255 * REC_STD));
        });
        return tf.stack(processedTensors);
    });
}

async function processImageBatch(imageDataArray, boundingBoxes) {
    const results = [];
    try {
        const tensor = preprocessImageForRecognition(imageDataArray);
        const predictions = await recognitionModel.executeAsync(tensor);
        const probabilities = tf.softmax(predictions, -1);
        const bestPath = tf.argMax(probabilities, -1).arraySync();
        
        const words = decodeText(bestPath.map(path => tf.tensor1d(path)));
        
        words.split(' ').forEach((word, index) => {
            if (word && word.trim() && boundingBoxes[index]) {
                results.push({
                    word: word.trim(),
                    boundingBox: boundingBoxes[index]
                });
            }
        });
        
        tensor.dispose();
        predictions.dispose();
        probabilities.dispose();
        
        return results;
    } catch (error) {
        throw new Error(`Batch processing failed: ${error.message}`);
    }
}

function decodeText(bestPath) {
    const blank = 126;
    let collapsed = "";
    let lastChar = null;

    for (const sequence of bestPath) {
        const values = sequence.dataSync();
        for (const k of values) {
            if (k !== blank && k !== lastChar) {         
                collapsed += VOCAB[k];
                lastChar = k;
            } else if (k === blank) {
                lastChar = null;
            }
        }
        collapsed += ' ';
    }
    return collapsed.trim();
}

// Worker message handler
self.onmessage = async function(e) {
    const { type, data } = e.data;
    
    try {
        switch(type) {
            case 'init':
                await initializeWorker();
                break;
                
            case 'processRegion':
                const { imageData, region } = data;
                const results = await processImageBatch([imageData], [region]);
                self.postMessage({ 
                    type: 'regionComplete', 
                    results,
                    regionId: data.regionId 
                });
                break;
                
            case 'detect':
                const tensor = preprocessImageForDetection(data.imageData);
                const prediction = await detectionModel.execute(tensor);
                const boxes = await extractBoundingBoxes(prediction);
                tensor.dispose();
                prediction.dispose();
                self.postMessage({ 
                    type: 'detectComplete', 
                    boxes 
                });
                break;
                
            case 'getMemoryInfo':
                const memInfo = await tf.memory();
                self.postMessage({ 
                    type: 'memoryInfo', 
                    info: memInfo 
                });
                break;
        }
    } catch (error) {
        self.postMessage({ 
            type: 'error', 
            error: error.message 
        });
    }
};

// app.js (Main Thread)
class WorkerPool {
    constructor(workerScript, poolSize) {
        this.workers = [];
        this.available = [];
        this.queue = [];
        this.poolSize = poolSize || navigator.hardwareConcurrency || 4;
        
        for (let i = 0; i < this.poolSize; i++) {
            const worker = new Worker(workerScript);
            worker.onmessage = this.handleWorkerMessage.bind(this);
            this.workers.push(worker);
            this.available.push(i);
        }
    }
    
    async initialize() {
        const initPromises = this.workers.map((worker, index) => {
            return new Promise((resolve) => {
                const handler = (e) => {
                    if (e.data.type === 'initialized') {
                        worker.removeEventListener('message', handler);
                        resolve();
                    }
                };
                worker.addEventListener('message', handler);
                worker.postMessage({ type: 'init' });
            });
        });
        
        await Promise.all(initPromises);
    }
    
    async processTask(task) {
        return new Promise((resolve, reject) => {
            const workerIndex = this.available.shift();
            
            if (workerIndex !== undefined) {
                const worker = this.workers[workerIndex];
                
                const handler = (e) => {
                    if (e.data.type === task.responseType) {
                        worker.removeEventListener('message', handler);
                        this.available.push(workerIndex);
                        this.processNextTask();
                        resolve(e.data);
                    } else if (e.data.type === 'error') {
                        reject(new Error(e.data.error));
                    }
                };
                
                worker.addEventListener('message', handler);
                worker.postMessage(task.message);
            } else {
                this.queue.push({ task, resolve, reject });
            }
        });
    }
    
    processNextTask() {
        if (this.queue.length > 0 && this.available.length > 0) {
            const { task, resolve, reject } = this.queue.shift();
            this.processTask(task).then(resolve).catch(reject);
        }
    }
    
    terminate() {
        this.workers.forEach(worker => worker.terminate());
        this.workers = [];
        this.available = [];
        this.queue = [];
    }
}

// Modified detectAndRecognizeText function
async function detectAndRecognizeText(imageElement) {
    const workerPool = new WorkerPool('worker.js');
    await workerPool.initialize();
    
    try {
        // Detection phase
        const detectionResult = await workerPool.processTask({
            message: {
                type: 'detect',
                data: { imageData: imageElement }
            },
            responseType: 'detectComplete'
        });
        
        const boundingBoxes = detectionResult.boxes;
        const results = [];
        
        // Recognition phase - process regions in parallel
        const recognitionPromises = boundingBoxes.map((box, index) => {
            return workerPool.processTask({
                message: {
                    type: 'processRegion',
                    data: {
                        imageData: imageElement,
                        region: box,
                        regionId: index
                    }
                },
                responseType: 'regionComplete'
            });
        });
        
        const recognitionResults = await Promise.all(recognitionPromises);
        
        // Combine and sort results
        recognitionResults.forEach(result => {
            if (result.results && result.results.length > 0) {
                results.push(...result.results);
            }
        });
        
        // Sort results by vertical position
        results.sort((a, b) => {
            return a.boundingBox.y - b.boundingBox.y;
        });
        
        return results;
        
    } catch (error) {
        console.error('Error in parallel processing:', error);
        throw error;
    } finally {
        workerPool.terminate();
    }
}

// Memory monitoring
function monitorMemoryUsage(workerPool) {
    return setInterval(async () => {
        const mainInfo = await tf.memory();
        console.log('Main Thread Memory:', {
            numTensors: mainInfo.numTensors,
            numDataBuffers: mainInfo.numDataBuffers
        });
        
        workerPool.workers.forEach((worker, index) => {
            worker.postMessage({ type: 'getMemoryInfo' });
        });
    }, 5000);
}

// Initialize the application
async function init() {
    showLoading('Initializing...');
    
    try {
        await tf.ready();
        const workerPool = new WorkerPool('worker.js');
        await workerPool.initialize();
        
        // Start memory monitoring
        monitorMemoryUsage(workerPool);
        
        await setupCamera();
        hideLoading();
        
        return workerPool;
    } catch (error) {
        console.error('Initialization failed:', error);
        showLoading('Initialization failed. Please refresh the page.');
        throw error;
    }
}

// Export necessary functions
export {
    init,
    detectAndRecognizeText,
    WorkerPool
};
