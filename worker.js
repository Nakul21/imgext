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

// Export necessary functions
export {
    init,
    detectAndRecognizeText
};
