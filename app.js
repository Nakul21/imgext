// Constants
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
const TARGET_SIZE = [512, 512];
// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const previewCanvas = document.getElementById('previewCanvas');
const captureButton = document.getElementById('captureButton');
const confirmButton = document.getElementById('confirmButton');
const retryButton = document.getElementById('retryButton');
const actionButtons = document.getElementById('actionButtons');
const sendButton = document.getElementById('sendButton');
const discardButton = document.getElementById('discardButton');
const resultElement = document.getElementById('result');
const apiResponseElement = document.getElementById('apiResponse');
const loadingIndicator = document.getElementById('loadingIndicator');
const appContainer = document.getElementById('appContainer');

let modelLoadingPromise;

let imageDataUrl = '';
let extractedText = '';
let extractedData = [];
let detectionModel;
let recognitionModel;


class WorkerPool {
    constructor(workerScript) {
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

async function isWebGPUSupported() {
    if (!navigator.gpu) {
        return false;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return false;
        }
        const device = await adapter.requestDevice();
        return !!device;
    } catch (e) {
        console.error("Error checking WebGPU support:", e);
        return false;
    }
}

async function fallbackToWebGLorCPU() {
        try {
            await tf.setBackend('webgl');
            console.log('Fallback to WebGL backend successful');
        } catch (e) {
            console.error('Failed to set WebGL backend:', e);
            useCPU();
        }
}

function showLoading(message) {
    loadingIndicator.textContent = message;
    loadingIndicator.style.display = 'block';
    //appContainer.style.display = 'none';
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
    //appContainer.style.display = 'block';
}

async function loadModels() {
    try {
        showLoading('Loading detection model...');
        detectionModel = await tf.loadGraphModel('models/db_mobilenet_v2/model.json');
        
        showLoading('Loading recognition model...');
        recognitionModel = await tf.loadGraphModel('models/crnn_mobilenet_v2/model.json');
        
        console.log('Models loaded successfully');
        hideLoading();
    } catch (error) {
        console.error('Error loading models:', error);
        showLoading('Error loading models. Please refresh the page.');
    }
}

function initializeModelLoading() {
    modelLoadingPromise = loadModels();
}

async function ensureModelsLoaded() {
    if (modelLoadingPromise) {
        await modelLoadingPromise;
    }
}

async function setupCamera() {
    showLoading('Setting up camera...');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 512 },
                height: { ideal: 512 }
            } 
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                hideLoading();
                resolve(video);
            };
        });
    } catch (error) {
        console.error('Error setting up camera:', error);
        showLoading('Error setting up camera. Please check permissions and refresh.');
    }
}

function preprocessImageForDetection(imageElement) {
    return tf.tidy(() => {
        return tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor(TARGET_SIZE)
            .toFloat()
            .sub(tf.scalar(255 * DET_MEAN))
            .div(tf.scalar(255 * DET_STD))
            .expandDims();
    });
}

function preprocessImageForRecognition(crops) {
    const targetSize = [32, 128];
    return tf.tidy(() => {
        const processedTensors = crops.map((crop) => {
            const h = crop.height;
            const w = crop.width;
            const aspectRatio = targetSize[1] / targetSize[0];
            
            // Precalculate resize dimensions
            const [resizeTarget, paddingTarget] = tf.tidy(() => {
                if (aspectRatio * h > w) {
                    const newWidth = Math.round((targetSize[0] * w) / h);
                    return [
                        [targetSize[0], newWidth],
                        [[0, 0], [0, targetSize[1] - newWidth], [0, 0]]
                    ];
                } else {
                    const newHeight = Math.round((targetSize[1] * h) / w);
                    return [
                        [newHeight, targetSize[1]],
                        [[0, targetSize[0] - newHeight], [0, 0], [0, 0]]
                    ];
                }
            });

            return tf.browser.fromPixels(crop)
                .resizeNearestNeighbor(resizeTarget)
                .pad(paddingTarget, 0)
                .toFloat();
        });

        return tf.stack(processedTensors)
            .sub(tf.scalar(255 * REC_MEAN))
            .div(tf.scalar(255 * REC_STD));
    });
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

async function getHeatMapFromImage(imageObject) {
    let tensor = preprocessImageForDetection(imageObject);
    let prediction = await detectionModel.execute(tensor);
    prediction = tf.squeeze(prediction, 0);
    if (Array.isArray(prediction)) {
        prediction = prediction[0];
    }
    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = imageObject.width;
    heatmapCanvas.height = imageObject.height;
    await tf.browser.toPixels(prediction, heatmapCanvas);
    tensor.dispose();
    prediction.dispose();
    return heatmapCanvas;
}

function clamp(number, size) {
    return Math.max(0, Math.min(number, size));
}

function transformBoundingBox(contour, id, size) {
    let offset = (contour.width * contour.height * 1.8) / (2 * (contour.width + contour.height));
    const p1 = clamp(contour.x - offset, size[1]) - 1;
    const p2 = clamp(p1 + contour.width + 2 * offset, size[1]) - 1;
    const p3 = clamp(contour.y - offset, size[0]) - 1;
    const p4 = clamp(p3 + contour.height + 2 * offset, size[0]) - 1;
    return {
        id,
        config: {
            stroke: getRandomColor(),
        },
        coordinates: [
            [p1 / size[1], p3 / size[0]],
            [p2 / size[1], p3 / size[0]],
            [p2 / size[1], p4 / size[0]],
            [p1 / size[1], p4 / size[0]],
        ],
    };
}

function extractBoundingBoxesFromHeatmap(heatmapCanvas, size) {
    let src = cv.imread(heatmapCanvas);
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(src, src, 77, 255, cv.THRESH_BINARY);
    cv.morphologyEx(src, src, cv.MORPH_OPEN, cv.Mat.ones(2, 2, cv.CV_8U));
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    
    const boundingBoxes = [];
    for (let i = 0; i < contours.size(); ++i) {
        const contourBoundingBox = cv.boundingRect(contours.get(i));
        if (contourBoundingBox.width > 2 && contourBoundingBox.height > 2) {
            boundingBoxes.unshift(transformBoundingBox(contourBoundingBox, i, size));
        }
    }
    
    src.delete();
    contours.delete();
    hierarchy.delete();
    return boundingBoxes;
}

function useCPU() {
    tf.setBackend('cpu');
    console.log('Switched to CPU backend');
}

function getRandomColor() {
    return '#' + Math.floor(Math.random()*16777215).toString(16);
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

async function createCropsEfficiently(boundingBoxes, imageElement) {
    const crops = [];
    const offscreenCanvas = document.createElement('canvas');
    const ctx = offscreenCanvas.getContext('2d', { alpha: false });
    
    for (const box of boundingBoxes) {
        const [x1, y1] = box.coordinates[0];
        const [x2, y2] = box.coordinates[2];
        const width = (x2 - x1) * imageElement.width;
        const height = (y2 - y1) * imageElement.height;
        const x = x1 * imageElement.width;
        const y = y1 * imageElement.height;
        
        // Reuse canvas instead of creating new ones
        offscreenCanvas.width = Math.min(width, 128);
        offscreenCanvas.height = Math.min(height, 32);
        
        ctx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
        ctx.drawImage(
            imageElement,
            x, y, width, height,
            0, 0, offscreenCanvas.width, offscreenCanvas.height
        );
        
        // Clone the canvas for storage
        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = offscreenCanvas.width;
        cropCanvas.height = offscreenCanvas.height;
        cropCanvas.getContext('2d').drawImage(offscreenCanvas, 0, 0);
        
        crops.push({
            canvas: cropCanvas,
            bbox: {
                x: Math.round(x),
                y: Math.round(y),
                width: Math.round(width),
                height: Math.round(height)
            }
        });
    }
    
    return crops;
}

function disableCaptureButton() {
    captureButton.disabled = true;
    captureButton.textContent = 'Processing...';
}

function enableCaptureButton() {
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
}


async function handleCapture() {
    disableCaptureButton();
    showLoading('Initializing...');
    
    try {
        await ensureModelsLoaded();
        
        const ctx = canvas.getContext('2d', { alpha: false });
        canvas.width = TARGET_SIZE[0];
        canvas.height = TARGET_SIZE[1];
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Optimize image quality while maintaining size
        imageDataUrl = canvas.toDataURL('image/jpeg', 0.95); // Increased quality
        
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = imageDataUrl;
        
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
        });
        
        console.log('Starting text detection and recognition...');
        extractedData = await detectAndRecognizeText(img);
        
        if (extractedData.length === 0) {
            resultElement.textContent = 'No text detected in image';
        } else {
            extractedText = extractedData.map(item => item.word).join(' ');
            resultElement.textContent = `Extracted Text: ${extractedText}`;
        }
        
        // Show preview and buttons
        previewCanvas.style.display = 'block';
        confirmButton.style.display = 'inline-block';
        retryButton.style.display = 'inline-block';
        captureButton.style.display = 'none';
        
    } catch (error) {
        console.error('Error during capture:', error);
        resultElement.textContent = 'Error occurred during processing. Please try again.';
    } finally {
        enableCaptureButton();
        hideLoading();
    }
}

function updateLoadingMessage(operation) {
    const messages = {
        detection: 'Detecting text regions...',
        processing: 'Processing detected regions...',
        recognition: 'Recognizing text...'
    };
    loadingIndicator.textContent = messages[operation] || 'Processing...';
}

function updateProgress(current, total) {
    const progress = Math.round((current / total) * 100);
    loadingIndicator.textContent = `Recognizing text... ${progress}%`;
}

function isMobile() {
    console.log('navigator.userAgent',navigator.userAgent);
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function handleConfirm() {
    toggleButtons(true);
    previewCanvas.style.display = 'none';
    confirmButton.style.display = 'none';
    retryButton.style.display = 'none';
}

function handleRetry() {
    resetUI();
}

async function handleSend() {
    if (!extractedText) return;
    apiResponseElement.textContent = 'Submitting...';
    let msgKey = new Date().getTime();
    try {
        const response = await fetch('https://kvdb.io/NyKpFtJ7v392NS8ibLiofx/'+msgKey, {
            method: 'PUT',
            body: JSON.stringify({
                extractetAt: msgKey,
                probableTextContent: extractedText,
                boundingBoxes: extractedData,
                userId: "imageExt",
            }),
            headers: {
                'Content-type': 'application/json; charset=UTF-8',
            },
        });

        if (response.status !== 200) {
            throw new Error('Failed to push this data to server');
        } 
        
        apiResponseElement.textContent = 'Submitted the extract with ID : ' + msgKey; 
        
    } catch (error) {
        console.error('Error submitting to server:', error);
        apiResponseElement.textContent = 'Error occurred while submitting to server';
    } finally {
        resetUI();
    }
}

function toggleButtons(showActionButtons) {
    captureButton.style.display = showActionButtons ? 'none' : 'block';
    actionButtons.style.display = showActionButtons ? 'block' : 'none';
}

function resetUI() {
    toggleButtons(false);
    resultElement.textContent = '';
    apiResponseElement.textContent = '';
    imageDataUrl = '';
    extractedText = '';
    extractedData = [];
    clearCanvas();
    previewCanvas.style.display = 'none';
    confirmButton.style.display = 'none';
    retryButton.style.display = 'none';
    captureButton.style.display = 'block';
}

function clearCanvas() {
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    previewCanvas.getContext('2d').clearRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function monitorMemoryUsage() {
    setInterval(() => {
        const info = tf.memory();
        console.log('Memory usage:', {
            numTensors: info.numTensors,
            numDataBuffers: info.numDataBuffers,
            unreliable: info.unreliable,
            reasons: info.reasons
        });
    }, 5000);
}

async function init() {
    showLoading('Initializing...');
    
    try {
        await tf.ready();
        const workerPool = new WorkerPool('worker.js');
        await workerPool.initialize();
        
        // Start memory monitoring
        //monitorMemoryUsage(workerPool);
        
        await setupCamera();
        hideLoading();
        
        return workerPool;
    } catch (error) {
        console.error('Initialization failed:', error);
        showLoading('Initialization failed. Please refresh the page.');
        throw error;
    }
}

// function monitorMemoryUsage(workerPool) {
//     return setInterval(async () => {
//         const mainInfo = await tf.memory();
//         console.log('Main Thread Memory:', {
//             numTensors: mainInfo.numTensors,
//             numDataBuffers: mainInfo.numDataBuffers
//         });
        
//         workerPool.workers.forEach((worker, index) => {
//             worker.postMessage({ type: 'getMemoryInfo' });
//         });
//     }, 5000);
// }

function loadOpenCV() {
    return new Promise((resolve) => {
        const script = document.createElement('script');
        script.src = 'https://docs.opencv.org/4.5.2/opencv.js';
        script.onload = () => resolve();
        document.body.appendChild(script);
    });
}

// Event Listeners
captureButton.addEventListener('click', handleCapture);
captureButton.addEventListener('touchstart', handleCapture);
confirmButton.addEventListener('click', handleConfirm);
confirmButton.addEventListener('touchstart', handleConfirm);
retryButton.addEventListener('click', handleRetry);
retryButton.addEventListener('touchstart', handleRetry);
sendButton.addEventListener('click', handleSend);
sendButton.addEventListener('touchstart', handleSend);
discardButton.addEventListener('click', resetUI);
discardButton.addEventListener('touchstart', resetUI);

// Initialize the application
init();

// Service Worker Registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('service-worker.js')
            .then(registration => {
                console.log('ServiceWorker registration successful with scope: ', registration.scope);
            }, err => {
                console.log('ServiceWorker registration failed: ', err);
            });
    });
}

let deferredPrompt;
const installBtn = document.getElementById('install-btn');

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    installBtn.style.display = 'block';

    installBtn.addEventListener('click', (e) => {
        installBtn.style.display = 'none';
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === 'accepted') {
                console.log('User accepted the A2HS prompt');
            } else {
                console.log('User dismissed the A2HS prompt');
            }
            deferredPrompt = null;
        });
    });
});

window.addEventListener('appinstalled', (evt) => {
    console.log('App was installed.');
    installBtn.style.display = 'none';
});

export default WorkerPool;
export { init , detectAndRecognizeText };

