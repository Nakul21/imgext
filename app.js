import WorkerPool from './worker-pool.js';

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

// Global state
let imageDataUrl = '';
let extractedText = '';
let extractedData = [];
let workerPool = null;
let isInitialized = false;
let detectionModel;
let recognitionModel;


function showLoading(message) {
    loadingIndicator.textContent = message;
    loadingIndicator.style.display = 'block';
    //appContainer.style.display = 'none';
}


function updateLoadingStatus(message) {
    console.log('Loading status:', message);
    loadingIndicator.textContent = message;
    loadingIndicator.style.display = 'block';
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
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
        await tf.setBackend('cpu');
        console.log('Fallback to CPU backend');
    }
}

async function loadModels() {
    try {
        updateLoadingStatus('Loading detection model...');
        detectionModel = await tf.loadGraphModel('models/db_mobilenet_v2/model.json');
        
        updateLoadingStatus('Loading recognition model...');
        recognitionModel = await tf.loadGraphModel('models/crnn_mobilenet_v2/model.json');
        
        console.log('Models loaded successfully');
    } catch (error) {
        console.error('Error loading models:', error);
        throw new Error('Failed to load models: ' + error.message);
    }
}

async function loadOpenCV() {
    updateLoadingStatus('Loading OpenCV...');
    return new Promise((resolve, reject) => {
        if (window.cv) {
            resolve();
            return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://docs.opencv.org/4.5.2/opencv.js';
        script.onload = () => {
            console.log('OpenCV loaded successfully');
            resolve();
        };
        script.onerror = () => {
            const error = new Error('Failed to load OpenCV');
            console.error(error);
            reject(error);
        };
        document.body.appendChild(script);
    });
}

async function initializeWorkerPool() {
    updateLoadingStatus('Initializing worker pool...');
    try {
        workerPool = new WorkerPool('worker.js');
        await workerPool.initialize();
        console.log('Worker pool initialized successfully');
        return workerPool;
    } catch (error) {
        console.error('Failed to initialize worker pool:', error);
        throw error;
    }
}

async function initializeCamera() {
    updateLoadingStatus('Setting up camera...');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 512 },
                height: { ideal: 512 }
            }
        });
        
        video.srcObject = stream;
        await new Promise((resolve, reject) => {
            video.onloadedmetadata = resolve;
            video.onerror = reject;
        });
        
        // Start playing the video
        await video.play();
        
        console.log('Camera initialized successfully');
        return stream;
    } catch (error) {
        console.error('Camera initialization failed:', error);
        throw new Error(`Camera setup failed: ${error.message}`);
    }
}

async function initializeTensorFlow() {
    updateLoadingStatus('Initializing TensorFlow...');
    try {
        await tf.ready();
        if (await isWebGPUSupported()) {
            await tf.setBackend('webgpu');
        } else {
            await fallbackToWebGLorCPU();
        }
        console.log('TensorFlow initialized successfully');
    } catch (error) {
        console.error('TensorFlow initialization failed:', error);
        throw error;
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

            return tf.browser.fromPixels(crop.canvas)
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

async function detectAndRecognizeText(imageElement) {
    try {
        // Get heatmap
        const heatmap = await getHeatMapFromImage(imageElement);
        
        // Extract bounding boxes
        const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmap, TARGET_SIZE);
        
        // Create crops
        const crops = await createCropsEfficiently(boundingBoxes, imageElement);
        
        // Process crops for recognition
        const processedCrops = preprocessImageForRecognition(crops);
        
        // Recognize text
        const predictions = await recognitionModel.predict(processedCrops);
        const texts = predictions.map(decodeText);
        
        // Combine results
        return boundingBoxes.map((box, index) => ({
            boundingBox: box,
            word: texts[index]
        }));
    } catch (error) {
        console.error('Error in text detection and recognition:', error);
        throw error;
    }
}

async function getHeatMapFromImage(imageElement) {
    let tensor = preprocessImageForDetection(imageElement);
    let prediction = await detectionModel.execute(tensor);
    prediction = tf.squeeze(prediction, 0);
    if (Array.isArray(prediction)) {
        prediction = prediction[0];
    }
    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = imageElement.width;
    heatmapCanvas.height = imageElement.height;
    await tf.browser.toPixels(prediction, heatmapCanvas);
    tensor.dispose();
    prediction.dispose();
    return heatmapCanvas;
}

function clamp(number, size) {
    return Math.max(0, Math.min(number, size));
}

function getRandomColor() {
    return '#' + Math.floor(Math.random()*16777215).toString(16);
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
        
        offscreenCanvas.width = Math.min(width, 128);
        offscreenCanvas.height = Math.min(height, 32);
        
        ctx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
        ctx.drawImage(
            imageElement,
            x, y, width, height,
            0, 0, offscreenCanvas.width, offscreenCanvas.height
        );
        
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
    if (!isInitialized) {
        console.error('Cannot capture - application not initialized');
        return;
    }

    disableCaptureButton();
    updateLoadingStatus('Processing image...');
    
    try {
        const ctx = canvas.getContext('2d');
        canvas.width = TARGET_SIZE[0];
        canvas.height = TARGET_SIZE[1];
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        imageDataUrl = canvas.toDataURL('image/jpeg', 0.95);
        
        const img = new Image();
        img.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
            img.src = imageDataUrl;
        });
        
        console.log('Starting text detection...');
        extractedData = await detectAndRecognizeText(img);
        
        if (extractedData.length === 0) {
            resultElement.textContent = 'No text detected in image';
        } else {
            extractedText = extractedData.map(item => item.word).join(' ');
            resultElement.textContent = `Extracted Text: ${extractedText}`;
        }
        
        // Update UI
        previewCanvas.style.display = 'block';
        confirmButton.style.display = 'inline-block';
        retryButton.style.display = 'inline-block';
        captureButton.style.display = 'none';
        
    } catch (error) {
        console.error('Capture failed:', error);
        resultElement.textContent = `Error: ${error.message}`;
    } finally {
        enableCaptureButton();
        loadingIndicator.style.display = 'none';
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
document.addEventListener('DOMContentLoaded', init);

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

export { init, detectAndRecognizeText };

