// Constants
const BATCH_SIZE = 16; // Define your batch size here
const TARGET_SIZE = [512, 512];
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
const MODEL_PATHS = {
    detection: 'models/db_mobilenet_v2/model.json',
    recognition: 'models/crnn_mobilenet_v2/model.json'
};

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
let reusableCanvas;

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
        detectionModel = await tf.loadGraphModel(MODEL_PATHS.detection);
        
        showLoading('Loading recognition model...');
        recognitionModel = await tf.loadGraphModel(MODEL_PATHS.recognition);
        
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

function preprocessImages(images) {
    // Ensure images is always an array
    const imageArray = Array.isArray(images) ? images : [images];
    
    const resizedImages = imageArray.map(img => {
        const canvas = document.createElement('canvas');
        canvas.width = TARGET_SIZE[0];
        canvas.height = TARGET_SIZE[1];
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        return tf.browser.fromPixels(canvas);
    });

    const batchTensor = tf.stack(resizedImages);
    const mean = tf.tensor1d([255 * DET_MEAN, 255 * DET_MEAN, 255 * DET_MEAN]);
    const std = tf.tensor1d([255 * DET_STD, 255 * DET_STD, 255 * DET_STD]);
    return batchTensor.div(std).sub(mean);
}

async function getHeatMapFromImage(images) {
    console.log('Entering getHeatMapFromImage');
    const batchTensor = preprocessImages(images);
    console.log('Batch tensor shape:', batchTensor.shape);
    
    const predictions = await detectionModel.execute(batchTensor);
    console.log('Predictions shape:', predictions.shape);
    
    let heatmapData;
    if (!Array.isArray(images)) {
        heatmapData = predictions.squeeze();
    } else {
        heatmapData = predictions;
    }
    console.log('Heatmap data shape:', heatmapData.shape);
    
    const boundingBoxes = await Promise.resolve(extractBoundingBoxesFromHeatmap(heatmapData, TARGET_SIZE));
    
    batchTensor.dispose();
    predictions.dispose();
    if (heatmapData !== predictions) {
        heatmapData.dispose();
    }
    
    console.log('Returning bounding boxes:', boundingBoxes.length);
    return boundingBoxes;
}

function preprocessCrops(crops) {
    const resizedCrops = crops.map(crop => {
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(crop, 0, 0, canvas.width, canvas.height);
        return tf.browser.fromPixels(canvas);
    });

    const batchTensor = tf.stack(resizedCrops);
    const mean = tf.tensor1d([255 * REC_MEAN, 255 * REC_MEAN, 255 * REC_MEAN]);
    const std = tf.tensor1d([255 * REC_STD, 255 * REC_STD, 255 * REC_STD]);
    return batchTensor.div(std).sub(mean).expandDims(0);
}

async function recognizeBatch(crops) {
    const batchTensor = preprocessCrops(crops);
    const predictions = await recognitionModel.executeAsync(batchTensor);
    const probabilities = tf.softmax(predictions, -1);
    const bestPaths = tf.unstack(tf.argMax(probabilities, -1), 0);
    batchTensor.dispose();
    predictions.dispose();
    probabilities.dispose();
    return bestPaths;
}

async function detectAndRecognizeText(imageElement) {
    if (isMobile()) {
        useCPU(); // Switch to CPU for mobile devices
    }

    const heatmapCanvas = await getHeatMapFromImage(imageElement);
    const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, TARGET_SIZE);

    previewCanvas.width = TARGET_SIZE[0];
    previewCanvas.height = TARGET_SIZE[1];
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);

    let fullText = '';
    const crops = [];

    for (const box of boundingBoxes) {
        const [x1, y1] = box.coordinates[0];
        const [x2, y2] = box.coordinates[2];
        const width = (x2 - x1) * imageElement.width;
        const height = (y2 - y1) * imageElement.height;
        const x = x1 * imageElement.width;
        const y = y1 * imageElement.height;

        ctx.strokeStyle = box.config.stroke;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        const croppedCanvas = document.createElement('canvas');
        croppedCanvas.width = Math.min(width, 128);
        croppedCanvas.height = Math.min(height, 32);
        croppedCanvas.getContext('2d').drawImage(
            imageElement, 
            x, y, width, height,
            0, 0, width, height
        );

        crops.push({
            canvas: croppedCanvas,
            bbox: {
                x: Math.round(x),
                y: Math.round(y),
                width: Math.round(width),
                height: Math.round(height)
            }
        });
    }

    // Process crops in batches
    const batchSize = BATCH_SIZE;
    for (let i = 0; i < crops.length; i += batchSize) {
        const batch = crops.slice(i, i + batchSize);
        const bestPaths = await recognizeBatch(batch.map(crop => crop.canvas));

        const words = decodeText(bestPaths);

        // Associate each word with its bounding box
        words.split(' ').forEach((word, index) => {
            if (word && batch[index]) {
                extractedData.push({
                    word: word,
                    boundingBox: batch[index].bbox
                });
            }
        });
    }

    return extractedData;
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
    showLoading('Processing image...');

    await ensureModelsLoaded();  // Ensure models are loaded before processing

    const targetSize = TARGET_SIZE;
    canvas.width = targetSize[0];
    canvas.height = targetSize[1];
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    imageDataUrl = canvas.toDataURL('image/jpeg', isMobile() ? 0.7 : 0.9);
    
    const img = new Image();
    img.src = imageDataUrl;
    img.onload = async () => {
        try {
            extractedData = await detectAndRecognizeText(img);
            extractedText = extractedData.map(item => item.word).join(' ');
            resultElement.textContent = `Extracted Text: ${extractedText}`;
            
            previewCanvas.style.display = 'block';
            confirmButton.style.display = 'inline-block';
            retryButton.style.display = 'inline-block';
            captureButton.style.display = 'none';
        } catch (error) {
            console.error('Error during text extraction:', error);
            resultElement.textContent = 'Error occurred during text extraction';
        } finally {
            enableCaptureButton();
            hideLoading();
            tf.disposeVariables();
        }
    };
}

function isMobile() {
    return false;
    //return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
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
    
function extractBoundingBoxesFromHeatmap(heatmapData, size) {
    console.log('Entering extractBoundingBoxesFromHeatmap');
    console.log('Heatmap data type:', typeof heatmapData);
    console.log('Is TensorFlow tensor:', heatmapData instanceof tf.Tensor);
    
    if (heatmapData instanceof tf.Tensor) {
        console.log('Tensor shape:', heatmapData.shape);
        console.log('Tensor rank:', heatmapData.rank);
    }
    
    let tensorData;
    if (heatmapData instanceof tf.Tensor) {
        // Ensure the tensor is 2D
        if (heatmapData.rank > 2) {
            console.log('Squeezing tensor');
            heatmapData = heatmapData.squeeze();
        }
        console.log('Squeezed tensor shape:', heatmapData.shape);
        
        // Attempt to synchronously get the data
        tensorData = heatmapData.dataSync();
        console.log('Initial tensorData length:', tensorData.length);
        
        // If the data is not immediately available, wait for it
        if (tensorData.length === 0) {
            console.log('Data not immediately available, waiting...');
            return new Promise((resolve) => {
                setTimeout(() => {
                    tensorData = heatmapData.dataSync();
                    console.log('Delayed tensorData length:', tensorData.length);
                    resolve(processData(tensorData, size));
                }, 1000); // Wait for 1 second
            });
        }
    } else if (Array.isArray(heatmapData) || heatmapData instanceof Float32Array || heatmapData instanceof Float64Array) {
        tensorData = heatmapData;
    } else {
        console.error('Invalid input type for heatmapData');
        return [];
    }
    
    return processData(tensorData, size);
}

function processData(tensorData, size) {
    console.log('Processing data');
    console.log('Data length:', tensorData.length);
    console.log('Expected data length:', size[0] * size[1]);
    
    // Check if the data length matches the expected size
    if (tensorData.length !== size[0] * size[1]) {
        console.error('Data length does not match expected size');
        return [];
    }
    
    // Create or reuse canvas
    if (!reusableCanvas) {
        reusableCanvas = document.createElement('canvas');
    }
    reusableCanvas.width = size[1];
    reusableCanvas.height = size[0];
    const ctx = reusableCanvas.getContext('2d');
    
    // Create ImageData with the correct dimensions
    const imageData = ctx.createImageData(size[1], size[0]);
    
    // Fill the ImageData
    for (let i = 0; i < tensorData.length; i++) {
        const value = Math.floor(tensorData[i] * 255); // Assuming values are in [0, 1]
        imageData.data[i * 4] = value;     // R
        imageData.data[i * 4 + 1] = value; // G
        imageData.data[i * 4 + 2] = value; // B
        imageData.data[i * 4 + 3] = 255;   // A (fully opaque)
    }
    
    ctx.putImageData(imageData, 0, 0);

    // Now use the canvas with cv.imread
    let src = cv.imread(reusableCanvas);
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
    
    console.log('Extracted bounding boxes:', boundingBoxes.length);
    return boundingBoxes;
}

function useCPU() {
    tf.setBackend('cpu');
    console.log('Switched to CPU backend');
}

function getRandomColor() {
    return '#' + Math.floor(Math.random()*16777215).toString(16);
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
    
    await tf.ready();
     
    if (await isWebGPUSupported()){
        console.log('WebGPU is supported. Attempting to set backend...');
        try {
            await tf.setBackend('webgpu');
            console.log('Successfully set WebGPU backend');
        } catch (e) {
            console.error('Failed to set WebGPU backend:', e);
            await fallbackToWebGLorCPU();
        }
    } else {
        console.log('WebGPU is not supported');
        await fallbackToWebGLorCPU();
    }
    
    initializeModelLoading();
    await setupCamera();
    
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
    
    hideLoading();
}

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
