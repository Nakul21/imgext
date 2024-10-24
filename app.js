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
const MOBILE_MAX_DIMENSION = 1024; // Maximum texture dimension for mobile
const DESKTOP_MAX_DIMENSION = 2048; // Maximum texture dimension for desktop
const MOBILE_BATCH_SIZE = 4; // Smaller batch size for mobile
const DESKTOP_BATCH_SIZE = 32; // Regular batch size for desktop
const MAX_TEXTURE_SIZE = 4096;

let modelLoadingPromise;

let imageDataUrl = '';
let extractedText = '';
let extractedData = [];
let detectionModel;
let recognitionModel;

function getDeviceCapabilities() {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    
    if (!gl) {
        return {
            maxTextureSize: MOBILE_MAX_DIMENSION,
            isMobile: true
        };
    }

    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    return {
        maxTextureSize: Math.min(maxTextureSize, isMobile ? MOBILE_MAX_DIMENSION : DESKTOP_MAX_DIMENSION),
        isMobile
    };
}

function calculateResizeDimensions(width, height, maxDimension) {
    if (width <= maxDimension && height <= maxDimension) {
        return { width, height };
    }
    
    const aspectRatio = width / height;
    
    if (width > height) {
        return {
            width: maxDimension,
            height: Math.round(maxDimension / aspectRatio)
        };
    } else {
        return {
            width: Math.round(maxDimension * aspectRatio),
            height: maxDimension
        };
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
        // Calculate scale to fit within texture limits while maintaining aspect ratio
        const scale = Math.min(
            MAX_TEXTURE_SIZE / imageElement.width,
            MAX_TEXTURE_SIZE / imageElement.height,
            1
        );
        
        const scaledWidth = Math.floor(imageElement.width * scale);
        const scaledHeight = Math.floor(imageElement.height * scale);
        
        // Create a temporary canvas for resizing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = scaledWidth;
        tempCanvas.height = scaledHeight;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(imageElement, 0, 0, scaledWidth, scaledHeight);
        
        // Convert to tensor with controlled dimensions
        let tensor = tf.browser.fromPixels(tempCanvas);
        
        // Resize to target size if different from current size
        if (scaledWidth !== TARGET_SIZE[0] || scaledHeight !== TARGET_SIZE[1]) {
            tensor = tensor.resizeNearestNeighbor(TARGET_SIZE);
        }
        
        tensor = tensor.toFloat();
        const mean = tf.scalar(255 * DET_MEAN);
        const std = tf.scalar(255 * DET_STD);
        return tensor.sub(mean).div(std).expandDims();
    });
}

function preprocessImageForRecognition(crops) {
    return tf.tidy(() => {
        const targetSize = [32, 128];
        const tensors = crops.map((crop) => {
            // Scale down if crop exceeds texture limits
            const scale = Math.min(
                MAX_TEXTURE_SIZE / crop.width,
                MAX_TEXTURE_SIZE / crop.height,
                1
            );
            
            const scaledWidth = Math.floor(crop.width * scale);
            const scaledHeight = Math.floor(crop.height * scale);
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = scaledWidth;
            tempCanvas.height = scaledHeight;
            const ctx = tempCanvas.getContext('2d');
            ctx.drawImage(crop, 0, 0, scaledWidth, scaledHeight);
            
            let h = scaledHeight;
            let w = scaledWidth;
            let resizeTarget, paddingTarget;
            let aspectRatio = targetSize[1] / targetSize[0];
            
            if (aspectRatio * h > w) {
                resizeTarget = [targetSize[0], Math.round((targetSize[0] * w) / h)];
                paddingTarget = [
                    [0, 0],
                    [0, targetSize[1] - Math.round((targetSize[0] * w) / h)],
                    [0, 0],
                ];
            } else {
                resizeTarget = [Math.round((targetSize[1] * h) / w), targetSize[1]];
                paddingTarget = [
                    [0, targetSize[0] - Math.round((targetSize[1] * h) / w)],
                    [0, 0],
                    [0, 0],
                ];
            }
            
            return tf.browser
                .fromPixels(tempCanvas)
                .resizeNearestNeighbor(resizeTarget)
                .pad(paddingTarget, 0)
                .toFloat()
                .expandDims();
        });
        
        const tensor = tf.concat(tensors);
        const mean = tf.scalar(255 * REC_MEAN);
        const std = tf.scalar(255 * REC_STD);
        return tensor.sub(mean).div(std);
    });
}

async function processBatch(batch, extractedData) {
    const deviceCaps = getDeviceCapabilities();
    const batchSize = deviceCaps.isMobile ? MOBILE_BATCH_SIZE : DESKTOP_BATCH_SIZE;
    
    // Process in smaller sub-batches if needed
    for (let i = 0; i < batch.length; i += batchSize) {
        const subBatch = batch.slice(i, i + batchSize);
        let inputTensor = null;
        let predictions = null;
        let probabilities = null;
        let bestPath = null;
        
        try {
            inputTensor = preprocessImageForRecognition(subBatch.map(crop => crop.canvas));
            predictions = await recognitionModel.executeAsync(inputTensor);
            probabilities = tf.softmax(predictions, -1);
            bestPath = tf.unstack(tf.argMax(probabilities, -1), 0);
            
            const words = decodeText(bestPath);
            
            words.split(' ').forEach((word, index) => {
                if (word && subBatch[index]) {
                    extractedData.push({
                        word: word,
                        boundingBox: subBatch[index].bbox
                    });
                }
            });
            
            // Force garbage collection after each sub-batch
            await tf.nextFrame();
            
        } finally {
            // Clean up tensors
            if (inputTensor) inputTensor.dispose();
            if (predictions) predictions.dispose();
            if (probabilities) probabilities.dispose();
            if (bestPath) bestPath.forEach(t => t.dispose());
        }
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

async function getHeatMapFromImage(imageObject) {
    let tensor = null;
    let prediction = null;
    try {
        // Wrap the tensor creation in tidy
        tensor = tf.tidy(() => {
            return preprocessImageForDetection(imageObject);
        });
        
        // Execute model outside tidy since it's async
        prediction = await detectionModel.execute(tensor);
        prediction = tf.squeeze(prediction, 0);
        if (Array.isArray(prediction)) {
            prediction = prediction[0];
        }
        
        const heatmapCanvas = document.createElement('canvas');
        heatmapCanvas.width = imageObject.width;
        heatmapCanvas.height = imageObject.height;
        await tf.browser.toPixels(prediction, heatmapCanvas);
        
        return heatmapCanvas;
    } finally {
        // Clean up tensors
        if (tensor) tensor.dispose();
        if (prediction) prediction.dispose();
    }
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

async function detectAndRecognizeText(imageElement) {
    const deviceCaps = getDeviceCapabilities();
    
    if (deviceCaps.isMobile) {
        //useCPU();
    }
    
    let heatmapCanvas = null;
    
    try {
        // Resize input image if needed
        const { width, height } = calculateResizeDimensions(
            imageElement.width,
            imageElement.height,
            deviceCaps.maxTextureSize
        );
        
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(imageElement, 0, 0, width, height);
        
        // Process resized image
        heatmapCanvas = await getHeatMapFromImage(tempCanvas);
        const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, TARGET_SIZE);
        
        // Setup preview canvas
        previewCanvas.width = width;
        previewCanvas.height = height;
        previewCanvas.getContext('2d').drawImage(tempCanvas, 0, 0);
        
        const crops = [];

        // Generate crops
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

        
        // Process in smaller batches for mobile
        const batchSize = deviceCaps.isMobile ? MOBILE_BATCH_SIZE : DESKTOP_BATCH_SIZE;
        
        for (let i = 0; i < crops.length; i += batchSize) {
            const batch = crops.slice(i, i + batchSize);
            await processBatch(batch, extractedData);
            
            // Force garbage collection between batches
            if (deviceCaps.isMobile) {
                await new Promise(resolve => setTimeout(resolve, 100)); // Add small delay on mobile
            }
            await tf.nextFrame();
        }
        
        return extractedData;
        
    } catch(error) {
        console.error('Error in detectAndRecognizeText:', error);
        throw error;
    } finally {
        // Clean up any remaining tensors
        tf.engine().startScope();
        tf.engine().endScope();
        await tf.nextFrame();
    }
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

    await ensureModelsLoaded();

    // Get the video dimensions
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    
    // Calculate scale to fit within texture limits
    const scale = Math.min(
        MAX_TEXTURE_SIZE / videoWidth,
        MAX_TEXTURE_SIZE / videoHeight,
        1
    );
    
    const scaledWidth = Math.floor(videoWidth * scale);
    const scaledHeight = Math.floor(videoHeight * scale);

    // Set canvas dimensions to scaled size
    canvas.width = scaledWidth;
    canvas.height = scaledHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, scaledWidth, scaledHeight);

    imageDataUrl = canvas.toDataURL('image/jpeg', isMobile() ? 0.7 : 0.9);
    
    const img = new Image();
    img.src = imageDataUrl;
    img.onload = async () => {
        try {
            extractedData = await detectAndRecognizeText(img);
            extractedText = extractedData.map(item => item.word).join(' ');
            resultElement.textContent = `Extracted Text: ${extractedText}`;
            
            // Set preview canvas dimensions
            previewCanvas.width = scaledWidth;
            previewCanvas.height = scaledHeight;
            previewCanvas.getContext('2d').drawImage(img, 0, 0);
            
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
            numBytesInGPU: info.numBytesInGPU,
            numBytes: info.numBytes,
            numDataBuffers: info.numDataBuffers,
            unreliable: info.unreliable,
            reasons: info.reasons
        });
    }, 3000);
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
    monitorMemoryUsage();
    
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
