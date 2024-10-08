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

window.Pica = window.pica;
const pica = new Pica();

let modelLoadingPromise;
let imageDataUrl = '';
let extractedText = '';
let extractedData = [];
let detectionModel;
let recognitionModel;

function getMaxTextureSize() {
    const ctx = document.createElement('canvas').getContext('webgl');
    if (!ctx) return 4096; // Default to a reasonable size if WebGL is not supported
    return ctx.getParameter(ctx.MAX_TEXTURE_SIZE);
}

async function isWebGPUSupported() {
    if (!navigator.gpu) return false;
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;
        const device = await adapter.requestDevice();
        return !!device;
    } catch (e) {
        console.error("Error checking WebGPU support:", e);
        return false;
    }
}

async function fallbackToWebGLorCPU() {
    if (isMobile()) {
        try {
            await tf.setBackend('webgl');
            console.log('Fallback to WebGL backend successful');
        } catch (e) {
            console.error('Failed to set WebGL backend:', e);
            useCPU();
        }
    } else {
        useCPU();
    }
}

function showLoading(message) {
    loadingIndicator.textContent = message;
    loadingIndicator.style.display = 'block';
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
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

async function preprocessImageForDetection(imageElement) {
    const maxTextureSize = getMaxTextureSize();
    const scale = Math.min(1, maxTextureSize / Math.max(imageElement.width, imageElement.height));
    
    const newWidth = Math.round(imageElement.width * scale);
    const newHeight = Math.round(imageElement.height * scale);

    const canvas = document.createElement('canvas');
    canvas.width = newWidth;
    canvas.height = newHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, newWidth, newHeight);

    let tensor = tf.tidy(() => {
        return tf.browser.fromPixels(canvas).toFloat();
    });

    let mean = tf.scalar(255 * DET_MEAN);
    let std = tf.scalar(255 * DET_STD);

    return tensor.sub(mean).div(std).expandDims();
}

async function preprocessImageForRecognition(crops) {
    const targetSize = [32, 128];

    const processedCrops = await Promise.all(crops.map(async (crop) => {
        let h = crop.height;
        let w = crop.width;
        let resizeTarget, paddingTarget;
        let aspectRatio = targetSize[1] / targetSize[0];

        if (w === 0 || h === 0) {
            console.warn('Invalid crop dimensions:', w, h);
            return null; // Skip this crop
        }

        if (aspectRatio * h > w) {
            resizeTarget = [targetSize[0], Math.max(1, Math.round((targetSize[0] * w) / h))];
            paddingTarget = [
                [0, 0],
                [0, Math.max(0, targetSize[1] - resizeTarget[1])],
                [0, 0],
            ];
        } else {
            resizeTarget = [Math.max(1, Math.round((targetSize[1] * h) / w)), targetSize[1]];
            paddingTarget = [
                [0, Math.max(0, targetSize[0] - resizeTarget[0])],
                [0, 0],
                [0, 0],
            ];
        }

        const canvas = document.createElement('canvas');
        canvas.width = resizeTarget[1];
        canvas.height = resizeTarget[0];

        try {
            await pica.resize(crop, canvas, {
                quality: 3,
                alpha: false,
            });

            return tf.tidy(() => {
                return tf.browser
                    .fromPixels(canvas)
                    .pad(paddingTarget, 0)
                    .toFloat()
                    .expandDims();
            });
        } catch (error) {
            console.error('Error processing crop:', error);
            return null; // Skip this crop if there's an error
        }
    }));

    // Filter out null values (skipped crops)
    const validProcessedCrops = processedCrops.filter(crop => crop !== null);

    if (validProcessedCrops.length === 0) {
        throw new Error('No valid crops to process');
    }

    const tensor = tf.concat(validProcessedCrops);
    let mean = tf.scalar(255 * REC_MEAN);
    let std = tf.scalar(255 * REC_STD);
    return tensor.sub(mean).div(std);
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
    let tensor = await preprocessImageForDetection(imageObject);
    
    if (isMobile() && tf.env().getBool('WEBGL_USE_SHAPES_UNIFORMS')) {
        tensor = tf.cast(tensor, 'float16');
    }
    
    let prediction = await detectionModel.execute({'x' : tensor});
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

async function detectAndRecognizeText(imageElement) {
    if (isMobile()) {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('Using WebGL backend for mobile');
    }

    const maxTextureSize = getMaxTextureSize();
    console.log('Max texture size:', maxTextureSize);

    // Preprocess the image, ensuring it fits within texture size limits
    const tensor = await preprocessImageForDetection(imageElement);
    
    // Run detection model
    const prediction = await detectionModel.execute({'x': tensor});
    const squeezedPrediction = tf.squeeze(prediction, 0);

    // Convert prediction to heatmap
    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = tensor.shape[2];
    heatmapCanvas.height = tensor.shape[1];
    await tf.browser.toPixels(squeezedPrediction, heatmapCanvas);

    // Extract bounding boxes
    const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, [tensor.shape[2], tensor.shape[1]]);

    // Scale bounding boxes back to original image size
    const scaleX = imageElement.width / tensor.shape[2];
    const scaleY = imageElement.height / tensor.shape[1];
    boundingBoxes.forEach(box => {
        box.coordinates = box.coordinates.map(coord => [
            coord[0] * scaleX,
            coord[1] * scaleY
        ]);
    });

    // Clean up tensors
    tensor.dispose();
    prediction.dispose();
    squeezedPrediction.dispose();

    // Process each bounding box for text recognition
    let extractedData = [];
    const recognitionBatchSize = isMobile() ? 4 : 16; // Smaller batch size for mobile

    for (let i = 0; i < boundingBoxes.length; i += recognitionBatchSize) {
        const batch = boundingBoxes.slice(i, i + recognitionBatchSize);
        
        // Prepare crops for recognition
        const crops = batch.map(box => {
            const [x1, y1] = box.coordinates[0];
            const [x2, y2] = box.coordinates[2];
            const width = x2 - x1;
            const height = y2 - y1;

            const cropCanvas = document.createElement('canvas');
            cropCanvas.width = width;
            cropCanvas.height = height;
            const ctx = cropCanvas.getContext('2d');
            ctx.drawImage(imageElement, x1, y1, width, height, 0, 0, width, height);

            return cropCanvas;
        });

        // Preprocess crops for recognition
        const inputTensor = await preprocessImageForRecognition(crops);

        // Run recognition model
        const predictions = await recognitionModel.executeAsync(inputTensor);
        const probabilities = tf.softmax(predictions, -1);
        const bestPath = tf.unstack(tf.argMax(probabilities, -1), 0);
        
        // Decode text
        const words = decodeText(bestPath);

        // Associate each word with its bounding box
        words.split(' ').forEach((word, index) => {
            if (word && batch[index]) {
                extractedData.push({
                    word: word,
                    boundingBox: {
                        x: Math.round(batch[index].coordinates[0][0]),
                        y: Math.round(batch[index].coordinates[0][1]),
                        width: Math.round(batch[index].coordinates[2][0] - batch[index].coordinates[0][0]),
                        height: Math.round(batch[index].coordinates[2][1] - batch[index].coordinates[0][1])
                    }
                });
            }
        });

        // Clean up tensors
        inputTensor.dispose();
        predictions.dispose();
        probabilities.dispose();
        bestPath.forEach(tensor => tensor.dispose());
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
                extractedAt: msgKey,
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
    
    await tf.ready();
    if (isMobile()) {
        console.log('Mobile device detected. Using WebGL backend.');
        await tf.setBackend('webgl');
        
        // Adjust WebGL settings for mobile
        const gl = tf.backend().getGPGPUContext().gl;
        gl.getExtension('OES_texture_float');
        gl.getExtension('WEBGL_color_buffer_float');
        
        // Reduce precision if possible
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        tf.env().set('WEBGL_RENDER_FLOAT32_CAPABLE', true);
    }
    
    if (await isWebGPUSupported()) {
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
