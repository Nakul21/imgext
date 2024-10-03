// Constants
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

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



let imageDataUrl = '';
let extractedText = '';
let detectionModel;
let recognitionModel;
let boundingBoxes = [];

async function loadModels() {
    try {
        detectionModel = await tf.loadGraphModel('models/db_mobilenet_v2/model.json');
        recognitionModel = await tf.loadGraphModel('models/crnn_mobilenet_v2/model.json');
        console.log('Models loaded successfully');
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
        } 
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

function preprocessImageForDetection(imageElement) {
    const targetSize = [512, 512];
    let tensor = tf.browser
        .fromPixels(imageElement)
        .resizeNearestNeighbor(targetSize)
        .toFloat();
    let mean = tf.scalar(255 * DET_MEAN);
    let std = tf.scalar(255 * DET_STD);
    return tensor.sub(mean).div(std).expandDims();
}

function preprocessImageForRecognition(crops) {
    const targetSize = [32, 128];
    const tensors = crops.map((crop) => {
        let h = crop.height;
        let w = crop.width;
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
        return tf.tidy(() => {
            return tf.browser
                .fromPixels(crop)
                .resizeNearestNeighbor(resizeTarget)
                .pad(paddingTarget, 0)
                .toFloat()
                .expandDims();
        });
    });
    const tensor = tf.concat(tensors);
    let mean = tf.scalar(255 * REC_MEAN);
    let std = tf.scalar(255 * REC_STD);
    return tensor.sub(mean).div(std);
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

function getRandomColor() {
    return '#' + Math.floor(Math.random()*16777215).toString(16);
}

async function detectAndRecognizeText(imageElement) {
    const size = [512, 512];
    const heatmapCanvas = await getHeatMapFromImage(imageElement);
    boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, size);
    console.log('extractBoundingBoxesFromHeatmap', boundingBoxes);

    previewCanvas.width = imageElement.width;
    previewCanvas.height = imageElement.height;
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);

    let fullText = '';
    const crops = [];
    const extractedWords = [];

    // Limit the number of bounding boxes processed on mobile
    const maxBoxes = isMobile() ? 10 : boundingBoxes.length;
    for (let i = 0; i < Math.min(maxBoxes, boundingBoxes.length); i++) {
        const box = boundingBoxes[i];
        // Draw bounding box
        const [x1, y1] = box.coordinates[0];
        const [x2, y2] = box.coordinates[2];
        const width = (x2 - x1) * imageElement.width;
        const height = (y2 - y1) * imageElement.height;
        const x = x1 * imageElement.width;
        const y = y1 * imageElement.height;

        ctx.strokeStyle = box.config.stroke;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Create crop
        const croppedCanvas = document.createElement('canvas');
        croppedCanvas.width = width;
        croppedCanvas.height = height;
        croppedCanvas.getContext('2d').drawImage(
            imageElement, 
            x, y, width, height,
            0, 0, width, height
        );

        crops.push(croppedCanvas);
    }

    // Process crops in smaller batches on mobile
    const batchSize = isMobile() ? 4 : 32;
    for (let i = 0; i < crops.length; i += batchSize) {
        const batch = crops.slice(i, i + batchSize);
        const inputTensor = preprocessImageForRecognition(batch);

        try {
            console.log('Input tensor shape:', inputTensor.shape);
            
            // Use lower precision on mobile
            const predictions = await recognitionModel.executeAsync(inputTensor, {
                precision: isMobile() ? 'low' : 'high'
            });

            console.log('Raw model output:', predictions);
            console.log('Model output type:', typeof predictions);
            console.log('Model output shape:', predictions.shape);

            let probabilities;
            if (Array.isArray(predictions)) {
                console.log('Model output is an array');
                probabilities = tf.softmax(predictions[0], -1);
            } else if (predictions instanceof tf.Tensor) {
                console.log('Model output is a tensor');
                probabilities = tf.softmax(predictions, -1);
            } else {
                throw new Error('Unexpected output format from recognition model');
            }

            console.log('Probabilities shape:', probabilities.shape);

            const bestPath = tf.argMax(probabilities, -1);
            console.log('Best path shape:', bestPath.shape);
            
            const words = decodeText(bestPath);
            console.log('Decoded words:', words);

            fullText += words + ' ';

            // Store extracted words for each bounding box
            for (let j = 0; j < batch.length; j++) {
                extractedWords.push({
                    boundingBox: boundingBoxes[i + j],
                    text: words
                });
            }

            tf.dispose([inputTensor, predictions, probabilities, bestPath]);
        } catch (error) {
            console.error('Error processing batch:', error);
            console.error('Error stack:', error.stack);
            // Continue with the next batch
        }
    }
    
    return { fullText: fullText.trim(), extractedWords };
}

function decodeText(bestPath) {
    const blank = 126;
    let collapsed = "";
    let lastChar = null;

    const values = bestPath.dataSync();
    for (const k of values) {
        if (k !== blank && k !== lastChar) {
            if (collapsed.length > 0 && !VOCAB[k].match(/[.,!?;:]/)) {
                collapsed += ' '; // Add space before new word
            }
            collapsed += VOCAB[k];
            lastChar = k;
        } else if (k === blank) {
            lastChar = null;
        }
    }
    return collapsed.trim();
}

// Add this helper function to safely log tensor information
function logTensorInfo(tensor, name) {
    console.log(`${name}:`);
    console.log('  Shape:', tensor.shape);
    console.log('  Dtype:', tensor.dtype);
    try {
        const data = tensor.dataSync();
        console.log('  First few values:', data.slice(0, 10));
    } catch (error) {
        console.log('  Unable to access data synchronously');
    }
}

function isMobile() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}
    
async function handleCapture() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    
    imageDataUrl = canvas.toDataURL('image/jpeg', isMobile() ? 0.7 : 0.9);
    resultElement.textContent = 'Processing image...';
    
    const img = new Image();
    img.src = imageDataUrl;
    img.onload = async () => {
        try {
            loadingIndicator.style.display = 'block';
            const { fullText, extractedWords } = await detectAndRecognizeText(img);
            extractedText = fullText;
            resultElement.textContent = `Extracted Text: ${extractedText}`;
            
            // Store extracted words for later use
            window.extractedWords = extractedWords;
            
            // Show preview canvas and confirmation buttons
            previewCanvas.style.display = 'block';
            confirmButton.style.display = 'inline-block';
            retryButton.style.display = 'inline-block';
            captureButton.style.display = 'none';
        } catch (error) {
            console.error('Error during text extraction:', error);
            resultElement.textContent = 'Error occurred during text extraction';
        } finally {
            loadingIndicator.style.display = 'none';
        }
    };
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
                data: extractedText,
                userId: "imageExt",
                boundingBoxes: boundingBoxes,
                extractedWords: window.extractedWords
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

async function init() {
    loadingIndicator.style.display = 'block';
    await Promise.all([
        loadModels(),
        loadOpenCV(),
        setupCamera()
    ]);
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
    loadingIndicator.style.display = 'none';
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
