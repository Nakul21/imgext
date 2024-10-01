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

let imageDataUrl = '';
let extractedText = '';
let detectionModel;
let recognitionModel;

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

async function preprocessImageForDetection(imageElement) {
    const targetSize = [512, 512];
    let img = tf.browser.fromPixels(imageElement);
    img = tf.image.resizeNearestNeighbor(img, targetSize);
    img = img.toFloat();
    let mean = tf.scalar(255 * DET_MEAN);
    let std = tf.scalar(255 * DET_STD);
    img = img.sub(mean).div(std);
    return img.expandDims();
}

async function preprocessImageForRecognition(imageElement) {
    const targetSize = [32, 128];
    let h = imageElement.height;
    let w = imageElement.width;
    let resizeTarget, paddingTarget;
    const aspectRatio = targetSize[1] / targetSize[0];
    
    if (aspectRatio * h > w) {
        resizeTarget = [targetSize[0], Math.round((targetSize[0] * w) / h)];
        paddingTarget = [
            [0, 0],
            [0, targetSize[1] - resizeTarget[1]],
            [0, 0]
        ];
    } else {
        resizeTarget = [Math.round((targetSize[1] * h) / w), targetSize[1]];
        paddingTarget = [
            [0, targetSize[0] - resizeTarget[0]],
            [0, 0],
            [0, 0]
        ];
    }

    return browser
      .fromPixels(imageElement)
      .resizeNearestNeighbor(resize_target)
      .pad(padding_target, 0)
      .toFloat()
      .expandDims();
}

function decodeText(bestPath) {
    let blank = 126;
    var words = [];
    for (const sequence of bestPath) {
        let collapsed = "";
        let added = false;
        const values = sequence.dataSync();
        const arr = Array.from(values);
        for (const k of arr) {
            if (k === blank) {
                added = false;
            } else if (k !== blank && added === false) {
                collapsed += VOCAB[k];
                added = true;
            }
        }
        words.push(collapsed);
    }
    return words.join(' ');
}

async function getHeatMapFromImage(imageObject) {
    let tensor = await preprocessImageForDetection(imageObject);
    let prediction = await detectionModel.execute(tensor);
    prediction = tf.squeeze(prediction,0);
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

function extractBoundingBoxesFromHeatmap(heatmapCanvas, size) {
    let src = cv.imread(heatmapCanvas);
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY);
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

function transformBoundingBox(contour, id, size) {
    let offset = (contour.width * contour.height * 1.8) / (2 * (contour.width + contour.height));
    const p1 = Math.max(0, Math.min(contour.x - offset, size[1])) - 1;
    const p2 = Math.max(0, Math.min(p1 + contour.width + 2 * offset, size[1])) - 1;
    const p3 = Math.max(0, Math.min(contour.y - offset, size[0])) - 1;
    const p4 = Math.max(0, Math.min(p3 + contour.height + 2 * offset, size[0])) - 1;
    return {
        id,
        x: p1,
        y: p3,
        width: p2 - p1,
        height: p4 - p3
    };
}

async function detectAndRecognizeText(imageElement) {
    const size = [imageElement.height, imageElement.width];
    const heatmapCanvas = await getHeatMapFromImage(imageElement);
    const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, size);
    
    // Draw bounding boxes on the preview canvas
    previewCanvas.width = imageElement.width;
    previewCanvas.height = imageElement.height;
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;

    let fullText = '';
    const crops = [];

    for (const box of boundingBoxes) {
        // Draw bounding box
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        const croppedCanvas = document.createElement('canvas');
        croppedCanvas.width = box.width;
        croppedCanvas.height = box.height;
        croppedCanvas.getContext('2d').drawImage(
            imageElement, 
            box.x, box.y, box.width, box.height,
            0, 0, box.width, box.height
        );

        const croppedImg = new Image();
        croppedImg.src = croppedCanvas.toDataURL('image/jpeg');
        await croppedImg.decode();
        crops.push(croppedImg);
    }

    let mean = scalar(255 * REC_MEAN);
    let std = scalar(255 * REC_STD);

    // Process crops in batches of 32
    const batchSize = 32;
    for (let i = 0; i < crops.length; i += batchSize) {
        const batch = crops.slice(i, i + batchSize);
        const inputTensors = await Promise.all(batch.map(crop => preprocessImageForRecognition(crop)));
        const inputTensorBatch = tf.concat(inputTensors);

        const predictions = await recognitionModel.executeAsync(inputTensorBatch.sub(mean).div(std));
        const probabilities = tf.softmax(predictions, -1);
        const bestPath = tf.unstack(tf.argMax(probabilities, -1), 0);
        
        const batchText = decodeText(bestPath);
        fullText += batchText + ' ';

        tf.dispose([inputTensorBatch, predictions, probabilities, ...bestPath]);
    }
    
    return fullText.trim();
}

function handleCapture() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    
    imageDataUrl = canvas.toDataURL('image/jpeg');
    resultElement.textContent = 'Processing image...';
    
    const img = new Image();
    img.src = imageDataUrl;
    img.onload = async () => {
        try {
            extractedText = await detectAndRecognizeText(img);
            resultElement.textContent = `Extracted Text: ${extractedText}`;
            
            // Show preview canvas and confirmation buttons
            previewCanvas.style.display = 'block';
            confirmButton.style.display = 'inline-block';
            retryButton.style.display = 'inline-block';
            captureButton.style.display = 'none';
        } catch (error) {
            console.error('Error during text extraction:', error);
            resultElement.textContent = 'Error occurred during text extraction';
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
                extractetAt: msgKey,
                data: extractedText,
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
    await loadModels();
    await loadOpenCV();
    await setupCamera();
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
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
