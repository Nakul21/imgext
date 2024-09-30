// Constants
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('captureButton');
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
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
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
    img = tf.image.resizeBilinear(img, targetSize);
    img = img.toFloat();
    let mean = tf.scalar(255 * DET_MEAN);
    let std = tf.scalar(255 * DET_STD);
    img = img.sub(mean).div(std);
    return img.expandDims(0);
}

async function preprocessImageForRecognition(imageElement) {
    const targetSize = [32, 128];
    let img = tf.browser.fromPixels(imageElement);
    img = tf.image.resizeBilinear(img, targetSize);
    img = img.toFloat();
    let mean = tf.scalar(255 * REC_MEAN);
    let std = tf.scalar(255 * REC_STD);
    img = img.sub(mean).div(std);
    return img.expandDims(0);
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
    let prediction = await detectionModel.predict(tensor);
    prediction = tf.squeeze(prediction);
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
            boundingBoxes.push(transformBoundingBox(contourBoundingBox, i, size));
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
    
    let fullText = '';
    for (const box of boundingBoxes) {
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

        const inputTensor = await preprocessImageForRecognition(croppedImg);
        const predictions = await recognitionModel.predict(inputTensor);
        const probabilities = tf.softmax(predictions, -1);
        const bestPath = tf.unstack(tf.argMax(probabilities, -1), 0);
        
        const text = decodeText(bestPath);
        fullText += text + ' ';

        tf.dispose([inputTensor, predictions, probabilities, ...bestPath]);
    }
    
    return fullText.trim();
}

captureButton.addEventListener('click', async () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    
    imageDataUrl = canvas.toDataURL('image/jpeg');
    resultElement.textContent = 'Processing image...';
    
    try {
        const img = new Image();
        img.src = imageDataUrl;
        await img.decode();

        extractedText = await detectAndRecognizeText(img);
        resultElement.textContent = `Extracted Text: ${extractedText}`;
        toggleButtons(true);

    } catch (error) {
        console.error('Error during text extraction:', error);
        resultElement.textContent = 'Error occurred during text extraction';
    }
});

sendButton.addEventListener('click', async () => {
    try {
        const response = await fetch('https://api.example.com/text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: extractedText }),
        });
        const data = await response.json();
        apiResponseElement.textContent = `API Response: ${JSON.stringify(data)}`;
    } catch (error) {
        console.error('Error sending data to API:', error);
        apiResponseElement.textContent = 'Error sending data to API';
    }
    resetUI();
});

discardButton.addEventListener('click', resetUI);

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
}

function clearCanvas() {
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

async function init() {
    await setupCamera();
    await loadModels();
    await loadOpenCV();
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
