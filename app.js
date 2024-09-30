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
  return words;
}

async function detectTextRegions(imageElement) {
    const inputTensor = await preprocessImageForDetection(imageElement);
    const predictions = await detectionModel.executeAsync(inputTensor);
    
    console.log('Detection model output:', predictions);

    let boxes;
    if (Array.isArray(predictions) && predictions.length > 0) {
        boxes = await extractBoundingBoxes(predictions[0]);
        tf.dispose([inputTensor, ...predictions]);
    } else if (predictions instanceof tf.Tensor) {
        boxes = await extractBoundingBoxes(predictions);
        tf.dispose([inputTensor, predictions]);
    } else {
        console.error('Unexpected output from detection model:', predictions);
        boxes = [];
    }

    
    return boxes;
}

async function extractBoundingBoxes(prediction, threshold = 0.3) {
    if (!prediction || !prediction.shape) {
        console.error('Invalid prediction tensor');
        return [];
    }
    
    const [height, width] = prediction.shape.slice(1, 3);
    const data = await prediction.array();
    const boxes = [];

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (data[0][y][x][0] > threshold) {
                boxes.push({x, y, width: 1, height: 1});
            }
        }
    }

    return mergeBoundingBoxes(boxes);
}

function mergeBoundingBoxes(boxes, threshold = 5) {
    const merged = [];
    for (const box of boxes) {
        let shouldMerge = false;
        for (const mergedBox of merged) {
            if (
                Math.abs(box.x - mergedBox.x) < threshold &&
                Math.abs(box.y - mergedBox.y) < threshold
            ) {
                mergedBox.x = Math.min(mergedBox.x, box.x);
                mergedBox.y = Math.min(mergedBox.y, box.y);
                mergedBox.width = Math.max(mergedBox.width, box.width);
                mergedBox.height = Math.max(mergedBox.height, box.height);
                shouldMerge = true;
                break;
            }
        }
        if (!shouldMerge) {
            merged.push({...box});
        }
    }
    return merged;
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

        // Detection step
        const boundingBoxes = await detectTextRegions(img);

        console.log('Detected bounding boxes:', boundingBoxes);

        // Recognition step
        let fullText = '';
        for (const box of boundingBoxes) {
            const croppedCanvas = document.createElement('canvas');
            croppedCanvas.width = box.width;
            croppedCanvas.height = box.height;
            croppedCanvas.getContext('2d').drawImage(
                img, 
                box.x, box.y, box.width, box.height,
                0, 0, box.width, box.height
            );

            const croppedImg = new Image();
            croppedImg.src = croppedCanvas.toDataURL('image/jpeg');
            await croppedImg.decode();

            const inputTensor = await preprocessImageForRecognition(croppedImg);
            const predictions = await recognitionModel.executeAsync(inputTensor);
            let probabilities = tf.softmax(predictions, -1);
            let bestPath = tf.unstack(tf.argMax(probabilities, -1), 0);
            
            const text = decodeText(bestPath);
            fullText += text + ' ';

            tf.dispose([inputTensor, predictions]);
        }
        
        extractedText = fullText.trim();
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
    captureButton.disabled = false;
    captureButton.textContent = 'Capture';
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
