const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('captureButton');
const actionButtons = document.getElementById('actionButtons');
const sendButton = document.getElementById('sendButton');
const discardButton = document.getElementById('discardButton');
const resultElement = document.getElementById('result');
const apiResponseElement = document.getElementById('apiResponse');

let model;
let extractedText = '';

async function initializeModel() {
    model = await tf.loadGraphModel('https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1', { fromTFHub: true });
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

captureButton.addEventListener('click', async () => {
    if (!model) {
        console.error('Model not loaded yet');
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageTensor = tf.browser.fromPixels(canvas);
    const resized = tf.image.resizeBilinear(imageTensor, [300, 300]);
    const casted = resized.cast('int32');
    const expanded = casted.expandDims(0);
    
    try {
        const predictions = await model.executeAsync(expanded);
        const boxes = await predictions[1].array();
        const classes = await predictions[2].array();
        const scores = await predictions[4].array();

        // Filter for text detections (class 73 is 'book' which often contains text)
        const textDetections = boxes[0].filter((box, i) => classes[0][i] === 73 && scores[0][i] > 0.5);
        
        extractedText = `Detected ${textDetections.length} potential text areas`;
        resultElement.textContent = extractedText;
        toggleButtons(true);
    } catch (error) {
        console.error('Error during object detection:', error);
        resultElement.textContent = 'Error occurred during object detection';
    } finally {
        imageTensor.dispose();
        resized.dispose();
        casted.dispose();
        expanded.dispose();
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
    extractedText = '';
}

async function init() {
    await setupCamera();
    await initializeModel();
}

init();

// Service Worker Registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/service-worker.js')
            .then(registration => {
                console.log('ServiceWorker registration successful with scope: ', registration.scope);
            }, err => {
                console.log('ServiceWorker registration failed: ', err);
            });
    });
}
