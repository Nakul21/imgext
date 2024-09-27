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
    model = await textRecognition.load();
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
    const preprocessedImage = tf.image.resizeBilinear(imageTensor, [320, 320]).div(tf.scalar(255));
    
    try {
        const result = await model.recognize(preprocessedImage);
        extractedText = result.join(' ');
        resultElement.textContent = extractedText;
        toggleButtons(true);
    } catch (error) {
        console.error('Error during text recognition:', error);
        resultElement.textContent = 'Error occurred during text recognition';
    } finally {
        imageTensor.dispose();
        preprocessedImage.dispose();
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
