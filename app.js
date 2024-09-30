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
let recognizer;

function initDocTR() {
    loadModel();
}

async function loadModel() {
    try {
        recognizer = await tf.loadGraphModel('models/crnn_mobilenet_v2/model.json');
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Call this function after the page has loaded
window.addEventListener('load', initDocTR);

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function preprocessImage(imageElement) {
    try {
        let img = tf.browser.fromPixels(imageElement).toFloat();
        // Adjust the size to match what the model expects
        img = tf.image.resizeBilinear(img, [32, 128]);
        const offset = tf.scalar(127.5);
        const normalized = img.sub(offset).div(offset);
        // Reshape to [1, 32, 128, 3] to add the batch dimension
        const batched = normalized.reshape([1, 32, 128, 3]);
        return batched;
    } catch (error) {
        resultElement.textContent = `Error in image preprocessing: ${error}`;
        throw error;
    }
}

function decodeText(predictions) {
    // Assuming the charset order matches the model's output
    const charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
    let text = '';
    
    // Check if predictions is a 2D array
    if (Array.isArray(predictions[0])) {
        for (let i = 0; i < predictions.length; i++) {
            let maxIndex = predictions[i].indexOf(Math.max(...predictions[i]));
            if (maxIndex < charset.length) {
                text += charset[maxIndex];
            }
        }
    } else {
        // If it's a 1D array, assume it's for a single character
        let maxIndex = predictions.indexOf(Math.max(...predictions));
        if (maxIndex < charset.length) {
            text = charset[maxIndex];
        }
    }
    
    return text;
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

        const inputTensor = await preprocessImage(img);
        
        // Use executeAsync and provide a named input
        const predictions = await recognizer.executeAsync({'x': inputTensor});
        
        // Process the predictions
        let outputArray;
        if (Array.isArray(predictions)) {
            outputArray = predictions.map(tensor => tensor.arraySync());
        } else {
            outputArray = await predictions.array();
        }
        
        // Log the output array for debugging
        console.log('Raw model output:', outputArray);
        
        // Decode the output array into text
        extractedText = decodeText(outputArray[0]); // Note: we're passing outputArray[0] here
        
        resultElement.textContent = `Extracted Text: ${extractedText}`;
        toggleButtons(true);

        // Don't forget to dispose of the tensors to free up memory
        tf.dispose([inputTensor, predictions]);
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
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
}

async function init() {
    await setupCamera();
    await loadModel();
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
