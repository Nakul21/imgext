// Add this at the beginning of your main script
let textDetectionWorker;

async function initializeWorker() {
    textDetectionWorker = new Worker('textDetectionWorker.js');
    
    return new Promise((resolve, reject) => {
        textDetectionWorker.onmessage = function(e) {
            const { type, error, results } = e.data;
            
            switch (type) {
                case 'initialized':
                    resolve();
                    break;
                case 'error':
                    reject(new Error(error));
                    break;
            }
        };
        
        textDetectionWorker.postMessage({ type: 'init' });
    });
}

// Modify your handleCapture function
async function handleCapture() {
    disableCaptureButton();
    showLoading('Processing image...');

    const targetSize = TARGET_SIZE;
    canvas.width = targetSize[0];
    canvas.height = targetSize[1];
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    imageDataUrl = canvas.toDataURL('image/jpeg');
    
    const img = new Image();
    img.src = imageDataUrl;
    img.onload = async () => {
        try {
            // Get image data from canvas
            const ctx = canvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            // Process image using web worker
            textDetectionWorker.onmessage = function(e) {
                const { type, results, error } = e.data;
                
                if (type === 'error') {
                    console.error('Worker error:', error);
                    resultElement.textContent = 'Error occurred during text extraction';
                    return;
                }
                
                if (type === 'results') {
                    extractedData = results;
                    extractedText = results.map(item => item.word).join(' ');
                    resultElement.textContent = `Extracted Text: ${extractedText}`;
                    
                    // Draw bounding boxes on preview canvas
                    drawBoundingBoxes(results);
                    
                    previewCanvas.style.display = 'block';
                    confirmButton.style.display = 'inline-block';
                    retryButton.style.display = 'inline-block';
                    captureButton.style.display = 'none';
                }
            };

            textDetectionWorker.postMessage({
                type: 'process',
                data: {
                    imageData,
                    width: canvas.width,
                    height: canvas.height
                }
            });

        } catch (error) {
            console.error('Error during text extraction:', error);
            resultElement.textContent = 'Error occurred during text extraction';
        } finally {
            enableCaptureButton();
            hideLoading();
        }
    };
}

// Add worker cleanup on page unload
window.addEventListener('unload', () => {
    if (textDetectionWorker) {
        textDetectionWorker.terminate();
    }
});

// Modify your init function to include worker initialization
async function init() {
    showLoading('Initializing...');
    
    try {
        await initializeWorker();
        await setupCamera();
        
        captureButton.disabled = false;
        captureButton.textContent = 'Capture';
        
        hideLoading();
    } catch (error) {
        console.error('Initialization error:', error);
        showLoading('Error initializing application. Please refresh.');
    }
}