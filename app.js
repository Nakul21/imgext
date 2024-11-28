// textDetectionWorker.js - No changes needed as it doesn't interact with UI

// Modified app.js with UI safeguards
let textDetectionWorker;
let modelLoadingPromise;

// UI State Management
const UIState = {
    INITIALIZING: 'initializing',
    READY: 'ready',
    PROCESSING: 'processing',
    RESULTS: 'results',
    ERROR: 'error'
};

function updateUIState(state, data = {}) {
    switch (state) {
        case UIState.INITIALIZING:
            showLoading('Initializing...');
            if (captureButton) captureButton.disabled = true;
            break;

        case UIState.READY:
            hideLoading();
            if (captureButton) {
                captureButton.disabled = false;
                captureButton.textContent = 'Capture';
                captureButton.style.display = 'block';
            }
            if (previewCanvas) previewCanvas.style.display = 'none';
            if (confirmButton) confirmButton.style.display = 'none';
            if (retryButton) retryButton.style.display = 'none';
            if (actionButtons) actionButtons.style.display = 'none';
            break;

        case UIState.PROCESSING:
            showLoading('Processing image...');
            if (captureButton) {
                captureButton.disabled = true;
                captureButton.textContent = 'Processing...';
            }
            break;

        case UIState.RESULTS:
            hideLoading();
            if (resultElement) resultElement.textContent = `Extracted Text: ${data.text || ''}`;
            if (previewCanvas) {
                previewCanvas.style.display = 'block';
                // Draw bounding boxes if provided
                if (data.boundingBoxes) {
                    const ctx = previewCanvas.getContext('2d');
                    drawBoundingBoxes(ctx, data.boundingBoxes);
                }
            }
            if (confirmButton) confirmButton.style.display = 'inline-block';
            if (retryButton) retryButton.style.display = 'inline-block';
            if (captureButton) captureButton.style.display = 'none';
            break;

        case UIState.ERROR:
            hideLoading();
            if (resultElement) resultElement.textContent = data.error || 'An error occurred';
            if (captureButton) {
                captureButton.disabled = false;
                captureButton.textContent = 'Capture';
            }
            break;
    }
}

function drawBoundingBoxes(ctx, boxes) {
    ctx.strokeStyle = '#FFD700'; // Using your CSS primary color
    ctx.lineWidth = 2;
    boxes.forEach(box => {
        const { x, y, width, height } = box.bbox;
        ctx.strokeRect(x, y, width, height);
    });
}

async function initializeWorker() {
    updateUIState(UIState.INITIALIZING);
    
    textDetectionWorker = new Worker('textDetectionWorker.js');
    
    return new Promise((resolve, reject) => {
        textDetectionWorker.onmessage = function(e) {
            const { type, error, results } = e.data;
            
            switch (type) {
                case 'initialized':
                    updateUIState(UIState.READY);
                    resolve();
                    break;
                    
                case 'error':
                    updateUIState(UIState.ERROR, { error: error });
                    reject(new Error(error));
                    break;
                    
                case 'results':
                    updateUIState(UIState.RESULTS, {
                        text: results.extractedText,
                        boundingBoxes: results.extractedData
                    });
                    break;
            }
        };
        
        textDetectionWorker.onerror = function(error) {
            updateUIState(UIState.ERROR, { error: error.message });
            reject(error);
        };
        
        textDetectionWorker.postMessage({ type: 'init' });
    });
}

async function handleCapture() {
    updateUIState(UIState.PROCESSING);

    const targetSize = TARGET_SIZE;
    if (!canvas || !video) return;

    canvas.width = targetSize[0];
    canvas.height = targetSize[1];
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    imageDataUrl = canvas.toDataURL('image/jpeg');
    
    const img = new Image();
    img.src = imageDataUrl;
    
    img.onload = async () => {
        try {
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            textDetectionWorker.postMessage({
                type: 'process',
                data: { imageData }
            });
        } catch (error) {
            updateUIState(UIState.ERROR, { 
                error: 'Error during text extraction' 
            });
        }
    };
}

// Modified event listeners to maintain UI state
confirmButton?.addEventListener('click', () => {
    toggleButtons(true);
    if (previewCanvas) previewCanvas.style.display = 'none';
    if (confirmButton) confirmButton.style.display = 'none';
    if (retryButton) retryButton.style.display = 'none';
});

retryButton?.addEventListener('click', () => {
    updateUIState(UIState.READY);
    resetUI();
});

// Keep your existing utility functions but add null checks
function showLoading(message) {
    if (loadingIndicator) {
        loadingIndicator.textContent = message;
        loadingIndicator.style.display = 'block';
    }
}

function hideLoading() {
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }
}

// Initialize the application
async function init() {
    try {
        await initializeWorker();
        await setupCamera();
        updateUIState(UIState.READY);
    } catch (error) {
        updateUIState(UIState.ERROR, { 
            error: 'Error initializing application. Please refresh.' 
        });
    }
}

// Clean up
window.addEventListener('unload', () => {
    if (textDetectionWorker) {
        textDetectionWorker.terminate();
    }
});
