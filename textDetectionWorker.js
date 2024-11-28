// Import required libraries in worker
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js');
importScripts('https://docs.opencv.org/4.5.2/opencv.js');

// Constants
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
const TARGET_SIZE = [512, 512];
const BASE_PATH = '/imgext/web-workers-impl/';

let detectionModel;
let recognitionModel;

// Helper functions
function preprocessImageForDetection(imageData) {
    let tensor = tf.tidy(() => {
        return tf.browser.fromPixels(imageData)
            .resizeNearestNeighbor(TARGET_SIZE)
            .toFloat();
    });
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
            paddingTarget = [[0, 0], [0, targetSize[1] - Math.round((targetSize[0] * w) / h)], [0, 0]];
        } else {
            resizeTarget = [Math.round((targetSize[1] * h) / w), targetSize[1]];
            paddingTarget = [[0, targetSize[0] - Math.round((targetSize[1] * h) / w)], [0, 0], [0, 0]];
        }
        return tf.tidy(() => {
            return tf.browser.fromPixels(crop)
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

function getRandomColor() {
    return '#' + Math.floor(Math.random()*16777215).toString(16);
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

async function detectAndRecognizeText(imageData) {
    try {
        // Set to CPU backend for consistent processing
        await tf.setBackend('cpu');
        
        // Detection phase
        const tensor = preprocessImageForDetection(imageData);
        const detection = await detectionModel.execute(tensor);
        const squeezed = tf.squeeze(detection, 0);
        
        // Get heatmap and extract bounding boxes
        const heatmapCanvas = document.createElement('canvas');
        heatmapCanvas.width = imageData.width;
        heatmapCanvas.height = imageData.height;
        await tf.browser.toPixels(squeezed, heatmapCanvas);
        
        const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, TARGET_SIZE);
        
        // Process each box for recognition
        const crops = [];
        for (const box of boundingBoxes) {
            const [x1, y1] = box.coordinates[0];
            const [x2, y2] = box.coordinates[2];
            const width = (x2 - x1) * imageData.width;
            const height = (y2 - y1) * imageData.height;
            const x = x1 * imageData.width;
            const y = y1 * imageData.height;

            const croppedCanvas = document.createElement('canvas');
            croppedCanvas.width = Math.min(width, 128);
            croppedCanvas.height = Math.min(height, 32);
            croppedCanvas.getContext('2d').drawImage(
                imageData,
                x, y, width, height,
                0, 0, width, height
            );

            crops.push({
                canvas: croppedCanvas,
                bbox: {
                    x: Math.round(x),
                    y: Math.round(y),
                    width: Math.round(width),
                    height: Math.round(height)
                }
            });
        }

        // Recognition phase
        const results = [];
        const batchSize = 8;
        for (let i = 0; i < crops.length; i += batchSize) {
            const batch = crops.slice(i, i + batchSize);
            const inputTensor = preprocessImageForRecognition(batch.map(crop => crop.canvas));
            
            const predictions = await recognitionModel.executeAsync(inputTensor);
            const probabilities = tf.softmax(predictions, -1);
            const bestPath = tf.unstack(tf.argMax(probabilities, -1), 0);
            
            const words = decodeText(bestPath);
            
            words.split(' ').forEach((word, index) => {
                if (word && batch[index]) {
                    results.push({
                        word: word,
                        boundingBox: batch[index].bbox
                    });
                }
            });

            tf.dispose([inputTensor, predictions, probabilities, ...bestPath]);
        }

        // Cleanup
        tf.dispose([tensor, detection, squeezed]);
        return results;
    } catch (error) {
        console.error('Error in detectAndRecognizeText:', error);
        throw error;
    } finally {
        tf.disposeVariables();
    }
}

// Worker message handling
self.onmessage = async function(e) {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            try {
                await tf.ready();
                await tf.setBackend('cpu');
                
                // Load models with correct path
                detectionModel = await tf.loadGraphModel(BASE_PATH + 'models/db_mobilenet_v2/model.json');
                recognitionModel = await tf.loadGraphModel(BASE_PATH + 'models/crnn_mobilenet_v2/model.json');
                
                console.log('Models loaded successfully');
                self.postMessage({ type: 'initialized' });
            } catch (error) {
                console.error('Initialization error:', error);
                self.postMessage({ type: 'error', error: error.message });
            }
            break;

        case 'process':
            try {
                const results = await detectAndRecognizeText(data.imageData);
                self.postMessage({ 
                    type: 'results', 
                    results: {
                        extractedData: results,
                        extractedText: results.map(item => item.word).join(' ')
                    }
                });
            } catch (error) {
                console.error('Processing error:', error);
                self.postMessage({ type: 'error', error: error.message });
            }
            break;
    }
};
