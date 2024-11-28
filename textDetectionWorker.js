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

async function detectAndRecognizeText(imageData) {
    try {
        const tensor = preprocessImageForDetection(imageData);
        const detection = await detectionModel.execute(tensor);
        const boxes = extractBoundingBoxes(detection);
        
        const results = await recognizeText(boxes, imageData);
        
        tf.dispose([tensor, detection]);
        return results;
    } catch (error) {
        throw error;
    }
}

// Worker message handling
self.onmessage = async function(e) {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            try {
                await tf.ready();
                await tf.setBackend('cpu'); // Use CPU in worker
                
                // Load models
                detectionModel = await tf.loadGraphModel('models/db_mobilenet_v2/model.json');
                recognitionModel = await tf.loadGraphModel('models/crnn_mobilenet_v2/model.json');
                
                self.postMessage({ type: 'initialized' });
            } catch (error) {
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
                self.postMessage({ type: 'error', error: error.message });
            }
            break;
    }
};
