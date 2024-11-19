importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

self.onmessage = async (event) => {
    const { batch, recMean, recStd, vocab } = event.data;

    // Load the recognition model
    const recognitionModel = await tf.loadGraphModel('models/crnn_mobilenet_v2/model.json');

    // Preprocess the images
    const inputTensor = preprocessImageForRecognition(batch, recMean, recStd);

    // Run inference
    const predictions = await recognitionModel.executeAsync(inputTensor);
    const probabilities = tf.softmax(predictions, -1);
    const bestPath = tf.unstack(tf.argMax(probabilities, -1), 0);

    // Decode the text
    const words = decodeText(bestPath, vocab);

    // Clean up
    tf.dispose([inputTensor, predictions, probabilities, ...bestPath]);

    // Send the result back to the main thread
    self.postMessage({ words });
};

function preprocessImageForRecognition(crops, recMean, recStd) {
    const targetSize = [32, 128];
    const tensors = crops.map((crop) => {
        let h = crop.height;
        let w = crop.width;
        let resizeTarget, paddingTarget;
        let aspectRatio = targetSize[1] / targetSize[0];
        if (aspectRatio * h > w) {
            resizeTarget = [targetSize[0], Math.round((targetSize[0] * w) / h)];
            paddingTarget = [
                [0, 0],
                [0, targetSize[1] - Math.round((targetSize[0] * w) / h)],
                [0, 0],
            ];
        } else {
            resizeTarget = [Math.round((targetSize[1] * h) / w), targetSize[1]];
            paddingTarget = [
                [0, targetSize[0] - Math.round((targetSize[1] * h) / w)],
                [0, 0],
                [0, 0],
            ];
        }
        return tf.tidy(() => {
            return tf.browser
                .fromPixels(crop)
                .resizeNearestNeighbor(resizeTarget)
                .pad(paddingTarget, 0)
                .toFloat()
                .expandDims();
        });
    });
    const tensor = tf.concat(tensors);
    let mean = tf.scalar(255 * recMean);
    let std = tf.scalar(255 * recStd);
    return tensor.sub(mean).div(std);
}

function decodeText(bestPath, vocab) {
    const blank = 126;
    let collapsed = "";
    let lastChar = null;

    for (const sequence of bestPath) {
        const values = sequence.dataSync();
        for (const k of values) {
            if (k !== blank && k !== lastChar) {         
                collapsed += vocab[k];
                lastChar = k;
            } else if (k === blank) {
                lastChar = null;
            }
        }
        collapsed += ' ';
    }
    return collapsed.trim();
}
