// worker.js
// This handles the heavy computational tasks
// importScripts("https://cdn.jsdelivr.net/npm/opencv@7.0.0/lib/opencv.min.js");
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js");

let detectionModel;
let recognitionModel;

// Create a promise to track OpenCV initialization
let cvReady = new Promise((resolve) => {
  // OpenCV will call this function when it's ready
  self.Module = {
    onRuntimeInitialized: () => {
      console.log("OpenCV runtime initialized");
      resolve(true);
    },
  };
});

// We'll wrap our library imports in a function to ensure proper order
async function initializeLibraries() {
  try {
    console.log("Starting library initialization...");

    // Import OpenCV first and wait for it to be ready
    importScripts("./opencv/opencv.js");
    await cvReady;
    console.log("OpenCV loaded and initialized");

    // Now import TensorFlow
    importScripts(
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js"
    );
    console.log("TensorFlow.js loaded");

    // Verify OpenCV is working properly
    if (typeof cv === "undefined") {
      throw new Error("OpenCV object is not defined after initialization");
    }

    // Test basic OpenCV functiona

    return true;
  } catch (error) {
    console.error("Library initialization failed:", error);
    throw error;
  }
}

// Constants moved from main file
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const VOCAB =
  "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

// Initialize OpenCV with proper callback handling
function loadOpenCV() {
  return new Promise((resolve, reject) => {
    // First, import the script
    importScripts(
      "https://cdn.jsdelivr.net/npm/opencv@7.0.0/lib/opencv.min.js"
    );

    // Check if cv is already available
    if (typeof cv !== "undefined" && cv.imread) {
      cvReady = true;
      resolve();
      return;
    }

    // If not immediately available, wait for it
    const checkCV = () => {
      if (typeof cv !== "undefined" && cv.imread) {
        cvReady = true;
        resolve();
      } else {
        // Check again in 100ms
        setTimeout(checkCV, 100);
      }
    };

    // Start checking
    checkCV();

    // Set a timeout to avoid infinite waiting
    setTimeout(() => {
      if (!cvReady) {
        reject(new Error("OpenCV failed to initialize after 10 seconds"));
      }
    }, 10000);
  });
}

async function getHeatMapFromImage(imageObject) {
  try {
    console.log("Worker: Starting heat map generation");

    // Preprocess the image
    let tensor = preprocessImageForDetection(imageObject);
    console.log("Worker: Image preprocessed");

    // Run detection model
    let prediction = await detectionModel.execute(tensor);
    prediction = tf.squeeze(prediction, 0);

    if (Array.isArray(prediction)) {
      prediction = prediction[0];
    }

    // Create OffscreenCanvas instead of regular canvas
    const heatmapCanvas = new OffscreenCanvas(
      imageObject.width,
      imageObject.height
    );
    await tf.browser.toPixels(prediction, heatmapCanvas);

    // Clean up tensors
    tensor.dispose();
    prediction.dispose();

    console.log("Worker: Heat map generated successfully");
    return heatmapCanvas;
  } catch (error) {
    console.error("Worker: Error in getHeatMapFromImage:", error);
    throw new Error(`Heat map generation failed: ${error.message}`);
  }
}

function preprocessImageForDetection(imageElement) {
  const targetSize = [512, 512];
  let tensor = tf.browser
    .fromPixels(imageElement)
    .resizeNearestNeighbor(targetSize)
    .toFloat();
  let mean = tf.scalar(255 * DET_MEAN);
  let std = tf.scalar(255 * DET_STD);
  return tensor.sub(mean).div(std).expandDims();
}

function extractBoundingBoxesFromHeatmap(heatmapCanvas, size) {
  let src = cv.imread(heatmapCanvas);
  cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
  cv.threshold(src, src, 77, 255, cv.THRESH_BINARY);
  cv.morphologyEx(src, src, cv.MORPH_OPEN, cv.Mat.ones(2, 2, cv.CV_8U));
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(
    src,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  );

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

// Handle messages from main thread
self.onmessage = async function (e) {
  const { type, data } = e.data;

  try {
    // Initialize libraries if this is the first message
    await initializeLibraries();

    switch (type) {
      case "LOAD_MODELS":
        await loadModels();
        self.postMessage({ type: "MODELS_LOADED" });
        break;

      case "PROCESS_IMAGE":
        const result = await processImage(data.imageData);
        self.postMessage({
          type: "PROCESS_COMPLETE",
          data: result,
        });
        break;
    }
  } catch (error) {
    console.error("Worker error:", error);
    self.postMessage({
      type: "ERROR",
      error: error.message,
    });
  }
};

// Model loading function
async function loadModels() {
  try {
    detectionModel = await tf.loadGraphModel(
      "models/db_mobilenet_v2/model.json"
    );
    recognitionModel = await tf.loadGraphModel(
      "models/crnn_mobilenet_v2/model.json"
    );
  } catch (error) {
    throw new Error("Failed to load models: " + error.message);
  }
}

// Main image processing function
async function processImage(imageData) {
  const img = await createImageBitmap(imageData);
  const heatmapCanvas = await getHeatMapFromImage(img);
  const boundingBoxes = extractBoundingBoxesFromHeatmap(
    heatmapCanvas,
    [512, 512]
  );

  const crops = await createCrops(img, boundingBoxes);
  const extractedText = await recognizeText(crops);

  return {
    text: extractedText,
    boundingBoxes,
  };
}
