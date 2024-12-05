self.HTMLImageElement = class HTMLImageElement {
    constructor() {
      this.data = null;
      this.width = 0;
      this.height = 0;
    }
  };
  
  self.HTMLCanvasElement = class HTMLCanvasElement {
    constructor() {
      this.data = null;
      this.width = 0;
      this.height = 0;
    }
  };
  
  // Constants
  const REC_MEAN = 0.694;
  const REC_STD = 0.298;
  const DET_MEAN = 0.785;
  const DET_STD = 0.275;
  const VOCAB =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
  
  let detectionModel;
  let recognitionModel;
  let cvReady = false;
  
  async function initializeLibraries() {
    try {
      await loadOpenCV();
      importScripts(
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js"
      );
      return true;
    } catch (error) {
      console.error("Initialization failed:", error);
      throw error;
    }
  }
  
  function loadOpenCV() {
    return new Promise((resolve, reject) => {
      self.Module = {
        onRuntimeInitialized: () => {
          cvReady = true;
          resolve();
        },
      };
  
      importScripts("./opencv/opencv.js");
  
      setTimeout(() => {
        if (!cvReady) reject(new Error("OpenCV initialization timeout"));
      }, 10000);
    });
  }
  
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
  
  function preprocessImageForDetection(imageElement) {
    const targetSize = [512, 512];
    let tensor = tf.browser
      .fromPixels(imageElement)
      .resizeNearestNeighbor(targetSize)
      .toFloat();
    return tensor
      .sub(tf.scalar(255 * DET_MEAN))
      .div(tf.scalar(255 * DET_STD))
      .expandDims();
  }
  
  async function getHeatMapFromImage(imageObject) {
    try {
      const width = Math.floor(Number(imageObject.width)) || 512;
      const height = Math.floor(Number(imageObject.height)) || 512;
  
      let tensor = preprocessImageForDetection(imageObject);
      let prediction = await detectionModel.execute(tensor);
      prediction = tf.squeeze(prediction, 0);
      prediction = Array.isArray(prediction) ? prediction[0] : prediction;
  
      const heatmapCanvas = new OffscreenCanvas(width, height);
      await tf.browser.toPixels(prediction, heatmapCanvas);
  
      tensor.dispose();
      prediction.dispose();
  
      return heatmapCanvas;
    } catch (error) {
      throw new Error(`Heat map generation failed: ${error.message}`);
    }
  }
  
  function transformBoundingBox(contour, id, size) {
    let offset =
      (contour.width * contour.height * 1.8) /
      (2 * (contour.width + contour.height));
    const x1 = Math.max(0, contour.x - offset) / size[1];
    const x2 = Math.min(size[1], contour.x + contour.width + offset) / size[1];
    const y1 = Math.max(0, contour.y - offset) / size[0];
    const y2 = Math.min(size[0], contour.y + contour.height + offset) / size[0];
  
    return {
      id,
      config: { stroke: "#ff0000" },
      coordinates: [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
      ],
    };
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
          const offset = (contourBoundingBox.width * contourBoundingBox.height * 1.8) / 
                        (2 * (contourBoundingBox.width + contourBoundingBox.height));
          
          const x1 = Math.max(0, contourBoundingBox.x - offset) / size[1];
          const x2 = Math.min(size[1], contourBoundingBox.x + contourBoundingBox.width + offset) / size[1];
          const y1 = Math.max(0, contourBoundingBox.y - offset) / size[0];
          const y2 = Math.min(size[0], contourBoundingBox.y + contourBoundingBox.height + offset) / size[0];
    
          boundingBoxes.unshift({
            id: i,
            config: { stroke: '#ff0000' },
            coordinates: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
          });
        }
      }
    
      src.delete();
      contours.delete();
      hierarchy.delete();
      return boundingBoxes;
    }
  
  async function createCrops(imageBitmap, boundingBoxes) {
    return Promise.all(
      boundingBoxes.map(async (box) => {
        const width = Math.max(1, Math.floor(box.width));
        const height = Math.max(1, Math.floor(box.height));
        const cropCanvas = new OffscreenCanvas(width, height);
        const ctx = cropCanvas.getContext("2d");
  
        try {
          ctx.drawImage(
            imageBitmap,
            Math.floor(box.x),
            Math.floor(box.y),
            width,
            height,
            0,
            0,
            width,
            height
          );
          return cropCanvas;
        } catch (error) {
          console.error("Error creating crop:", error);
        }
      })
    ).then((crops) => crops.filter(Boolean));
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
      collapsed += " ";
    }
    return collapsed.trim();
  }
  
  async function recognizeText(crops) {
    return Promise.all(
      crops.map(async (crop) => {
        let tensor = tf.browser
          .fromPixels(crop)
          .resizeNearestNeighbor([32, 128])
          .expandDims(0)
          .toFloat()
          .div(255.0)
          .sub(REC_MEAN)
          .div(REC_STD);
  
        const prediction = await recognitionModel.executeAsync(tensor);
        const text = decodeText(prediction.arraySync()[0]);
  
        tensor.dispose();
        prediction.dispose();
  
        return text;
      })
    );
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
    let mean = tf.scalar(255 * REC_MEAN);
    let std = tf.scalar(255 * REC_STD);
    return tensor.sub(mean).div(std);
  }
  
  async function processImage(imageData) {
    const size = [512, 512];
    const img = await createImageBitmap(imageData);
    const heatmapCanvas = await getHeatMapFromImage(imageData);
    const boundingBoxes = extractBoundingBoxesFromHeatmap(heatmapCanvas, size);
    console.log("extractBoundingBoxesFromHeatmap", boundingBoxes);
  
    self.postMessage({
      type: "BOUNDING_BOXES",
      data: { boundingBoxes },
    });
  
    if (!boundingBoxes?.length) return { text: [], boundingBoxes: [] };
  
    let fullText = "";
      const crops = [];
  
  //   const crops = await createCrops(img, boundingBoxes);
    for (const box of boundingBoxes) {
        // Draw bounding box
        const [x1, y1] = box.coordinates[0];
          const [x2, y2] = box.coordinates[2];
        const width = (x2 - x1) * imageData.width;
        const height = (y2 - y1) * imageData.height;
        const x = x1 * imageData.width;
        const y = y1 * imageData.height;
  
        // ctx.strokeStyle = box.config.stroke;
        // ctx.lineWidth = 2;
        // ctx.strokeRect(x, y, width, height);
  
        // Create crop
        const cropCanvas = new OffscreenCanvas(width, height);
        const ctx = cropCanvas.getContext("2d");
        ctx
        .drawImage(img, x, y, width, height, 0, 0, width, height);
  
        crops.push(cropCanvas);
    }
  
    //   const extractedText = await recognizeText(crops);
  
    // Process crops in batches
  
    const batchSize = 32;
    for (let i = 0; i < crops.length; i += batchSize) {
      const batch = crops.slice(i, i + batchSize);
      const inputTensor = preprocessImageForRecognition(batch);
  
      const predictions = await recognitionModel.executeAsync(inputTensor);
      const probabilities = tf.softmax(predictions, -1);
      const bestPath = tf.unstack(tf.argMax(probabilities, -1), 0);
  
      const words = decodeText(bestPath);
      fullText += words + " ";
  
      tf.dispose([inputTensor, predictions, probabilities, ...bestPath]);
    }
    img.close();
    console.log("fullText ", fullText, "boundingBoxes ", boundingBoxes);
    return { text: fullText.trim(), boundingBoxes };
  }
  
  self.onmessage = async function (e) {
    try {
      await initializeLibraries();
  
      switch (e.data.type) {
        case "LOAD_MODELS":
          await loadModels();
          self.postMessage({ type: "MODELS_LOADED" });
          break;
        case "PROCESS_IMAGE":
          const result = await processImage(e.data.data.imageData);
          self.postMessage({ type: "PROCESS_COMPLETE", data: result });
          break;
      }
    } catch (error) {
      self.postMessage({ type: "ERROR", error: error.message });
    }
  };
  