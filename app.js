// Constants
const REC_MEAN = 0.694;
const REC_STD = 0.298;
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const VOCAB =
  "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

// DOM Elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const previewCanvas = document.getElementById("previewCanvas");
const captureButton = document.getElementById("captureButton");
const confirmButton = document.getElementById("confirmButton");
const retryButton = document.getElementById("retryButton");
const actionButtons = document.getElementById("actionButtons");
const sendButton = document.getElementById("sendButton");
const discardButton = document.getElementById("discardButton");
const resultElement = document.getElementById("result");
const apiResponseElement = document.getElementById("apiResponse");

let imageDataUrl = "";
let extractedText = "";
let imageWorker;

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "environment",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

function handleConfirm() {
  toggleButtons(true);
  previewCanvas.style.display = "none";
  confirmButton.style.display = "none";
  retryButton.style.display = "none";
}

function handleRetry() {
  resetUI();
}

async function handleSend() {
  if (!extractedText) return;
  apiResponseElement.textContent = "Submitting...";
  let msgKey = new Date().getTime();
  try {
    const response = await fetch(
      "https://kvdb.io/NyKpFtJ7v392NS8ibLiofx/" + msgKey,
      {
        method: "PUT",
        body: JSON.stringify({
          extractetAt: msgKey,
          data: extractedText,
          userId: "imageExt",
        }),
        headers: {
          "Content-type": "application/json; charset=UTF-8",
        },
      }
    );

    if (response.status !== 200) {
      throw new Error("Failed to push this data to server");
    }

    apiResponseElement.textContent =
      "Submitted the extract with ID : " + msgKey;
  } catch (error) {
    console.error("Error submitting to server:", error);
    apiResponseElement.textContent =
      "Error occurred while submitting to server";
  } finally {
    resetUI();
  }
}

function toggleButtons(showActionButtons) {
  captureButton.style.display = showActionButtons ? "none" : "block";
  actionButtons.style.display = showActionButtons ? "block" : "none";
}

function resetUI() {
  toggleButtons(false);
  resultElement.textContent = "";
  apiResponseElement.textContent = "";
  imageDataUrl = "";
  extractedText = "";
  clearCanvas();
  previewCanvas.style.display = "none";
  confirmButton.style.display = "none";
  retryButton.style.display = "none";
  captureButton.style.display = "block";
}

function clearCanvas() {
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  previewCanvas
    .getContext("2d")
    .clearRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function loadOpenCV() {
  return new Promise((resolve) => {
    const script = document.createElement("script");
    script.src = "https://docs.opencv.org/4.5.2/opencv.js";
    script.onload = () => resolve();
    document.body.appendChild(script);
  });
}

function displayBoundingBoxes(boxes) {
  previewCanvas.style.display = "block";
  const ctx = previewCanvas.getContext("2d");
  const image = new Image();

  image.onload = () => {
    previewCanvas.width = image.width;
    previewCanvas.height = image.height;
    ctx.drawImage(image, 0, 0);

    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    boxes.forEach((box) => {
      ctx.strokeRect(box.x, box.y, box.width, box.height);
    });
  };

  image.src = imageDataUrl;
}

async function initializeWorker() {
  imageWorker = new Worker("worker.js");

  imageWorker.onmessage = function (e) {
    const { type, data, error } = e.data;

    if (type === "ERROR") {
      console.error("Worker error:", error);
      resultElement.textContent = error;
      document.getElementById("processingOverlay").classList.add("hidden");
    }

    switch (type) {
      case "MODELS_LOADED":
        console.log("Models loaded in worker");
        captureButton.disabled = false;
        captureButton.textContent = "Capture";
        break;

      case "PROCESS_COMPLETE":
        handleProcessingComplete(data);
        break;
    }
  };

  imageWorker.postMessage({ type: "LOAD_MODELS" });
}

function handleCapture() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  imageDataUrl = canvas.toDataURL("image/jpeg");
  resultElement.textContent = "Processing image...";
  document.getElementById("processingOverlay").classList.remove("hidden");

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const processingTimeout = setTimeout(() => {
    console.error("Processing timeout after 30 seconds");
    resultElement.textContent = "Processing took too long. Please try again.";
    document.getElementById("processingOverlay").classList.add("hidden");

    if (imageWorker) {
      imageWorker.terminate();
      initializeWorker();
    }
  }, 3000000);

  try {
    imageWorker.postMessage(
      {
        type: "PROCESS_IMAGE",
        data: { imageData },
      },
      [imageData.data.buffer]
    );

    const messageHandler = (e) => {
      clearTimeout(processingTimeout);
      const { type, data, error } = e.data;

      if (type === "PROCESS_COMPLETE") {
        handleProcessingComplete(data);
      } else if (type === "ERROR") {
        console.error("Worker error:", error);
        resultElement.textContent = "Error during processing: " + error;
        document.getElementById("processingOverlay").classList.add("hidden");
      }

      imageWorker.removeEventListener("message", messageHandler);
    };

    imageWorker.addEventListener("message", messageHandler);
  } catch (error) {
    clearTimeout(processingTimeout);
    console.error("Error sending data to worker:", error);
    resultElement.textContent = "Error starting image processing";
    document.getElementById("processingOverlay").classList.add("hidden");
  }
}

function handleProcessingComplete(result) {
  if (!result || !result.text) {
    resultElement.textContent = "Error: No text extracted";
    document.getElementById("processingOverlay").classList.add("hidden");
    return;
  }

  extractedText = result.text;
  resultElement.textContent = `Extracted Text: ${extractedText}`;

  const ctx = previewCanvas.getContext("2d");
  const image = new Image();

  image.onload = () => {
    previewCanvas.width = image.width;
    previewCanvas.height = image.height;
    ctx.drawImage(image, 0, 0);

    if (result.boundingBoxes?.length > 0) {
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      result.boundingBoxes.forEach((box) => {
        const [x1, y1] = box.coordinates[0];
        const [x2, y2] = box.coordinates[2];
        const width = (x2 - x1) * image.width;
        const height = (y2 - y1) * image.height;
        const x = x1 * image.width;
        const y = y1 * image.height;
        ctx.strokeRect(x, y, width, height);
      });
    }
  };

  image.src = imageDataUrl;

  document.getElementById("processingOverlay").classList.add("hidden");
  previewCanvas.style.display = "block";
  confirmButton.style.display = "inline-block";
  retryButton.style.display = "inline-block";
  captureButton.style.display = "none";
}

// Event Listeners
captureButton.addEventListener("click", handleCapture);
captureButton.addEventListener("touchstart", handleCapture);
confirmButton.addEventListener("click", handleConfirm);
confirmButton.addEventListener("touchstart", handleConfirm);
retryButton.addEventListener("click", handleRetry);
retryButton.addEventListener("touchstart", handleRetry);
sendButton.addEventListener("click", handleSend);
sendButton.addEventListener("touchstart", handleSend);
discardButton.addEventListener("click", resetUI);
discardButton.addEventListener("touchstart", resetUI);

// Service Worker Registration
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("service-worker.js").then(
      (registration) =>
        console.log(
          "ServiceWorker registration successful with scope: ",
          registration.scope
        ),
      (err) => console.log("ServiceWorker registration failed: ", err)
    );
  });
}

async function init() {
  await initializeWorker();
  await setupCamera();
  await loadOpenCV();
}

function cleanup() {
  if (imageWorker) {
    imageWorker.terminate();
  }
}

window.addEventListener("beforeunload", cleanup);

init();
