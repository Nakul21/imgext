// ... (previous code remains the same)

function decodeText(predictions) {
    // Assuming the charset order matches the model's output
    const charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
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
    // ... (previous code remains the same)
    
    try {
        // ... (previous code remains the same)
        
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

// ... (rest of the code remains the same)
