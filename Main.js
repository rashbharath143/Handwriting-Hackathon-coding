document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('previewImage');
    const placeholderPreview = document.getElementById('placeholderPreview');
    const drawingCanvas = document.getElementById('drawingCanvas');
    const clearBtn = document.getElementById('clearBtn');
    const recognizeBtn = document.getElementById('recognizeBtn');
    const resultsCard = document.getElementById('resultsCard');
    const recognizedDigit = document.getElementById('recognizedDigit');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const tryAgainBtn = document.getElementById('tryAgainBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    
    // Canvas setup
    const ctx = drawingCanvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    
    // Set up drawing canvas
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
    
    // Drawing functions
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        e.preventDefault();
        
        const [currentX, currentY] = getCoordinates(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
        
        [lastX, lastY] = [currentX, currentY];
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    function getCoordinates(event) {
        let x, y;
        
        // Get coordinates for both mouse and touch events
        if (event.type.includes('touch')) {
            const rect = drawingCanvas.getBoundingClientRect();
            const touch = event.touches[0] || event.changedTouches[0];
            x = touch.clientX - rect.left;
            y = touch.clientY - rect.top;
        } else {
            const rect = drawingCanvas.getBoundingClientRect();
            x = event.clientX - rect.left;
            y = event.clientY - rect.top;
        }
        
        // Scale coordinates if canvas displayed size differs from its internal size
        x = x * (drawingCanvas.width / drawingCanvas.offsetWidth);
        y = y * (drawingCanvas.height / drawingCanvas.offsetHeight);
        
        return [x, y];
    }
    
    // Event listeners for drawing
    drawingCanvas.addEventListener('mousedown', startDrawing);
    drawingCanvas.addEventListener('mousemove', draw);
    drawingCanvas.addEventListener('mouseup', stopDrawing);
    drawingCanvas.addEventListener('mouseout', stopDrawing);
    
    // Touch support
    drawingCanvas.addEventListener('touchstart', startDrawing);
    drawingCanvas.addEventListener('touchmove', draw);
    drawingCanvas.addEventListener('touchend', stopDrawing);
    
    // Clear canvas
    clearBtn.addEventListener('click', function() {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        hideResults();
    });
    
    // Preview uploaded image
    imageInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                placeholderPreview.style.display = 'none';
            };
            reader.readAsDataURL(file);
            hideResults();
        }
    });
    
    // Upload form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) {
            showError('Please select an image file');
            return;
        }
        
        // Check file type
        const fileType = file.type;
        if (!fileType.match('image/jpeg') && !fileType.match('image/png')) {
            showError('Please upload a JPG or PNG image');
            return;
        }
        
        // Create FormData
        const formData = new FormData();
        formData.append('image', file);
        
        // Send request to server
        sendImageToServer(formData);
    });
    
    // Recognize button for drawn digit
    recognizeBtn.addEventListener('click', function() {
        // Convert canvas to base64 image data
        const imageData = drawingCanvas.toDataURL('image/png');
        // Remove the data URL prefix to get only the base64 string
        const base64Data = imageData.replace(/^data:image\/png;base64,/, '');
        
        // Send to server
        sendDrawingToServer(base64Data);
    });
    
    // Try again button
    tryAgainBtn.addEventListener('click', function() {
        hideResults();
    });
    
    // Function to send image file to server
    function sendImageToServer(formData) {
        showLoading();
        hideError();
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Error processing image');
                });
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            displayResults(data.digit, data.confidence);
        })
        .catch(error => {
            hideLoading();
            showError(error.message);
        });
    }
    
    // Function to send drawn digit to server
    function sendDrawingToServer(imageData) {
        showLoading();
        hideError();
        
        fetch('/predict_from_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_data: imageData
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Error processing image');
                });
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            displayResults(data.digit, data.confidence);
        })
        .catch(error => {
            hideLoading();
            showError(error.message);
        });
    }
    
    // Functions to handle UI state
    function showLoading() {
        loadingSpinner.style.display = 'block';
        resultsCard.style.display = 'none';
    }
    
    function hideLoading() {
        loadingSpinner.style.display = 'none';
    }
    
    function showError(message) {
        errorAlert.style.display = 'block';
        errorMessage.textContent = message;
    }
    
    function hideError() {
        errorAlert.style.display = 'none';
    }
    
    function hideResults() {
        resultsCard.style.display = 'none';
        hideError();
    }
    
    function displayResults(digit, confidence) {
        // Update results
        recognizedDigit.textContent = digit;
        
        // Round confidence to 2 decimal places
        const confidencePercent = confidence.toFixed(2);
        confidenceValue.textContent = confidencePercent;
        
        // Update progress bar
        confidenceBar.style.width = confidencePercent + '%';
        
        // Set color based on confidence
        if (confidence >= 80) {
            confidenceBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-success';
        } else if (confidence >= 50) {
            confidenceBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-warning';
        } else {
            confidenceBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-danger';
        }
        
        // Show results card
        resultsCard.style.display = 'block';
    }
});
