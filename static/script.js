// CN2VF-Net Detection Interface JavaScript

// Global state
let currentImage = null;
let currentResults = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const uploadForm = document.getElementById('uploadForm');
const confidenceSlider = document.getElementById('confidenceThreshold');
const confidenceValue = document.getElementById('confidenceValue');
const resultsSection = document.getElementById('resultsSection');
const imageResult = document.getElementById('imageResult');
const resultImage = document.getElementById('resultImage');
const bboxCanvas = document.getElementById('bboxCanvas');
const predictedClass = document.getElementById('predictedClass');
const confidenceBadge = document.getElementById('confidenceBadge');
const fireProbBar = document.getElementById('fireProbBar');
const fireProbValue = document.getElementById('fireProbValue');
const smokeProbBar = document.getElementById('smokeProbBar');
const smokeProbValue = document.getElementById('smokeProbValue');
const neutralProbBar = document.getElementById('neutralProbBar');
const neutralProbValue = document.getElementById('neutralProbValue');
const bboxCoords = document.getElementById('bboxCoords');
const predictBtn = document.getElementById('predictBtn');
const alertBox = document.getElementById('alertBox');

// Initialize confidence slider
confidenceSlider.addEventListener('input', (e) => {
    confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
    if (currentResults) {
        drawBoundingBox();
    }
});

// Upload area click
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Handle file selection
function handleFile(file) {
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    // Store the file
    currentImage = file;
    
    // Update UI
    uploadPlaceholder.innerHTML = `
        <div class="upload-icon">✓</div>
        <p><strong>${file.name}</strong></p>
        <p class="upload-hint">${(file.size / 1024).toFixed(2)} KB - Click or drop to change</p>
    `;
    
    predictBtn.disabled = false;
}

// Form submit handler
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!currentImage) {
        alert('Please select an image first');
        return;
    }
    
    await runDetection();
});

// Run detection
async function runDetection() {
    const threshold = parseFloat(confidenceSlider.value);
    
    // Show loading state
    predictBtn.disabled = true;
    predictBtn.textContent = 'Detecting...';
    resultsSection.style.display = 'none';
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', currentImage);
        formData.append('confidence', threshold);
        
        // Send request
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Store results
        currentResults = result;
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Detection error:', error);
        alert(`Detection failed: ${error.message}`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Run Detection';
    }
}

// Display detection results
function displayResults(result) {
    // Load and display image
    const reader = new FileReader();
    reader.onload = (e) => {
        resultImage.src = e.target.result;
        resultImage.onload = () => {
            // Set canvas size to match image display size
            const rect = resultImage.getBoundingClientRect();
            bboxCanvas.width = resultImage.naturalWidth;
            bboxCanvas.height = resultImage.naturalHeight;
            
            // Draw bounding box
            drawBoundingBox();
        };
    };
    reader.readAsDataURL(currentImage);
    
    // Update class and confidence
    const classNames = ['Fire', 'Smoke', 'Neutral'];
    const className = classNames[result.predicted_class];
    predictedClass.textContent = className;
    confidenceBadge.textContent = `${(result.confidence * 100).toFixed(1)}%`;
    
    // Color-code confidence badge
    if (result.confidence >= 0.8) {
        confidenceBadge.style.background = '#4CAF50'; // Green
    } else if (result.confidence >= 0.5) {
        confidenceBadge.style.background = '#FFA726'; // Orange
    } else {
        confidenceBadge.style.background = '#F44336'; // Red
    }
    
    // Update probability bars
    updateProbabilities(result.probabilities);
    
    // Update bbox coordinates
    const bbox = result.bbox;
    bboxCoords.innerHTML = `
        <span>X: ${bbox[0].toFixed(3)}</span>
        <span>Y: ${bbox[1].toFixed(3)}</span>
        <span>W: ${bbox[2].toFixed(3)}</span>
        <span>H: ${bbox[3].toFixed(3)}</span>
    `;
    
    // Show alert for fire/smoke
    if (result.predicted_class === 0 || result.predicted_class === 1) {
        alertBox.innerHTML = `⚠️ <strong>${className} Detected!</strong> Immediate attention may be required.`;
        alertBox.style.display = 'block';
        alertBox.style.borderColor = result.predicted_class === 0 ? '#F44336' : '#FFA726';
    } else {
        alertBox.style.display = 'none';
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update probability bars
function updateProbabilities(probs) {
    // Fire
    const fireProb = probs[0] * 100;
    fireProbBar.style.width = `${fireProb}%`;
    fireProbBar.textContent = `${fireProb.toFixed(1)}%`;
    fireProbValue.textContent = `${fireProb.toFixed(1)}%`;
    
    // Smoke
    const smokeProb = probs[1] * 100;
    smokeProbBar.style.width = `${smokeProb}%`;
    smokeProbBar.textContent = `${smokeProb.toFixed(1)}%`;
    smokeProbValue.textContent = `${smokeProb.toFixed(1)}%`;
    
    // Neutral
    const neutralProb = probs[2] * 100;
    neutralProbBar.style.width = `${neutralProb}%`;
    neutralProbBar.textContent = `${neutralProb.toFixed(1)}%`;
    neutralProbValue.textContent = `${neutralProb.toFixed(1)}%`;
}

// Draw bounding box on canvas
function drawBoundingBox() {
    if (!currentResults || !resultImage.complete) return;
    
    const ctx = bboxCanvas.getContext('2d');
    const width = bboxCanvas.width;
    const height = bboxCanvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Get bbox in normalized coordinates [x_center, y_center, w, h]
    const bbox = currentResults.bbox;
    const [x_center, y_center, w, h] = bbox;
    
    // Convert to pixel coordinates
    const x = (x_center - w / 2) * width;
    const y = (y_center - h / 2) * height;
    const box_width = w * width;
    const box_height = h * height;
    
    // Choose color based on class
    const classColors = ['#FF6B35', '#7F8C8D', '#4CAF50'];
    const color = classColors[currentResults.predicted_class];
    
    // Draw bbox
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, box_width, box_height);
    
    // Draw label background
    const classNames = ['Fire', 'Smoke', 'Neutral'];
    const label = `${classNames[currentResults.predicted_class]} ${(currentResults.confidence * 100).toFixed(1)}%`;
    ctx.font = 'bold 18px Arial';
    const textMetrics = ctx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = 24;
    
    ctx.fillStyle = color;
    ctx.fillRect(x, y - textHeight - 5, textWidth + 10, textHeight + 5);
    
    // Draw label text
    ctx.fillStyle = 'white';
    ctx.fillText(label, x + 5, y - 10);
}

// Check model and training status on page load
async function checkStatus() {
    try {
        const response = await fetch('/status');
        const status = await response.json();
        
        // Update status badge
        const statusBadge = document.querySelector('.status-badge');
        if (statusBadge && status.model_loaded) {
            statusBadge.innerHTML = `
                <span class="status-dot"></span>
                <span>Model Loaded | ${status.parameters} params</span>
            `;
        }
        
    } catch (error) {
        console.error('Status check failed:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
});
