// CN2VF-Net Training Results JavaScript

// DOM Elements
const currentEpoch = document.getElementById('currentEpoch');
const totalEpochs = document.getElementById('totalEpochs');
const trainingTime = document.getElementById('trainingTime');
const bestAccuracy = document.getElementById('bestAccuracy');
const trainingStatus = document.getElementById('trainingStatus');
const modelParams = document.getElementById('modelParams');

// Load training metrics on page load
async function loadTrainingMetrics() {
    try {
        const response = await fetch('/api/training-metrics');
        
        if (!response.ok) {
            throw new Error(`Failed to fetch metrics: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update metrics
        if (data.metrics) {
            updateMetrics(data.metrics);
        }
        
        // Update status
        if (data.status) {
            updateStatus(data.status);
        }
        
    } catch (error) {
        console.error('Error loading metrics:', error);
        showError('Failed to load training metrics. Make sure training has been started.');
    }
}

// Update metrics display
function updateMetrics(metrics) {
    // Current Epoch
    if (currentEpoch && metrics.current_epoch !== undefined) {
        currentEpoch.textContent = metrics.current_epoch || 0;
    }
    
    // Total Epochs
    if (totalEpochs && metrics.total_epochs) {
        totalEpochs.textContent = metrics.total_epochs;
    }
    
    // Training Time
    if (trainingTime && metrics.training_time) {
        trainingTime.textContent = formatTime(metrics.training_time);
    }
    
    // Best Accuracy
    if (bestAccuracy && metrics.best_accuracy !== undefined) {
        const accuracy = (metrics.best_accuracy * 100).toFixed(2);
        bestAccuracy.textContent = `${accuracy}%`;
        
        // Color code based on accuracy
        if (metrics.best_accuracy >= 0.9) {
            bestAccuracy.style.color = '#4CAF50';
        } else if (metrics.best_accuracy >= 0.75) {
            bestAccuracy.style.color = '#FFA726';
        } else {
            bestAccuracy.style.color = '#F44336';
        }
    }
    
    // Model Parameters
    if (modelParams && metrics.parameters) {
        modelParams.textContent = metrics.parameters;
    }
    
    // Additional metrics
    if (metrics.train_loss !== undefined) {
        const trainLossEl = document.getElementById('trainLoss');
        if (trainLossEl) {
            trainLossEl.textContent = metrics.train_loss.toFixed(4);
        }
    }
    
    if (metrics.val_loss !== undefined) {
        const valLossEl = document.getElementById('valLoss');
        if (valLossEl) {
            valLossEl.textContent = metrics.val_loss.toFixed(4);
        }
    }
    
    if (metrics.train_accuracy !== undefined) {
        const trainAccEl = document.getElementById('trainAccuracy');
        if (trainAccEl) {
            trainAccEl.textContent = `${(metrics.train_accuracy * 100).toFixed(2)}%`;
        }
    }
    
    if (metrics.learning_rate !== undefined) {
        const lrEl = document.getElementById('learningRate');
        if (lrEl) {
            lrEl.textContent = metrics.learning_rate.toExponential(2);
        }
    }
}

// Update training status
function updateStatus(status) {
    if (trainingStatus) {
        trainingStatus.textContent = status;
        
        // Color code based on status
        if (status === 'Training Complete') {
            trainingStatus.style.color = '#4CAF50';
        } else if (status === 'Training In Progress') {
            trainingStatus.style.color = '#FFA726';
        } else if (status === 'Not Started') {
            trainingStatus.style.color = '#7F8C8D';
        } else if (status.includes('Error') || status.includes('Failed')) {
            trainingStatus.style.color = '#F44336';
        }
    }
}

// Format time in seconds to HH:MM:SS
function formatTime(seconds) {
    if (!seconds || seconds === 0) return '00:00:00';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Show error message
function showError(message) {
    const metricsGrid = document.querySelector('.metrics-grid');
    if (metricsGrid) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert-box';
        errorDiv.style.gridColumn = '1 / -1';
        errorDiv.innerHTML = `⚠️ <strong>Error:</strong> ${message}`;
        metricsGrid.insertAdjacentElement('beforebegin', errorDiv);
    }
}

// Load model checkpoint
async function loadModel() {
    const loadBtn = document.getElementById('loadModelBtn');
    if (!loadBtn) return;
    
    loadBtn.disabled = true;
    loadBtn.textContent = 'Loading...';
    
    try {
        const response = await fetch('/load-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to load model');
        }
        
        const result = await response.json();
        
        if (result.success) {
            alert(`Model loaded successfully! Accuracy: ${(result.accuracy * 100).toFixed(2)}%`);
            // Reload page to update status
            window.location.reload();
        } else {
            throw new Error(result.error || 'Unknown error');
        }
        
    } catch (error) {
        console.error('Error loading model:', error);
        alert(`Failed to load model: ${error.message}`);
    } finally {
        loadBtn.disabled = false;
        loadBtn.textContent = 'Load Best Model';
    }
}

// Refresh metrics periodically if training is in progress
function startAutoRefresh() {
    setInterval(async () => {
        try {
            const response = await fetch('/api/training-metrics');
            const data = await response.json();
            
            if (data.status === 'Training In Progress') {
                updateMetrics(data.metrics);
                updateStatus(data.status);
            } else if (data.status === 'Training Complete') {
                // Stop auto-refresh if training is complete
                location.reload();
            }
        } catch (error) {
            // Silently fail - don't interrupt user experience
            console.error('Auto-refresh error:', error);
        }
    }, 5000); // Refresh every 5 seconds
}

// Check if TensorBoard is accessible
async function checkTensorBoard() {
    const tensorboardLink = document.querySelector('a[href*="6006"]');
    if (!tensorboardLink) return;
    
    try {
        const response = await fetch('http://localhost:6006', { mode: 'no-cors' });
        // If we get here, TensorBoard is likely running
        tensorboardLink.style.opacity = '1';
    } catch (error) {
        // TensorBoard not running
        tensorboardLink.style.opacity = '0.5';
        tensorboardLink.title = 'TensorBoard not running. Start it with: tensorboard --logdir=runs';
    }
}

// Add training progress visualization
function visualizeProgress() {
    const metricsCard = document.querySelector('.metrics-grid');
    if (!metricsCard || !currentEpoch || !totalEpochs) return;
    
    const current = parseInt(currentEpoch.textContent) || 0;
    const total = parseInt(totalEpochs.textContent) || 50;
    const progress = (current / total) * 100;
    
    // Create progress bar if it doesn't exist
    let progressBar = document.getElementById('trainingProgressBar');
    if (!progressBar) {
        const progressContainer = document.createElement('div');
        progressContainer.style.cssText = 'margin: 20px 0; padding: 20px; background: #F5F7FA; border-radius: 10px;';
        progressContainer.innerHTML = `
            <h4 style="margin-bottom: 10px; color: #2C3E50;">Training Progress</h4>
            <div class="progress">
                <div id="trainingProgressBar" class="progress-bar fire-bar" style="width: 0%;">
                    <span>0%</span>
                </div>
            </div>
        `;
        metricsCard.parentElement.insertBefore(progressContainer, metricsCard);
        progressBar = document.getElementById('trainingProgressBar');
    }
    
    // Update progress bar
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = `${progress.toFixed(1)}%`;
    }
}

// Export metrics to CSV
function exportMetrics() {
    // This would require storing historical metrics
    alert('Metrics export feature - check training_log.json and TensorBoard for detailed metrics');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadTrainingMetrics();
    checkTensorBoard();
    visualizeProgress();
    
    // Check if training is in progress and start auto-refresh
    setTimeout(() => {
        const status = trainingStatus?.textContent;
        if (status === 'Training In Progress') {
            startAutoRefresh();
        }
    }, 1000);
    
    // Attach load model button handler
    const loadBtn = document.getElementById('loadModelBtn');
    if (loadBtn) {
        loadBtn.addEventListener('click', loadModel);
    }
});

// Refresh button handler
const refreshBtn = document.getElementById('refreshBtn');
if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
        location.reload();
    });
}
