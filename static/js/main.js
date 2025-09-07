// AI Shield Frontend JavaScript

// Global variables
let analysisInProgress = false;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize file upload handlers
    initializeFileUpload();
    
    // Initialize theme
    initializeTheme();
    
    // Initialize notifications
    initializeNotifications();
}

// Tooltip initialization
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// File upload enhancements
function initializeFileUpload() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        // Add drag and drop support
        const wrapper = input.closest('.mb-4') || input.parentElement;
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            wrapper.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            wrapper.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            wrapper.addEventListener(eventName, unhighlight, false);
        });
        
        wrapper.addEventListener('drop', handleDrop, false);
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        function highlight(e) {
            wrapper.classList.add('dragover');
        }
        
        function unhighlight(e) {
            wrapper.classList.remove('dragover');
        }
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                input.files = files;
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    });
}

// Theme management
function initializeTheme() {
    const savedTheme = localStorage.getItem('ai-shield-theme');
    if (savedTheme) {
        document.body.setAttribute('data-theme', savedTheme);
    }
}

function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.body.setAttribute('data-theme', newTheme);
    localStorage.setItem('ai-shield-theme', newTheme);
}

// Notification system
function initializeNotifications() {
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
}

// Utility functions
function showLoading(message = 'Processing...') {
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.innerHTML = `
        <div class="text-center">
            <div class="spinner-border spinner-border-lg text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div class="mt-3">
                <h5>${message}</h5>
            </div>
        </div>
    `;
    
    document.body.appendChild(loadingOverlay);
    return loadingOverlay;
}

function hideLoading(overlay) {
    if (overlay && overlay.parentNode) {
        overlay.parentNode.removeChild(overlay);
    }
}

// API utilities
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Form validation
function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('is-invalid');
            isValid = false;
        } else {
            field.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// File validation
function validateFile(file, allowedTypes, maxSizeMB = 500) {
    const errors = [];
    
    // Check file type
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(extension)) {
        errors.push(`File type ${extension} not allowed. Allowed types: ${allowedTypes.join(', ')}`);
    }
    
    // Check file size
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxSizeBytes) {
        errors.push(`File size (${(file.size / 1024 / 1024).toFixed(2)}MB) exceeds maximum allowed size (${maxSizeMB}MB)`);
    }
    
    return errors;
}

// Progress tracking
function createProgressTracker(container, sessionId) {
    const progressHTML = `
        <div class="progress mb-3" style="height: 25px;">
            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                 role="progressbar" style="width: 0%">
                <span class="progress-text">0%</span>
            </div>
        </div>
        <div class="progress-status text-center">
            <h6 class="current-step">Initializing...</h6>
        </div>
    `;
    
    container.innerHTML = progressHTML;
    
    const progressBar = container.querySelector('.progress-bar');
    const progressText = container.querySelector('.progress-text');
    const currentStep = container.querySelector('.current-step');
    
    function updateProgress(progress, step, status = 'running') {
        progressBar.style.width = progress + '%';
        progressText.textContent = progress + '%';
        
        if (step) {
            currentStep.textContent = step;
        }
        
        if (status === 'completed') {
            progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
            progressBar.classList.add('bg-success');
        } else if (status === 'error') {
            progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
            progressBar.classList.add('bg-danger');
        }
    }
    
    // Start polling for updates
    const pollInterval = setInterval(async () => {
        try {
            const status = await apiCall(`/status/${sessionId}`);
            
            updateProgress(
                status.progress || 0,
                status.current_step || 'Processing...',
                status.status
            );
            
            if (status.status === 'completed' || status.status === 'error') {
                clearInterval(pollInterval);
                
                if (status.status === 'completed') {
                    setTimeout(() => {
                        window.location.href = `/results/${sessionId}`;
                    }, 1000);
                }
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 2000);
    
    return { updateProgress, stop: () => clearInterval(pollInterval) };
}

// Chart utilities (for future enhancement)
function createVulnerabilityChart(containerId, data) {
    // Placeholder for chart implementation
    // Could use Chart.js or D3.js for visualization
    console.log('Chart data:', data);
}

// Export functions for global use
window.AIShield = {
    showLoading,
    hideLoading,
    apiCall,
    validateForm,
    validateFile,
    createProgressTracker,
    toggleTheme
};

// Auto-refresh functionality for dashboard
function startAutoRefresh(callback, interval = 30000) {
    const refreshInterval = setInterval(callback, interval);
    
    // Stop auto-refresh when user switches tabs
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            clearInterval(refreshInterval);
        } else {
            // Restart when user returns
            setTimeout(() => startAutoRefresh(callback, interval), 1000);
        }
    });
    
    return refreshInterval;
}

// Local storage utilities
function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(`ai-shield-${key}`, JSON.stringify(data));
    } catch (error) {
        console.warn('Failed to save to localStorage:', error);
    }
}

function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const data = localStorage.getItem(`ai-shield-${key}`);
        return data ? JSON.parse(data) : defaultValue;
    } catch (error) {
        console.warn('Failed to load from localStorage:', error);
        return defaultValue;
    }
}

// Error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    
    // Show user-friendly error message
    const errorAlert = document.createElement('div');
    errorAlert.className = 'alert alert-danger alert-dismissible fade show position-fixed top-0 end-0 m-3';
    errorAlert.style.zIndex = '9999';
    errorAlert.innerHTML = `
        <strong>Error:</strong> Something went wrong. Please try again.
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(errorAlert);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorAlert.parentNode) {
            errorAlert.parentNode.removeChild(errorAlert);
        }
    }, 5000);
});

// Service worker registration (for future PWA support)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}