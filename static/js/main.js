/**
 * Research Router Web App - Main JavaScript
 */

// Global utilities
const ResearchRouter = {
    // Show loading modal
    showLoading: function(text = 'Loading...') {
        const modal = document.getElementById('loadingModal');
        const textElement = document.getElementById('loadingText');
        
        if (modal && textElement) {
            textElement.textContent = text;
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();
            return bsModal;
        }
    },

    // Hide loading modal
    hideLoading: function() {
        const modal = document.getElementById('loadingModal');
        if (modal) {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        }
    },

    // Show toast notification
    showToast: function(message, type = 'info') {
        const toastContainer = this.getOrCreateToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove toast element after it's hidden
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
        
        return bsToast;
    },

    // Get or create toast container
    getOrCreateToastContainer: function() {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1055';
            document.body.appendChild(container);
        }
        return container;
    },

    // API helper functions
    api: {
        // Generic API call
        call: async function(url, options = {}) {
            const defaultOptions = {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            };

            const response = await fetch(url, { ...defaultOptions, ...options });
            
            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: 'Network error' }));
                throw new Error(error.error || `HTTP ${response.status}`);
            }
            
            return await response.json();
        },

        // Create session
        createSession: async function(name) {
            return await this.call('/api/sessions', {
                method: 'POST',
                body: JSON.stringify({ name })
            });
        },

        // Switch session
        switchSession: async function(sessionName) {
            return await this.call(`/api/sessions/${sessionName}/switch`, {
                method: 'POST'
            });
        },

        // Get session status
        getSessionStatus: async function(sessionName) {
            return await this.call(`/api/sessions/${sessionName}/status`);
        },

        // Upload files
        uploadFiles: async function(files) {
            const formData = new FormData();
            files.forEach(file => formData.append('files', file));
            
            return await fetch('/api/upload', {
                method: 'POST',
                body: formData
            }).then(response => {
                if (!response.ok) {
                    return response.json().then(error => {
                        throw new Error(error.error || `HTTP ${response.status}`);
                    });
                }
                return response.json();
            });
        },

        // Query knowledge graph
        query: async function(question, mode = 'global') {
            return await this.call('/api/query', {
                method: 'POST',
                body: JSON.stringify({ question, mode })
            });
        },

        // Get session history
        getHistory: async function(sessionName, limit = 20) {
            return await this.call(`/api/sessions/${sessionName}/history?limit=${limit}`);
        }
    },

    // Utility functions
    utils: {
        // Format file size
        formatFileSize: function(bytes) {
            if (bytes === 0) return '0 Bytes';
            
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },

        // Format timestamp
        formatTimestamp: function(timestamp) {
            return new Date(timestamp).toLocaleString();
        },

        // Debounce function
        debounce: function(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },

        // Check if element is in viewport
        isInViewport: function(element) {
            const rect = element.getBoundingClientRect();
            return (
                rect.top >= 0 &&
                rect.left >= 0 &&
                rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                rect.right <= (window.innerWidth || document.documentElement.clientWidth)
            );
        },

        // Smooth scroll to element
        scrollToElement: function(element, offset = 0) {
            const elementPosition = element.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - offset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        },

        // Copy text to clipboard
        copyToClipboard: async function(text) {
            try {
                await navigator.clipboard.writeText(text);
                ResearchRouter.showToast('Copied to clipboard!', 'success');
                return true;
            } catch (err) {
                console.error('Failed to copy text: ', err);
                ResearchRouter.showToast('Failed to copy text', 'danger');
                return false;
            }
        },

        // Validate file type
        isValidFileType: function(file) {
            const allowedTypes = [
                'application/pdf',
                'text/plain',
                'text/markdown',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ];
            
            const allowedExtensions = ['.pdf', '.txt', '.md', '.doc', '.docx'];
            
            return allowedTypes.includes(file.type) || 
                   allowedExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        }
    },

    // Chat streaming functionality for real-time chat integration
    chatStream: {
        // Poll for chat messages from an operation
        pollChatMessages: function(operationId, onMessage, onComplete, onError) {
            const pollInterval = 500; // Poll every 500ms
            let lastMessageCount = 0;
            
            const poll = async () => {
                try {
                    const response = await fetch(`/api/chat/${operationId}`);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    // Process new messages
                    if (data.messages.length > lastMessageCount) {
                        const newMessages = data.messages.slice(lastMessageCount);
                        newMessages.forEach(message => {
                            if (onMessage) onMessage(message);
                        });
                        lastMessageCount = data.messages.length;
                    }
                    
                    // Check if operation is complete
                    if (data.complete) {
                        if (onComplete) onComplete(data.messages);
                        return; // Stop polling
                    }
                    
                    // Continue polling
                    setTimeout(poll, pollInterval);
                    
                } catch (error) {
                    console.error('Error polling chat messages:', error);
                    if (onError) onError(error.message);
                }
            };
            
            // Start polling
            poll();
        }
    }
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert[data-auto-dismiss="true"]');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Add loading state to forms
    const forms = document.querySelectorAll('form[data-loading="true"]');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                const originalText = submitButton.innerHTML;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Loading...';
                
                // Re-enable after 30 seconds as fallback
                setTimeout(() => {
                    submitButton.disabled = false;
                    submitButton.innerHTML = originalText;
                }, 30000);
            }
        });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + / to focus search or main input
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            const searchInput = document.getElementById('messageInput') || 
                              document.getElementById('sessionName') ||
                              document.querySelector('input[type="text"]');
            if (searchInput) {
                searchInput.focus();
            }
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            const openModal = document.querySelector('.modal.show');
            if (openModal) {
                const bsModal = bootstrap.Modal.getInstance(openModal);
                if (bsModal) {
                    bsModal.hide();
                }
            }
        }
    });

    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                ResearchRouter.utils.scrollToElement(target, 20);
            }
        });
    });

    // Back button handling
    const backButton = document.getElementById('backButton');
    if (backButton) {
        backButton.addEventListener('click', function() {
            if (window.history.length > 1) {
                window.history.back();
            } else {
                window.location.href = '/';
            }
        });
    }

    // Auto-resize textareas
    const textareas = document.querySelectorAll('textarea[data-auto-resize="true"]');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    });

    // Confirmation for dangerous actions
    const confirmButtons = document.querySelectorAll('[data-confirm]');
    confirmButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            const message = this.dataset.confirm;
            if (!confirm(message)) {
                e.preventDefault();
                e.stopPropagation();
                return false;
            }
        });
    });

    console.log('Research Router Web App initialized');
});

// Make ResearchRouter globally available
window.ResearchRouter = ResearchRouter;