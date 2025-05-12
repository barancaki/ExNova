document.addEventListener('DOMContentLoaded', () => {
    initializeDropzones();
    initializeFileUploads();
    initializePromptSubmission();
});

function initializeDropzones() {
    const dropzones = document.querySelectorAll('.dropzone');
    
    dropzones.forEach(dropzone => {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-over');
        });

        dropzone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            handleFiles(files, dropzone);
        });

        dropzone.addEventListener('click', () => {
            const fileInput = dropzone.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.click();
            }
        });
    });
}

function initializeFileUploads() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', (e) => {
            const files = e.target.files;
            handleFiles(files, input.closest('.dropzone'));
        });
    });
}

function handleFiles(files, dropzone) {
    const fileList = dropzone.querySelector('.file-list') || createFileList(dropzone);
    fileList.innerHTML = '';

    Array.from(files).forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <i class="fas fa-file-excel"></i>
            <span>${file.name}</span>
            <small>(${formatFileSize(file.size)})</small>
        `;
        fileList.appendChild(fileItem);
    });
}

function createFileList(dropzone) {
    const fileList = document.createElement('div');
    fileList.className = 'file-list';
    dropzone.appendChild(fileList);
    return fileList;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function initializePromptSubmission() {
    const form = document.querySelector('#promptForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const loadingSpinner = document.querySelector('.loading-spinner');
        const responseContainer = document.querySelector('.response-container');

        try {
            loadingSpinner.style.display = 'flex';
            const response = await fetch('/get-ai-response', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            displayResponse(data.response);
        } catch (error) {
            showError('An error occurred while processing your request.');
            console.error('Error:', error);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });
}

function displayResponse(response) {
    const responseContainer = document.querySelector('.response-container');
    const responseContent = document.querySelector('.response-content');
    
    if (!responseContent) {
        const content = document.createElement('div');
        content.className = 'response-content';
        responseContainer.appendChild(content);
    }
    
    responseContent.textContent = response;
    responseContainer.style.display = 'block';
    responseContainer.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show';
    errorDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(errorDiv, container.firstChild);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
} 