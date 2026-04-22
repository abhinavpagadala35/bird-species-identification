document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resetBtn = document.getElementById('reset-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadPanel = document.getElementById('upload-panel');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');
    
    let currentFile = null;

    // --- Drag and Drop Handling ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, (e) => {
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG/PNG).');
            return;
        }
        
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // --- Reset ---
    resetBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        
        // Reset accordions
        document.querySelectorAll('.accordion-item').forEach(item => {
            item.classList.remove('active');
        });
    });

    newAnalysisBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadPanel.classList.remove('hidden');
        resetBtn.click();
    });

    // --- Analysis ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        uploadPanel.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred');
            }

            renderResults(data);

        } catch (error) {
            alert(`Error: ${error.message}`);
            uploadPanel.classList.remove('hidden');
        } finally {
            loadingSection.classList.add('hidden');
        }
    });

    function renderResults(data) {
        // Render Predictions
        const list = document.getElementById('predictions-list');
        list.innerHTML = '';
        
        data.predictions.forEach((pred, index) => {
            const li = document.createElement('li');
            li.className = 'prediction-item';
            
            // Highlight the top prediction slightly more if needed, 
            // but CSS already styles the items.
            
            li.innerHTML = `
                <span class="species">${pred.species}</span>
                <span class="prob">Confidence: ${(pred.probability * 100).toFixed(2)}%</span>
            `;
            list.appendChild(li);
        });

        // Render LLM Info
        document.getElementById('info-description').textContent = data.info.Description || '-';
        document.getElementById('info-habitat').textContent = data.info.Habitat || '-';
        document.getElementById('info-diet').textContent = data.info.Diet || '-';
        document.getElementById('info-funfact').textContent = data.info['Fun Fact'] || '-';

        resultsSection.classList.remove('hidden');
    }

    // --- Accordion Logic for Info Panels ---
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    
    accordionHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const currentItem = header.parentElement;
            const isActive = currentItem.classList.contains('active');
            
            // Optional: Close all other accordions for a cleaner flow
            document.querySelectorAll('.accordion-item').forEach(item => {
                item.classList.remove('active');
            });

            // If it wasn't active before, open it now (toggle)
            if (!isActive) {
                currentItem.classList.add('active');
            }
        });
    });
});
