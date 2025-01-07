# In your terminal, write the following (assuming you also have access to the hendrix ssh server):
# ssh -L 8228:localhost:8228 hendrix
# After logging in, run this script with python - the app will be open on the forwarded 8228 port.
# Go to localhost:8228 in your browser

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from pathlib import Path

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# Create Flask app with explicit template folder
app = Flask(__name__, 
            template_folder=os.path.join(SCRIPT_DIR, 'templates'))

class LabelingSession:
    def __init__(self, crops_dir):
        self.crops_dir = Path(crops_dir)
        self.load_data()
    
    def load_data(self):
        # Load images
        self.image_files = sorted([
            f for f in os.listdir(self.crops_dir) 
            if f.endswith('.png') and f != "annotations.json"
        ])
        
        # Load existing annotations if any
        self.annotations_file = self.crops_dir / "annotations.json"
        self.annotations = {}
        if self.annotations_file.exists():
            with open(self.annotations_file) as f:
                self.annotations = json.load(f)
        
        # Load metadata if available
        self.metadata = {}
        metadata_file = self.crops_dir / "crop_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
    
    def save_annotations(self):
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def get_image_info(self, idx):
        if 0 <= idx < len(self.image_files):
            image_file = self.image_files[idx]
            return {
                'filename': image_file,
                'current_text': self.annotations.get(image_file, ""),
                'metadata': self.metadata.get(image_file, {}),
                'total_images': len(self.image_files),
                'current_index': idx
            }
        return None

def setup_templates():
    """Create templates directory and index.html"""
    templates_dir = os.path.join(SCRIPT_DIR, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>OCR Labeling Tool</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .image-container { margin: 20px 0; text-align: center; }
        img { max-width: 100%; max-height: 400px; }
        .controls { display: flex; gap: 10px; margin: 20px 0; }
        input[type="text"] { flex-grow: 1; padding: 5px; font-size: 16px; }
        button { padding: 5px 15px; cursor: pointer; }
        .metadata { color: #666; margin: 10px 0; }
        .progress { margin: 10px 0; }
        .keyboard-hints {
            background: #f5f5f5;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>OCR Labeling Tool</h1>
    
    <div class="keyboard-hints">
        <strong>Keyboard shortcuts:</strong><br>
        Enter - Save and next image<br>
        Left Arrow - Previous image<br>
        Right Arrow - Next image
    </div>
    
    <div class="progress">
        Image <span id="current">0</span> of <span id="total">0</span>
    </div>
    
    <div class="metadata" id="metadata"></div>
    
    <div class="image-container">
        <img id="current-image" src="" alt="Current image">
    </div>
    
    <div class="controls">
        <button onclick="prevImage()">&lt; Previous</button>
        <input type="text" id="text-input" placeholder="Enter text...">
        <button onclick="nextImage()">Next &gt;</button>
        <button onclick="saveAndNext()">Save & Next</button>
    </div>
    
    <script>
        let currentIdx = 0;
        let currentFilename = '';
        
        async function loadImage(idx) {
            const response = await fetch(`/info/${idx}`);
            if (response.ok) {
                const info = await response.json();
                currentFilename = info.filename;
                
                document.getElementById('current-image').src = `/image/${idx}`;
                document.getElementById('text-input').value = info.current_text;
                document.getElementById('current').textContent = idx + 1;
                document.getElementById('total').textContent = info.total_images;
                
                // Display metadata
                const metadata = info.metadata;
                let metadataText = '';
                if (metadata.source_image) {
                    metadataText += `Source: ${metadata.source_image} | `;
                }
                if (metadata.column !== undefined) {
                    metadataText += `Column: ${metadata.column} | `;
                }
                if (metadata.row !== undefined) {
                    metadataText += `Row: ${metadata.row}`;
                }
                document.getElementById('metadata').textContent = metadataText;
                
                document.getElementById('text-input').focus();
            }
        }
        
        async function saveAnnotation() {
            const text = document.getElementById('text-input').value;
            const response = await fetch('/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: currentFilename,
                    text: text
                })
            });
            return response.ok;
        }
        
        async function saveAndNext() {
            if (await saveAnnotation()) {
                nextImage();
            }
        }
        
        function prevImage() {
            if (currentIdx > 0) {
                currentIdx--;
                loadImage(currentIdx);
            }
        }
        
        function nextImage() {
            currentIdx++;
            loadImage(currentIdx);
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', async function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                await saveAndNext();
            } else if (e.key === 'ArrowLeft') {
                prevImage();
            } else if (e.key === 'ArrowRight') {
                nextImage();
            }
        });
        
        // Load first image on start
        loadImage(0);
    </script>
</body>
</html>
    """
    
    index_path = os.path.join(templates_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(index_html)
    print(f"Created template at: {index_path}")

# Create templates before initializing routes
setup_templates()

# Initialize session
# session = LabelingSession('../data/ocr_training_crops')
session = LabelingSession('../data/ocr_validation_crops')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image/<int:idx>')
def get_image(idx):
    if 0 <= idx < len(session.image_files):
        return send_file(
            session.crops_dir / session.image_files[idx],
            mimetype='image/png'
        )
    return "Image not found", 404

@app.route('/info/<int:idx>')
def get_info(idx):
    info = session.get_image_info(idx)
    if info:
        return jsonify(info)
    return "Not found", 404

@app.route('/save', methods=['POST'])
def save_annotation():
    data = request.json
    filename = data.get('filename')
    text = data.get('text', '').strip()
    
    if filename and text:
        session.annotations[filename] = text
        session.save_annotations()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

if __name__ == '__main__':
    print(f"Templates directory: {app.template_folder}")
    app.run(host='0.0.0.0', port=8228, debug=True)