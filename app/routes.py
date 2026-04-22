from flask import current_app as app, render_template, request, jsonify
from io import BytesIO
from PIL import Image

# Initialize AI components
from model.classifier import BirdClassifier
from llm.generator import BirdInfoGenerator

classifier = BirdClassifier()
generator = BirdInfoGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Process image in memory
        img_bytes = file.read()
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        
        # 1. ML Classification
        predictions = classifier.predict(image, top_k=3)
        top_species = predictions[0]['species']
        
        # 2. LLM Info Generation
        if "Class" not in top_species and "Model Not Loaded" not in top_species:
            bird_info = generator.generate_info(top_species)
        else:
            bird_info = {
                "Description": "Please train the model first.",
                "Habitat": "-",
                "Diet": "-",
                "Fun Fact": "-"
            }

        return jsonify({
            'predictions': predictions,
            'info': bird_info
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
