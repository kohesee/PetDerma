import os
import csv
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class_names = ['Flea_Allergy', 'Health', 'Ringworm', 'Scabies']

class CatSkinDiseasePredictor:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.class_names = ['Flea_Allergy', 'Health', 'Ringworm', 'Scabies']

        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {"error": f"Error opening image: {str(e)}"}

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        confidence_scores = probabilities.cpu().numpy()

        result = {
            "filename": os.path.basename(image_path),
            "probabilities": {class_name: float(confidence_scores[i]) * 100 for i, class_name in enumerate(self.class_names)},
            "prediction": self.class_names[np.argmax(confidence_scores)],
            "confidence": float(np.max(confidence_scores)) * 100
        }
        
        return result


disease_info = {
    "Flea_Allergy": {
        "description": "Flea allergy dermatitis (FAD) is a skin condition in cats caused by an allergic reaction to flea saliva.",
        "symptoms": "Intense itching, hair loss, skin redness, scabs, and hot spots, particularly around the base of the tail, head, neck, and thighs.",
        "treatment": "Flea control products, anti-inflammatory medications, and keeping the environment flea-free."
    },
    "Health": {
        "description": "A healthy cat skin is free from skin conditions and diseases.",
        "symptoms": "Smooth, clean coat, no excessive scratching, no visible redness, lesions, or parasites.",
        "treatment": "Regular grooming, a balanced diet, and routine veterinary check-ups to maintain skin health."
    },
    "Ringworm": {
        "description": "Ringworm is a fungal infection that affects the skin, hair, and occasionally nails of cats.",
        "symptoms": "Circular patches of hair loss, redness, scaling, and crusty skin, most commonly on the head, ears, and forelimbs.",
        "treatment": "Antifungal medications (oral and topical), environmental decontamination, and sometimes clipping the coat in long-haired cats."
    },
    "Scabies": {
        "description": "Scabies (mange) is caused by the Sarcoptes scabiei mite, which burrows into the skin causing intense irritation.",
        "symptoms": "Severe itching, redness, scaling, crusty skin lesions, hair loss, especially on the ears, face, legs, and belly.",
        "treatment": "Anti-parasitic medications, medicated baths, and environmental treatment to eliminate mites."
    }
}


FEEDBACK_CSV = 'feedback_data.csv'
if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'image_path', 'predicted_class', 'probability', 'confirmed_conditions', 'doctor_notes'])

def create_chart(probabilities):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(class_names))
    
    prob_values = [probabilities[class_name] for class_name in class_names]
    
    bars = ax.barh(y_pos, prob_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ') for name in class_names])
    ax.invert_yaxis()
    ax.set_xlabel('Probability (%)')
    ax.set_title('Disease Probability')
    
    predicted_idx = np.argmax(prob_values)
    bars[predicted_idx].set_color('#4CAF50')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_position = width + 1
        ax.text(label_position, bar.get_y() + bar.get_height()/2, 
                f'{prob_values[i]:.1f}%', 
                va='center')
    
    ax.set_xlim(0, 110)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    chart_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return chart_img

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
       
        if 'image' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['image']
        

        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
           
            model = CatSkinDiseasePredictor('cat_skin_disease_model.pth')
            result = model.predict_image(filepath)
            
            if "error" in result:
                return render_template('index.html', error=result["error"])
            
            
            predicted_class = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']
            
            
            chart_img = create_chart(probabilities)
            
            
            disease_description = disease_info.get(predicted_class, {}).get('description', "No information available.")
            disease_symptoms = disease_info.get(predicted_class, {}).get('symptoms', "No symptom information available.")
            disease_treatment = disease_info.get(predicted_class, {}).get('treatment', "No treatment information available.")
            
            
            return render_template('index.html', 
                                   prediction=predicted_class.replace('_', ' '),
                                   prediction_key=predicted_class,  
                                   confidence=f"{confidence:.1f}",
                                   chart_img=chart_img,
                                   image_url=filepath,
                                   description=disease_description,
                                   symptoms=disease_symptoms,
                                   treatment=disease_treatment)
                
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")
    
    return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
   
    user_feedback = data.get('user_feedback', {})
    confirmed_conditions = ','.join(user_feedback.get('confirmed_conditions', [])) if isinstance(user_feedback, dict) else ''
    doctor_notes = user_feedback.get('notes', '') if isinstance(user_feedback, dict) else ''
    
   
    with open(FEEDBACK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            data['image_path'],
            data['predicted_class'],
            data['probability'],
            confirmed_conditions,
            doctor_notes
        ])
    
    return jsonify({'status': 'success'})

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
