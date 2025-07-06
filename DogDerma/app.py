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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class_names = [
    'Dermatitis',
    'Fungal_infections',
    'Healthy',
    'Hypersensitivity',
    'demodicosis',
    'ringworm'
]

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

FEEDBACK_CSV = 'feedback_data.csv'
if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'image_path', 'predicted_class', 'probability', 'confirmed_conditions', 'doctor_notes', 'doctor_name', 'dog_age'])

disease_info = {
    'Dermatitis': {
        'description': "Dermatitis in dogs is a general term for skin inflammation, leading to redness, itching, and rash. Common causes include allergies (atopy, food), irritants, and secondary bacterial or yeast infections.",
        'treatment': "Treatment often involves identifying and addressing the underlying cause, along with medications like corticosteroids or antihistamines to manage itching and inflammation. Special shampoos and topical treatments are also frequently used. Treating secondary infections is crucial."
    },
    'Fungal_infections': {
        'description': "Fungal infections in dogs can cause a variety of skin issues, with ringworm and yeast dermatitis (Malassezia) being the most common. Symptoms include itching, redness, hair loss, scaling, and a characteristic circular lesion in the case of ringworm. Yeast infections often occur in skin folds.",
        'treatment': "Treatment depends on the specific fungus but typically involves antifungal medications, either topical (creams, shampoos) or oral. Environmental decontamination is important for ringworm."
    },
    'Healthy': {
        'description': "Healthy dog skin is characterized by a soft, pliable texture, a pink or pigmented color (depending on the breed), and a coat that is shiny and free of excessive shedding. There should be no signs of itching, redness, or irritation.",
        'treatment': "Maintaining healthy skin in dogs involves a balanced diet rich in omega-3 fatty acids, regular grooming, appropriate bathing, parasite control, and avoiding allergens and irritants."
    },
    'Hypersensitivity': {
        'description': "Hypersensitivity in dogs refers to allergic reactions of the skin. Common manifestations include atopic dermatitis (environmental allergies), food allergies, and contact allergies. Symptoms include intense itching, scratching, rubbing, chewing, leading to redness, hair loss, secondary infections, and skin lesions.",
        'treatment': "Treatment involves identifying and avoiding the allergen (if possible), managing itching with antihistamines, corticosteroids, or other immunomodulatory drugs (like Apoquel or Cytopoint), and treating secondary infections. Food trials are used for food allergies. Allergy testing and immunotherapy (allergy shots) may be helpful for atopy."
    },
    'demodicosis': {
        'description': "Demodicosis in dogs is caused by Demodex mites. There are two main forms: localized (often seen in puppies, with small, patchy areas of hair loss) and generalized (more severe, potentially indicating an underlying immune deficiency). Symptoms include hair loss, scaling, redness, and secondary bacterial infections.",
        'treatment': "Treatment depends on the form. Localized demodicosis may resolve on its own. Generalized demodicosis requires aggressive treatment with miticidal medications (like oral ivermectin, milbemycin oxime, or topical amitraz), often combined with antibiotics for secondary infections. Underlying health issues need to be addressed."
    },
    'ringworm': {
        'description': "Ringworm in dogs is a fungal infection that affects the skin and hair. It's zoonotic, meaning it can be transmitted to humans. Lesions are often circular, with hair loss, scaling, and redness. It can be itchy, though not always severely.",
        'treatment': "Treatment involves antifungal medications, either topical or oral. Environmental decontamination is crucial to prevent spread. Clipping the hair around lesions can help with topical treatment. Treatment duration is typically prolonged (several weeks to months)."
    }
}

def create_chart(probabilities):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(class_names))
    
    probs_percent = probabilities * 100
    
    bars = ax.barh(y_pos, probs_percent, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ') for name in class_names])
    ax.invert_yaxis()
    ax.set_xlabel('Probability (%)')
    ax.set_title('Disease Probability')
    
    predicted_idx = np.argmax(probabilities)
    bars[predicted_idx].set_color('#4CAF50')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_position = width + 1
        ax.text(label_position, bar.get_y() + bar.get_height()/2, 
                f'{probs_percent[i]:.1f}%', 
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
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
                
                predicted_idx = np.argmax(probabilities)
                predicted_class = class_names[predicted_idx]
                confidence = probabilities[predicted_idx] * 100
                
                chart_img = create_chart(probabilities)
                
                disease_description = disease_info.get(predicted_class, {}).get('description', "No information available.")
                disease_treatment = disease_info.get(predicted_class, {}).get('treatment', "No treatment information available.")
                
                return render_template('index.html', 
                                      prediction=predicted_class,
                                      confidence=f"{confidence:.1f}",
                                      chart_img=chart_img,
                                      image_url=filepath,
                                      description=disease_description,
                                      treatment=disease_treatment)
                
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")
    
    return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    doctor_name = request.form.get('doctor_name', '').strip()
    dog_age = request.form.get('dog_age', '').strip()
    confirmed_conditions = request.form.getlist('conditions')
    doctor_notes = request.form.get('doctor_notes', '').strip()
    image_path = request.form.get('image_path', '')
    predicted_class = request.form.get('predicted_class', '')
    probability = request.form.get('probability', '')

    if not doctor_name or not dog_age:
        return render_template('index.html', error="Doctor's name and dog age are required.")

    with open(FEEDBACK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            image_path,
            predicted_class,
            probability,
            ','.join(confirmed_conditions),
            doctor_notes,
            doctor_name,
            dog_age
        ])

    return render_template('feedback_submitted.html', doctor_name=doctor_name, dog_age=dog_age)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5002, debug=True)
