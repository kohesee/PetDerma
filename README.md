# 🐶 PetDerma - AI-Powered Pet Skin Disease Diagnosis Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-blue.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-blue.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

PetDerma is an advanced AI-powered web application designed to diagnose skin diseases in pets using computer vision and deep learning technologies. The platform features specialized diagnostic modules for both cats and dogs, providing accurate disease detection with confidence scores and treatment recommendations.

## ✨ Key Features

### 🐱 **CatDerma Module**
- **Disease Detection**: Flea Allergy, Healthy Skin, Ringworm, Scabies
- **AI Model**: ResNet-50 based classification
- **Real-time Analysis**: Instant diagnostic results with confidence scores
- **Visual Feedback**: Probability distribution charts for all conditions

### 🐕 **DogDerma Module**
- **Disease Detection**: Dermatitis, Fungal Infections, Healthy Skin, Hypersensitivity, Demodicosis, Ringworm
- **Advanced Diagnostics**: Comprehensive skin condition analysis
- **Treatment Guidance**: Detailed information about each condition
- **User Feedback System**: Integrated feedback collection for model improvement

## 🛠️ Technology Stack

### **Datasets**
- **Dog Skin Dataset**: [Dog Skin Disease v3](https://www.kaggle.com/datasets/vekified/dog-skin-disease-v3)
- **Cat Skin Dataset**: [Cat Skin Disease v3](https://www.kaggle.com/datasets/vekified/cat-skin-disease-v3)

### **Backend Technologies**
- **Python 3.8+**: Core programming language
- **Flask 3.0+**: Web framework for API and routing
- **PyTorch 2.0+**: Deep learning framework for AI models

### **AI/ML Libraries**
- **torchvision**: Computer vision transformations and models
- **ResNet-50**: Pre-trained convolutional neural network architecture
- **PIL (Pillow)**: Image processing and manipulation
- **NumPy**: Numerical computing and array operations

### **Data Science & Visualization**
- **Matplotlib**: Statistical plotting and visualization
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities

### **Frontend Technologies**
- **HTML5/CSS3**: Modern web standards
- **Responsive Design**: Mobile-first approach

## 📁 Project Structure

```
PetDerma/
├── app.py                          # Main application launcher
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── templates/
│   └── index.html                  # Main landing page
├── static/                         # Static assets
├── CatDerma/                       # Cat skin disease module
│   ├── app.py                      # CatDerma Flask application
│   ├── cat_skin_disease_model.pth  # Trained PyTorch model
│   ├── cat_skin_model.ipynb        # Training Dataset to get model
│   ├── feedback_data.csv           # User feedback storage
│   ├── requirements.txt            # Module-specific dependencies
│   ├── templates/
│   │   ├── index.html              # CatDerma interface
│   │   ├── home.html               # Results display
│   │   └── about.html              # Information page
│   └── static/uploads/             # Uploaded images storage
└── DogDerma/                       # Dog skin disease module
    ├── app.py                      # DogDerma Flask application
    ├── best_model.pth              # Trained PyTorch model
    ├── dog_skin_model.ipynb        # Training Dataset to get model
    ├── feedback_data.csv           # User feedback storage
    ├── requirements.txt            # Module-specific dependencies
    ├── templates/
    │   ├── index.html              # DogDerma interface
    │   ├── home.html               # Results display
    │   └── about.html              # Information page
    └── static/uploads/             # Uploaded images storage
```

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### **Step 1: Clone the Repository**
```bash
git clone <repository-url>
cd PetDerma
```

### **Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
# Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install required packages
pip install -r requirements.txt
```

## 🎮 Running the Application

### **Launch Application**
```bash
python app.py
```
This command will:
- Start the main PetDerma application on `http://localhost:5000`
- Automatically launch CatDerma on `http://localhost:5001`
- Automatically launch DogDerma on `http://localhost:5002`

## 📖 Usage Guide

### **Getting Started**
1. **Access the Platform**: Navigate to `http://localhost:5000`
2. **Choose Your Pet**: Select either "Cat Diagnosis" or "Dog Diagnosis"
3. **Upload Image**: Select a clear image of your pet's skin condition
4. **Get Results**: View diagnostic results with confidence scores
5. **Review Information**: Read detailed condition descriptions and treatment advice

### **Best Practices for Image Upload**
- **Image Quality**: Use high-resolution, well-lit images
- **Focus Area**: Ensure the affected skin area is clearly visible
- **File Formats**: Supports JPG, JPEG, PNG formats
- **Image Size**: Optimal size is 224x224 pixels (automatically resized)

### **Understanding Results**
- **Confidence Score**: Percentage indicating model certainty
- **Probability Distribution**: Visual chart showing likelihood of each condition
- **Condition Information**: Detailed descriptions, symptoms, and treatment options
- **Report Maker**: Select respective options and Generate Report

## 🧪 Model Information

### **Architecture**
- **Base Model**: ResNet-50 (pre-trained on ImageNet)
- **Custom Classification Layer**: Adapted for pet skin conditions
- **Input Size**: 224x224 RGB images
- **Normalization**: ImageNet standard normalization

### **Cat Disease Classes**
1. **Flea Allergy**: Allergic reaction to flea saliva
2. **Healthy**: Normal, healthy skin condition
3. **Ringworm**: Fungal infection affecting skin and hair
4. **Scabies**: Parasitic mite infestation

### **Dog Disease Classes**
1. **Dermatitis**: Inflammatory skin condition
2. **Fungal Infections**: Various fungal skin diseases
3. **Healthy**: Normal, healthy skin condition
4. **Hypersensitivity**: Allergic skin reactions
5. **Demodicosis**: Demodex mite infestation
6. **Ringworm**: Fungal infection affecting skin and hair

## 🤝 Contributing

We welcome contributions to improve PetDerma! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: PetDerma is designed to assist in identifying potential skin conditions but should not replace professional veterinary diagnosis and treatment. Always consult with a qualified veterinarian for proper medical advice and treatment of your pet's health conditions.

**Made with ❤️ for pet health and well-being**
