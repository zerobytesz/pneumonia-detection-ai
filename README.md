🧠 Deep Learning-Based Pneumonia Detection from Chest X-Ray Images

Using DenseNet121 Transfer Learning and Grad-CAM Explainability

⸻

🚀 Project Overview

Pneumonia is a serious respiratory infection responsible for millions of deaths worldwide, particularly in regions with limited access to expert radiologists. Early detection using chest X-ray imaging is critical but often constrained by time, expertise, and availability of medical professionals.

This project presents a deep learning-based automated pneumonia detection system that leverages transfer learning (DenseNet121) and Grad-CAM explainability to provide both accurate predictions and visual interpretability.

The system is designed as an end-to-end clinical decision support tool, capable of:
• Classifying chest X-rays as Normal or Pneumonia
• Providing confidence scores
• Highlighting affected lung regions using Grad-CAM
• Simulating risk adjustments via a custom risk simulation module
• Delivering results through an interactive web dashboard

⸻

🔬 Key Features
• ✅ DenseNet121-based Transfer Learning
• ✅ High Accuracy (~95.35%)
• ✅ AUC-ROC: 0.9862 (excellent separability)
• ✅ Grad-CAM Explainability (visual attention maps)
• ✅ Risk Simulation Module (custom innovation)
• ✅ Interactive Flask-based Dashboard UI
• ✅ Real-time inference on uploaded X-ray images

⸻

🧠 System Architecture

The pipeline consists of the following stages: 1. Data Input
Chest X-ray image (JPEG/PNG) 2. Preprocessing
• Resize to 224×224
• Normalize pixel values
• Data augmentation during training 3. Feature Extraction
• DenseNet121 pretrained on ImageNet
• Fine-tuned for medical imaging 4. Classification Head
• Global Average Pooling
• Dense layers with ReLU activation
• Dropout for regularization
• Sigmoid output (binary classification) 5. Explainability Module
• Grad-CAM applied to last convolutional layer
• Heatmap overlay on original X-ray 6. Simulation Layer
• Adjusts prediction using risk factor input
• Provides sensitivity analysis 7. User Interface
• Upload → Analyze → Visualize results

⸻

📊 Model Performance

The model was evaluated on a held-out test set (624 images).

🔹 Performance Metrics

Metric Score
Accuracy 95.35%
AUC-ROC 0.9862
Precision ~96%
Recall ~95%
F1 Score ~95.5%

⸻

📉 Confusion Matrix

Observations:
• True Negatives (Normal correctly predicted): 222
• True Positives (Pneumonia correctly predicted): 373
• False Positives: 12
• False Negatives: 17

The model demonstrates high sensitivity and specificity, making it suitable for clinical screening.

⸻

📈 ROC Curve

    •	AUC = 0.9862
    •	Indicates near-perfect classification capability

⸻

🔍 Explainability with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) enables the model to highlight regions in the X-ray that influenced its decision.
• Pneumonia cases → localized high-intensity regions (lung infiltrates)
• Normal cases → diffuse or minimal activation

This enhances clinical trust and interpretability.

⸻

🖥️ Web Application (Dashboard)

The system includes a fully functional Flask-based dashboard:

Features:
• Upload X-ray image
• Real-time prediction
• Confidence score display
• Grad-CAM heatmap visualization
• Risk simulation parameter

⸻

⚙️ Tech Stack

Core Technologies:
• Python 3.x
• TensorFlow / Keras
• NumPy

Visualization & Processing:
• OpenCV
• Matplotlib
• Seaborn

Web Framework:
• Flask
• HTML/CSS/JavaScript

⸻

📁 Project Structure

PNEUMONIA_PROJECT/
│
├── app.py # Flask application
├── train.py # Model training
├── evaluate.py # Evaluation metrics & graphs
├── predict.py # Inference logic
├── gradcam.py # Explainability module
├── simulation.py # Risk simulation logic
│
├── model/
│ └── model.h5 # Trained model
│
├── outputs/
│ ├── confusion_matrix.png
│ └── roc_curve.png
│
├── templates/
│ └── index.html # UI
│
├── static/ # UI assets
├── requirements.txt
└── README.md

⸻

▶️ How to Run Locally

1. Clone the repository

git clone https://github.com/zerobytesz/pneumonia-detection-ai.git
cd pneumonia-detection-ai

⸻

2. Install dependencies

pip install -r requirements.txt

⸻

3. Run the application

python app.py

⸻

4. Open in browser

http://127.0.0.1:5000

⸻

📦 Dataset
• Dataset: Chest X-Ray Pneumonia Dataset (Kermany et al.)
• Total Images: 5,856
• Classes: Normal, Pneumonia

⚠️ Dataset is not included due to size constraints.

⸻

⚠️ Notes
• Pre-trained model (model.h5) is included for direct inference
• No retraining required to run the application
• Designed for academic and research use

⸻

💡 Use Case

This system can serve as a clinical decision support tool for:
• Early pneumonia screening
• Assisting radiologists
• Deployment in low-resource healthcare settings

⸻

🔮 Future Scope
• Multi-class classification (bacterial vs viral pneumonia)
• Integration with hospital PACS systems
• Mobile / edge deployment
• Ensemble models (DenseNet + EfficientNet)
• Real-time clinical validation

⸻

👥 Authors
• Raghav Tyagi
• Patil Tejas Ashok

⸻

📜 License

This project is intended for academic and research purposes only.

⸻

⭐ Acknowledgment

We thank VIT University and faculty for their support and guidance in completing this project.

⸻

🚀 Final Note

This project combines deep learning performance with explainable AI, bridging the gap between black-box models and clinical usability.
:::
