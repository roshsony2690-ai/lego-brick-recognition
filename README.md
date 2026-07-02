#  Automated LEGO Brick Recognition System

A deep learning-based system that automatically identifies LEGO bricks from images using a ResNet50 model with transfer learning and confidence calibration.

## 📸 Demo

### Home Page
Upload a LEGO brick photo for instant identification.

![Home Page]
<img width="1919" height="1029" alt="Screenshot 2026-07-02 095448" src="https://github.com/user-attachments/assets/3f10907c-2dde-41c6-85cf-c96abe92ed29" />


### Prediction Result
Get instant predictions with confidence scores and top 3 alternatives.

![Prediction Result]
<img width="1919" height="910" alt="Screenshot 2026-07-02 095547" src="https://github.com/user-attachments/assets/c4ab6150-cbe5-41f5-9d74-f73ed571d3b5" />


### Batch Processing
Process multiple images at once and download results as CSV.

![Batch Processing]
<img width="1919" height="953" alt="Screenshot 2026-07-02 095733" src="https://github.com/user-attachments/assets/cd371aa7-30e0-4eaa-977a-4b618927f855" />


### Dashboard
View dataset statistics and class distribution.

![Dashboard]
<img width="1919" height="960" alt="Screenshot 2026-07-02 095749" src="https://github.com/user-attachments/assets/54ee1ebc-d303-4afe-b6e0-15e6b8c95058" />


## ✨ Features

- **Single Image Recognition** - Upload one LEGO brick photo and get instant identification
- **Batch Processing** - Upload multiple images at once and download results as CSV
- **Confidence Calibration** - Realistic confidence scores (no overconfidence!)
- **Interactive Dashboard** - View dataset statistics and class distribution
- **LEGO-Themed UI** - Beautiful interface with LEGO branding
- **16 LEGO Brick Classes** - Recognizes 16 different brick types

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                LEGO BRICK RECOGNITION SYSTEM                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📸 Step 1: Image Upload                                         │
│      User uploads a LEGO brick image                             │
│                            │                                     │
│                            ▼                                     │
│  🖼️ Step 2: Image Preprocessing                                 │
│      • Resize to 128 × 128 pixels                                │
│      • Normalize pixel values                                    │
│                            │                                     │
│                            ▼                                     │
│  🧠 Step 3: Feature Extraction & Classification                  │
│      ResNet50 (Transfer Learning)                                │
│      • Extracts visual features                                  │
│      • Predicts brick category                                   │
│                            │                                     │
│                            ▼                                     │
│  📊 Step 4: Confidence Calibration                               │
│      Temperature Scaling (T = 1.5)                               │
│      Produces better-calibrated confidence scores                │
│                            │                                     │
│                            ▼                                     │
│  🏆 Step 5: Prediction Output                                    │
│      • Predicted LEGO Brick Class                                │
│      • Confidence Percentage                                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| **Deep Learning** | TensorFlow, Keras, ResNet50 |
| **Web Interface** | Streamlit |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Plotly |
| **Backend** | Python 3.11 |

## 📁 Project Structure

```
lego-brick-recognition/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
├── images/               # Screenshots
│   ├── home.png
│   ├── prediction.png
│   ├── batch.png
│   └── dashboard.png
├── dataset/              # Training data (16 classes)
│   ├── 3005 Brick 1x1/
│   ├── 3024 Plate 1x1/
│   └── ...
├── src/                  # Source code
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
└── models/               # Trained models (Git LFS)
    └── lego_model_final.h5
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/roshsony2690-ai/lego-brick-recognition.git
cd lego-brick-recognition
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)
```bash
python src/train.py
```

### 5. Run the App
```bash
streamlit run app.py
```

## 🎯 How to Use

### Single Image Prediction
1. Open the app in your browser
2. Click "Browse files" and select a LEGO brick image
3. View the prediction with confidence score
4. See top 3 predictions with progress bars

### Batch Processing
1. Go to the "Batch Processing" tab
2. Upload multiple images or enter a folder path
3. Click "Process Batch"
4. Download results as CSV

### Dashboard
1. View total classes and model information
2. See class distribution charts
3. Monitor system status

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Architecture** | ResNet50 (Transfer Learning) |
| **Input Size** | 128x128 pixels |
| **Classes** | 16 LEGO brick types |
| **Validation Accuracy** | ~85% |
| **Temperature Scaling** | 1.5 (Confidence Calibration) |

## 🧱 Recognized LEGO Bricks

The system can identify these 16 LEGO brick types:
- 11214 Bush 3M friction with Cross axle
- 18651 Cross Axle 2M with Snap friction
- 2357 Brick corner 1x2x2
- 3003 Brick 2x2
- 3004 Brick 1x2
- 3005 Brick 1x1
- 3022 Plate 2x2
- 3023 Plate 1x2
- 3024 Plate 1x1
- 3040 Roof Tile 1x2x45deg
- 3069 Flat Tile 1x2
- 3673 Peg 2M
- 3713 Bush for Cross Axle
- 3794 Plate 1X2 with 1 Knob
- 6632 Technic Lever 3M
- 32123 half Bush

## 🔧 Key Technical Features

### Transfer Learning with ResNet50
- Pre-trained on ImageNet (1.2 million images)
- Fine-tuned for LEGO brick recognition
- Fast training with limited data

### Confidence Calibration
- Temperature scaling (T=1.5)
- Prevents overconfidence
- Realistic confidence scores

### Data Augmentation
- Rotation (30°)
- Zoom (30%)
- Brightness variation
- Horizontal flip
- Shear transformation

### Regularization
- Dropout (0.5, 0.3)
- L2 regularization
- Batch Normalization
- Early Stopping

## 🚀 Deployment

The app is deployed on **Streamlit Community Cloud**:
```
https://roshsony2690-ai-lego-brick-recognition.streamlit.app
```

## 📊 Future Improvements

- [ ] Add more LEGO brick classes (50+)
- [ ] Mobile app deployment
- [ ] Real-time video recognition
- [ ] 3D printing integration
- [ ] Improved accuracy with larger images (224x224)

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

**Rosh Sony**
- GitHub: [@roshsony2690-ai](https://github.com/roshsony2690-ai)

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning
- Streamlit for the web interface
- ResNet50 pre-trained weights from ImageNet
- LEGO for the inspiration!

---
