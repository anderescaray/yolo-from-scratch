# 🛒 AutoCheckout: Self-Checkout System with YOLO Object Detection

> Automated retail checkout system using custom-trained YOLO with active learning and semi-supervised training

![Demo GIF - grabación de tu webcam detectando productos]

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

An end-to-end computer vision system that enables **automatic product detection and pricing** for self-checkout applications. Built from scratch using YOLO architecture with a novel training approach combining:

- **Transfer Learning** from generic retail dataset
- **Active Learning** for efficient data labeling
- **Semi-Supervised Learning** to leverage unlabeled data
- **Clustering in latent space** for similar product grouping

### Real-World Application
Webcam-based detection system that automatically:
1. Detects products passing through checkout
2. Identifies product type
3. Adds to shopping cart with pricing
4. Fully containerized for deployment anywhere

---

## 🏗️ System Architecture
```
[Webcam Input] → [YOLO Detector] → [Product Classifier] → [Shopping Cart + Pricing]
                        ↓
            [Active Learning Pipeline]
                        ↓
            [Semi-Supervised Training]
                        ↓
            [Model Retraining Loop]
```

---

## 🧠 Technical Approach

### Phase 1: Base Model Training
- Custom YOLO implementation from scratch
- Initial training on generic supermarket dataset (Roboflow)
- Baseline detection capabilities established

### Phase 2: Custom Dataset Creation (Active Learning)
1. **High-confidence predictions** (>0.9) → Auto-labeled for training
2. **Low-confidence predictions** (<0.9) → Flagged for manual review
3. **Latent space clustering**: Group similar unlabeled products
4. **Manual labeling**: Only label cluster representatives (efficient!)
5. **Propagate labels**: Similar products get same label automatically

### Phase 3: Transfer Learning
- Modified final layer for custom product categories
- Fine-tuned on curated dataset
- Iterative improvement through active learning loop

### Phase 4: Deployment (MLOps)
- Dockerized application
- Real-time webcam inference
- Shopping cart logic with pricing dictionary

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Webcam

### Run the Application
```bash
# Clone the repository
git clone https://github.com/tu-usuario/auto-checkout-yolo.git
cd auto-checkout-yolo

# Build and run
docker-compose up

# Access the application
# Camera feed with detections: http://localhost:8000
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **mAP@0.5** | X.XX |
| **Inference Time** | XX ms |
| **Products Detected** | XX categories |
| **Training Efficiency** | XX% reduction in labeling time (active learning) |
| **Model Size** | XX MB |

### Key Achievements
✅ Custom YOLO trained from scratch  
✅ Active learning reduced manual labeling by XX%  
✅ Semi-supervised learning improved accuracy by XX%  
✅ Real-time detection (>30 FPS) on CPU  
✅ Production-ready Docker deployment  

---

## 🛠️ Tech Stack

**Deep Learning:**
- PyTorch
- Custom YOLO implementation
- Transfer Learning

**Data Pipeline:**
- Active Learning framework
- Semi-Supervised Learning
- K-Means clustering for latent space analysis
- Roboflow (initial dataset)

**MLOps:**
- Docker & Docker Compose
- FastAPI (inference API)
- OpenCV (webcam integration)
- Model versioning

**Tools:**
- Python 3.9+
- NumPy, Pandas
- Matplotlib, Seaborn

---

## 📁 Project Structure
```
auto-checkout-yolo/
├── README.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   ├── model/
│   │   ├── yolo.py              # YOLO architecture
│   │   └── train.py             # Training pipeline
│   ├── active_learning/
│   │   ├── clustering.py        # Latent space clustering
│   │   └── labeling.py          # Active learning logic
│   ├── inference/
│   │   ├── detector.py          # Real-time detection
│   │   └── api.py               # FastAPI endpoints
│   └── utils/
│       └── pricing.py           # Product pricing dictionary
├── data/
│   ├── initial_dataset/         # Roboflow generic dataset
│   └── custom_dataset/          # Actively learned dataset
├── models/
│   └── checkpoints/             # Model versions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_active_learning_analysis.ipynb
│   └── 03_results_visualization.ipynb
└── requirements.txt
```

---

## 🔬 Active Learning Process

### 1. Initial Predictions
```python
# Model predicts on unlabeled data
predictions = model.predict(unlabeled_data)

# High confidence → Auto-label
auto_labeled = predictions[predictions.confidence > 0.9]

# Low confidence → Manual review needed
review_needed = predictions[predictions.confidence < 0.9]
```

### 2. Latent Space Clustering
```python
# Extract features from CNN backbone
features = model.extract_features(review_needed)

# Cluster similar products
clusters = kmeans.fit_predict(features)

# Label one representative per cluster
# Propagate label to all cluster members
```

### 3. Iterative Training
- Retrain with newly labeled data
- Repeat until desired accuracy
- **Result**: XX% less manual labeling vs. full dataset annotation

---

## 💡 Key Innovations

1. **Custom YOLO from Scratch**: Deep understanding of architecture
2. **Hybrid Learning**: Combines active + semi-supervised approaches
3. **Efficient Labeling**: Clustering reduces annotation effort
4. **Production-Ready**: Docker ensures portability

---

## 🎥 Demo

[Insertar GIF o video de la webcam detectando productos]
```
🛒 Shopping Cart:
- Coca Cola x2 ......... 2.50€
- Bread ................ 1.20€
- Apple x3 ............. 3.60€
----------------------------
TOTAL .................. 7.30€
```

---

## 📈 Future Improvements

- [ ] Add barcode detection as fallback
- [ ] Multi-object tracking for moving products
- [ ] Cloud deployment (AWS/GCP)
- [ ] Mobile app integration
- [ ] Expand to 100+ product categories

---

## 🎓 Academic Context

This project was developed as a **Bachelor's Thesis** in Data Science at Universidad Pública de Navarra (UPNA), 2025.

**Advisor:** [Nombre del tutor]  
**Grade:** [Cuando lo sepas]

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🤝 Contact

**[Tu Nombre]**  
📧 [tu-email]  
💼 [LinkedIn](tu-linkedin)  
🐙 [GitHub](tu-github)

---

## 🙏 Acknowledgments

- Initial dataset: [Roboflow Supermarket Dataset](link)
- YOLO paper: [Redmon et al., 2016](link)
- Active Learning framework inspired by [paper/resource]

---

**⭐ If you found this project interesting, please star the repository!**