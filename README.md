# license_detection

# 🚗 Intelligent License Plate Reader & Vehicle Insights Dashboard

## 📌 Overview
This project is an AI-powered License Plate Recognition System that detects vehicles from images, extracts license plate numbers using OCR, corrects text using Transformer models, stores results in a database, and visualizes vehicle insights in a dashboard.

---

## 🔥 Tech Stack
- YOLOv8 (Object Detection)
- EasyOCR (Optical Character Recognition)
- Hugging Face Transformers (Text Correction)
- SQLite (Database)
- Streamlit (Frontend)
- Streamlit Cloud (Deployment)

---

## 🚀 Features
- Real-time vehicle detection
- License plate text extraction
- OCR error correction using Transformer
- Vehicle visit count analytics
- Frequent vehicle detection
- Time trend visualization

---

## 🏗️ Model Architecture
Image → YOLOv8 → Plate Crop → EasyOCR → Text Cleaning → Transformer Correction → Database → Streamlit Dashboard

---

## 📊 Sample Output
Detected Plate: TN01AB8594

Dashboard shows:
- Most frequent vehicles
- Entry time trends
- Total visits per vehicle

---

## ☁️ Deployment
Deployed on Streamlit Cloud.
