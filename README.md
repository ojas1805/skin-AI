# SkinAI 🧴 — AI-Powered Skin Analysis System

> An end-to-end AI skin analysis web app that detects skin tone, analyzes skin conditions across 5 facial zones, and generates personalized skincare recommendations — built with a focus on South Asian and diverse skin tones.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🔍 What Problem Does This Solve?

Existing skin analysis tools like Minimalist's "Skin Insights" have key limitations:

| Problem | Our Solution |
|---------|-------------|
| Single photo, any lighting → poor accuracy | Live camera with real-time quality checks |
| Bias toward lighter skin tones | Built with Fitzpatrick IV–VI (Indian skin) in focus |
| One-shot analysis, no tracking | Weekly rescan + progress tracking |
| Recommends only brand's own products | Ingredient-first recommendations with reasoning |
| Photo uploaded to cloud servers | On-device face detection via MediaPipe |
| Generic result language | LLM-generated personalized explanation |

---

## ✨ Features

- **Face Detection** — MediaPipe FaceMesh with 468 facial landmarks
- **Skin Tone Classification** — Fitzpatrick scale I–VI using ITA angle in LAB colorspace
- **Undertone Detection** — Warm / Cool / Neutral
- **Acne Detection** — Custom-trained YOLOv8 model
- **5 Skin Conditions** analyzed across 5 facial zones:
  - Acne · Pores · Oiliness · Pigmentation · Dryness
- **AI Recommendations** — LLM-powered RAG pipeline (LLaMA 3.3 via Groq)
  - Products to use with specific reasons
  - Products to avoid with explanations
  - Full morning and night routine
- **Web UI** — Clean, responsive interface built with FastAPI + HTML/CSS

---

## 🧠 How It Works

```
Photo Input
    ↓
MediaPipe FaceMesh (468 landmarks)
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│  Skin Tone CV   │  Condition CV    │  Face Structure │
│  ITA Angle      │  YOLOv8 Acne     │  Zone Mapping   │
│  Fitzpatrick    │  Pores/Oiliness  │  5 Face Zones   │
│  I–VI           │  Pigmentation    │                 │
└─────────────────┴──────────────────┴─────────────────┘
    ↓
Feature Fusion → Skin Profile JSON
    ↓
RAG Pipeline (ChromaDB + Ingredient KB)
    ↓
LLM (LLaMA 3.3 70B via Groq API)
    ↓
Personalized Recommendations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Face Detection | MediaPipe FaceMesh |
| Acne Detection | YOLOv8 (custom trained) |
| Skin Analysis | OpenCV, NumPy |
| Recommendation | LangChain, Groq API, LLaMA 3.3 |
| Backend | FastAPI, Python 3.11 |
| Frontend | HTML5, CSS3, JavaScript |
| Model Training | Google Colab, Ultralytics |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Mac / Linux / Windows
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ojas1805/skin-AI.git
cd skin-AI

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### Run

```bash
uvicorn src.api:app --reload
```

Open your browser at `http://localhost:8000`

---

## 📁 Project Structure

```
skin-AI/
├── src/
│   ├── api.py              # FastAPI routes
│   ├── face_detector.py    # MediaPipe face detection
│   ├── skin_tone.py        # Fitzpatrick + ITA angle
│   ├── skin_condition.py   # YOLOv8 + CV condition analysis
│   ├── recommender.py      # LLM recommendation engine
│   └── ingredients_db.py   # Skincare ingredient knowledge base
├── static/
│   └── style.css           # Frontend styles
├── templates/
│   └── index.html          # Web UI
├── best.pt                 # Trained YOLOv8 acne model
├── main.py                 # CLI entry point
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance

The YOLOv8 acne detection model was trained on Google Colab (T4 GPU):

| Metric | Score |
|--------|-------|
| mAP50 | 0.995 |
| Precision | 0.995 |
| Recall | 0.992 |
| Training epochs | 30 |

---

## 🎯 Roadmap

- [ ] Progress tracking (rescan every 2 weeks, show % improvement)
- [ ] Live camera with real-time quality guide
- [ ] India-first diverse skin tone dataset
- [ ] Shopify embed widget for D2C brands
- [ ] Brand analytics dashboard
- [ ] On-device inference (no photo sent to server)

---

## 💡 Business Application

This system is designed to be sold as a B2B SaaS product to D2C skincare brands (Minimalist, Mamaearth, Dot & Key, etc.) as a white-label skin analysis widget that:

- Embeds on any Shopify store in 10 minutes
- Increases product page conversion by recommending the right SKUs
- Gives brands analytics on their customers' skin profiles
- Differentiates from competitors with explainable AI recommendations

---

## 👤 Built By

**Ojas Singh**
B.Tech Computer Science — Bennett University
[GitHub](https://github.com/ojas1805) · [LinkedIn](https://linkedin.com/in/ojas-singh)

---

## 📄 License

MIT License — feel free to use, modify and distribute.