# SkinAI — AI-Powered Skin Analysis System

An end-to-end AI skin analysis web app that detects skin tone, analyzes skin conditions, and generates personalized skincare recommendations.

## What it does

- Detects face using MediaPipe (468 landmarks)
- Classifies skin tone using Fitzpatrick scale I–VI
- Detects acne using a custom-trained YOLOv8 model
- Analyzes 5 skin conditions across 5 face zones
- Generates personalized product recommendations using LLM

## Tech Stack

- **CV layer**: MediaPipe, YOLOv8, OpenCV
- **NLP layer**: Groq API (LLaMA 3.3), RAG pipeline
- **Backend**: FastAPI, Python 3.11
- **Frontend**: HTML, CSS, JavaScript

## Setup

1. Clone the repo
2. Create virtual environment
3. Install dependencies
4. Add API keys
5. Run the app

\`\`\`bash
git clone https://github.com/ojas1805/skin-ai.git
cd skin-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

Create a \`.env\` file:
\`\`\`
GROQ_API_KEY=your_key_here
\`\`\`

Run:
\`\`\`bash
uvicorn src.api:app --reload
\`\`\`

Open \`http://localhost:8000\`

## Built by
Ojas Singh — B.Tech CS, Bennett University
