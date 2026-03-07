# 🎭 Emotion Detection Module
### Group 13 | VIT AIDS | Sem II 2025–26

---

## 📦 Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```
Then open → **http://localhost:7860**

---

## 🧠 Model Used

| Property | Detail |
|----------|--------|
| Model | `j-hartmann/emotion-english-distilroberta-base` |
| Base | DistilRoBERTa |
| Emotions | anger, disgust, fear, joy, neutral, sadness, surprise |
| Source | HuggingFace 🤗 |

The model is **auto-downloaded** on first run (~300MB). Cached locally after that.

---

## 🗺️ Project Pipeline (All Modules)

```
Input Text
    │
    ▼
┌─────────────────────┐
│  Emotion Detection  │  ◄── You are here (Module 1)
│  (DistilRoBERTa)    │
└────────┬────────────┘
         │ detected emotion + confidence
         ▼
┌─────────────────────┐
│  Tone Rewriter      │  (Module 2 — T5 / GPT-2)
│  Angry → Neutral    │
│  Sad → Motivational │
└────────┬────────────┘
         │ rewritten text
         ▼
┌─────────────────────┐
│  Evaluation         │  (Module 3)
│  BLEU + Alignment   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Visualization      │  (Module 4)
│  Sentiment Shift    │
└─────────────────────┘
```

---

## 📁 File Structure

```
emotion_detector/
├── app.py              ← Main Gradio app (this module)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 📊 Output

- **Detected emotion** with confidence score
- **Bar chart** of all 7 emotion probabilities
- **Suggested rewrite target** tone
- **Raw score table** for all emotions
