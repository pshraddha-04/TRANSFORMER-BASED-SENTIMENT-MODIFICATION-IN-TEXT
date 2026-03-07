"""
Emotion Detection Module
========================
Part of: A Transformer-Based Framework for Controlled Sentiment Transformation in Text
Group 13 | VIT | AIDS Sem II 2025-26

Uses: j-hartmann/emotion-english-distilroberta-base
Detects: anger, disgust, fear, joy, neutral, sadness, surprise
"""

import gradio as gr
import torch
from transformers import pipeline
import plotly.graph_objects as go
import pandas as pd

# 1. Load Model (cached after first run)

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

print("Loading emotion detection model...")
emotion_classifier = pipeline(
    task="text-classification",
    model=EMOTION_MODEL,
    top_k=None,          # returns ALL emotion scores
    device=0 if torch.cuda.is_available() else -1
)
print("Model loaded successfully!")

# Emoji map for each emotion
EMOTION_META = {
    "anger":    {"emoji": "😠", "color": "#e74c3c", "target_tone": "Neutral"},
    "disgust":  {"emoji": "🤢", "color": "#8e44ad", "target_tone": "Neutral"},
    "fear":     {"emoji": "😨", "color": "#e67e22", "target_tone": "Calm"},
    "joy":      {"emoji": "😄", "color": "#2ecc71", "target_tone": "Professional"},
    "neutral":  {"emoji": "😐", "color": "#95a5a6", "target_tone": "Professional"},
    "sadness":  {"emoji": "😢", "color": "#3498db", "target_tone": "Motivational"},
    "surprise": {"emoji": "😲", "color": "#f1c40f", "target_tone": "Neutral"},
}

# 2. Core Detection Logic

def detect_emotion(text: str):
    """Run emotion classification and return structured results."""
    if not text or not text.strip():
        return None, None, None, None

    results = emotion_classifier(text)[0]  # list of {label, score}
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    top_emotion = results_sorted[0]["label"]
    top_score   = results_sorted[0]["score"]
    meta        = EMOTION_META.get(top_emotion, {})

    # ── Summary card (HTML) ──
    emoji        = meta.get("emoji", "")
    target_tone  = meta.get("target_tone", "Neutral")
    bar_color    = meta.get("color", "#555")
    confidence   = f"{top_score * 100:.1f}%"

    summary_html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; padding: 20px;
                border-radius: 12px; background: #1e1e2e; color: #cdd6f4;
                border-left: 5px solid {bar_color}; margin-bottom: 10px;">
        <div style="font-size: 2.2em; margin-bottom: 6px;">{emoji}</div>
        <div style="font-size: 1.5em; font-weight: 700; color: {bar_color};">
            {top_emotion.upper()}
        </div>
        <div style="font-size: 1em; color: #a6adc8; margin-top: 4px;">
            Confidence: <strong style="color:#cdd6f4">{confidence}</strong>
        </div>
        <div style="margin-top: 12px; padding: 8px 14px; background: #313244;
                    border-radius: 8px; font-size: 0.95em;">
            💡 Suggested rewrite target: 
            <strong style="color: {bar_color};">{target_tone}</strong>
        </div>
    </div>
    """

    # ── Plotly bar chart ──
    labels = [r["label"].capitalize() for r in results_sorted]
    scores = [r["score"] * 100 for r in results_sorted]
    colors = [EMOTION_META.get(r["label"], {}).get("color", "#888") for r in results_sorted]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        title="Emotion Confidence Scores",
        xaxis=dict(title="Confidence (%)", range=[0, 110], showgrid=True,
                   gridcolor="#2a2a3e", color="#aaa"),
        yaxis=dict(color="#aaa", autorange="reversed"),
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4", size=13),
        margin=dict(l=10, r=40, t=50, b=40),
        height=320
    )

    # ── DataFrame for raw scores ──
    df = pd.DataFrame({
        "Emotion":    [r["label"].capitalize() for r in results_sorted],
        "Score (%)":  [f"{r['score']*100:.2f}%" for r in results_sorted],
        "Target Tone": [EMOTION_META.get(r["label"], {}).get("target_tone", "-")
                        for r in results_sorted]
    })

    return summary_html, fig, df, top_emotion


# 3. Gradio UI

EXAMPLES = [
    ["I can't believe you did this again! This is completely unacceptable!"],
    ["I'm so tired. Nothing seems to work out for me lately."],
    ["Please find the attached report for your review."],
    ["Oh wow, I had no idea this was possible! That's incredible!"],
    ["I feel a bit anxious about tomorrow's presentation."],
]

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    title="Emotion Detector | Group 13 VIT"
) as demo:

    gr.Markdown("""
    # 🎭 Emotion Detection Module
    **Project:** A Transformer-Based Framework for Controlled Sentiment Transformation  
    **Group 13 | VIT AIDS | Sem II 2025–26**
    
    > Detects emotion in text as the first step before tone rewriting.  
    > Model: `j-hartmann/emotion-english-distilroberta-base` (DistilRoBERTa, 7 emotions)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Enter Text",
                placeholder="Type or paste any text here...",
                lines=5,
                max_lines=10
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 Detect Emotion", variant="primary", scale=2)
                clear_btn  = gr.Button("🗑️ Clear", scale=1)

            gr.Examples(examples=EXAMPLES, inputs=input_text, label="Try an Example")

        with gr.Column(scale=3):
            summary_out = gr.HTML(label="Detected Emotion")
            chart_out   = gr.Plot(label="Emotion Scores")

    with gr.Accordion("📊 Raw Score Table", open=False):
        table_out = gr.Dataframe(
            headers=["Emotion", "Score (%)", "Target Tone"],
            label="All Emotion Scores"
        )

    gr.Markdown("""
    ---
    ### 🗺️ What's Next in the Pipeline?
    | Step | Module | Status |
    |------|--------|--------|
    | 1 | **Emotion Detection** (this module) | ✅ Current |
    | 2 | Controlled Text Generation (T5/GPT-2) | 🔜 Next |
    | 3 | BLEU + Emotion Alignment Evaluation | 🔜 Upcoming |
    | 4 | Sentiment Shift Visualization | 🔜 Upcoming |
    """)

    # ── Wire up events ──
    submit_btn.click(
        fn=detect_emotion,
        inputs=input_text,
        outputs=[summary_out, chart_out, table_out, gr.State()]
    )
    clear_btn.click(
        fn=lambda: ("", None, None, None, None),
        outputs=[input_text, summary_out, chart_out, table_out, gr.State()]
    )
    input_text.submit(
        fn=detect_emotion,
        inputs=input_text,
        outputs=[summary_out, chart_out, table_out, gr.State()]
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
