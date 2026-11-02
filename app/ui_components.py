"""
Streamlit UI Components & Custom Styling
Provides clean agronomic cards, metric displays, and CSS layout styling for the web demo.
"""

import textwrap
from typing import Dict, Optional
import streamlit as st

CUSTOM_CSS = textwrap.dedent("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    .crop-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #dce4ec;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        margin-bottom: 16px;
    }
    .badge-healthy {
        background-color: #d4edda;
        color: #155724;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.85rem;
    }
    .badge-diseased {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.85rem;
    }
    .badge-robust {
        background-color: #e3f2fd;
        color: #0d47a1;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
    }
    .badge-base {
        background-color: #fff3cd;
        color: #856404;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
    }
    .treatment-box {
        background-color: #f4faf6;
        border-left: 4px solid #28a745;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 12px;
        color: #1c3d28;
        font-size: 0.92rem;
    }
    .prevention-box {
        background-color: #f0f7fb;
        border-left: 4px solid #17a2b8;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 10px;
        color: #113c4c;
        font-size: 0.92rem;
    }
</style>
""")

def inject_custom_css():
    """Inject styling into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_diagnosis_card(
    crop: str,
    condition: str,
    status: str,
    confidence: float,
    symptoms: str,
    treatment: str,
    prevention: str,
    is_robust_model: bool = True
):
    """Render a diagnosis card with clinical agronomic guidance."""
    badge_cls = "badge-healthy" if status.lower() == "healthy" else "badge-diseased"
    model_tag_cls = "badge-robust" if is_robust_model else "badge-base"
    model_tag = "Robust MobileNetV2 (Attention + Aug)" if is_robust_model else "Standard Baseline MobileNetV2"
    
    card_html = textwrap.dedent(f"""
<div class="crop-card">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span class="{model_tag_cls}">{model_tag}</span>
<span class="{badge_cls}">{status.upper()}</span>
</div>
<h3 style="margin-top: 4px; margin-bottom: 4px; color: #1e392a; font-size: 1.3rem;">{crop} - {condition}</h3>
<p style="color: #6c757d; font-size: 0.95rem; margin-bottom: 10px;">Confidence: <b>{confidence * 100:.1f}%</b></p>
<p style="margin-bottom: 8px; font-size: 0.93rem; color: #2c3e50;"><b>Botanical Symptoms:</b><br>{symptoms}</p>
<div class="treatment-box">
<b>Recommended Agronomic Treatment:</b><br>{treatment}
</div>
<div class="prevention-box">
<b>Preventative Management:</b><br>{prevention}
</div>
</div>
""")
    st.markdown(card_html, unsafe_allow_html=True)
