"""
Gemini AI Explanation Service Module
====================================
Provides an optional explanation layer using the Gemini API.
Translates ML predictions and SHAP attributions into clear, professional
civil and materials engineering insights.

Security & Operational Integrity:
- Never hardcodes or prints API keys.
- Loads GEMINI_API_KEY securely from .env, os.environ, or st.secrets.
- Does NOT alter, recalculate, or generate numerical ML predictions.
- Handles missing keys, network timeouts, 503 service overloads, and API errors gracefully without crashing.
"""

import os
import json
import urllib.request
import urllib.error

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_gemini_api_key():
    """
    Retrieve Gemini API key securely without exposing it.
    Checks environment variables first, then streamlit secrets safely.
    """
    # 1. Environment variable check (.env or OS environment) FIRST
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    # 2. Streamlit secrets check (safely)
    try:
        import streamlit as st
        # Safely inspect st.secrets without throwing missing secrets file warning
        if hasattr(st, "secrets"):
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    key = st.secrets["GEMINI_API_KEY"]
                    if key and isinstance(key, str) and key.strip():
                        return key.strip()
            except Exception:
                pass
    except Exception:
        pass

    return None


def generate_engineering_explanation(
    predicted_strength,
    model_name="XGBoost",
    top_pos_contributors=None,
    top_neg_contributors=None,
    input_features=None,
    baseline_strength=None,
    timeout_seconds=12
):
    """
    Send structured prediction and SHAP data to Gemini API for human-readable explanation.
    
    Returns:
    --------
    dict: {
        "success": bool,
        "explanation": str,
        "error": str or None
    }
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return {
            "success": False,
            "explanation": "AI explanation is temporarily unavailable. (GEMINI_API_KEY not configured)",
            "error": "Missing API Key"
        }

    # Format structured context payload
    pos_str = ", ".join([f"{item['Feature']} (+{item['Impact']:.2f} MPa)" for item in (top_pos_contributors or [])])
    neg_str = ", ".join([f"{item['Feature']} ({item['Impact']:.2f} MPa)" for item in (top_neg_contributors or [])])
    
    system_instruction = (
        "You are an engineering AI explanation assistant.\n"
        "Explain the supplied machine-learning prediction for geopolymer composite compressive strength.\n"
        "Use ONLY the supplied prediction and SHAP information.\n"
        "Do not invent experimental evidence.\n"
        "Do not claim causality from SHAP.\n"
        "Distinguish model attribution from established engineering knowledge.\n"
        "Use concise, technically accurate language suitable for a civil/materials engineering audience.\n\n"
        "Explain:\n"
        "1. Predicted compressive strength\n"
        "2. Strongest positive material contributors\n"
        "3. Strongest negative material contributors\n"
        "4. Key chemical / material relationships\n"
        "5. Engineering recommendations & limitations\n\n"
        "Do not modify or recalculate the numerical prediction."
    )

    prompt_content = f"""
    Prediction Details:
    - Target: 28-day Compressive Strength
    - Predicted Value: {predicted_strength:.2f} MPa
    - Primary Model: {model_name}
    - Top Positive SHAP Drivers (+ Strength): {pos_str if pos_str else 'N/A'}
    - Top Negative SHAP Drivers (- Strength): {neg_str if neg_str else 'N/A'}
    """

    # Active Gemini Model Candidates with multi-model failover
    model_candidates = [
        'gemini-3.5-flash',
        'gemini-3.6-flash',
        'gemini-3.7-flash',
        'gemini-2.5-flash',
        'gemini-flash-latest'
    ]

    last_error = None
    
    for model_m in model_candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_m}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "parts": [{"text": prompt_content}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 450,
                "temperature": 0.2
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                
                candidates = res_data.get('candidates', [])
                if candidates:
                    text_parts = candidates[0].get('content', {}).get('parts', [])
                    if text_parts:
                        explanation_text = text_parts[0].get('text', '').strip()
                        return {
                            "success": True,
                            "explanation": explanation_text,
                            "error": None
                        }
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8') if e.fp else str(e)
            last_error = f"HTTP Error {e.code}"
            # On 404 (not found), 503 (service unavailable/overloaded), 429 (rate limit), 500/502/504, try next candidate model!
            if e.code in [404, 503, 429, 500, 502, 504]:
                continue
            else:
                break
        except Exception as e:
            last_error = str(e)
            continue

    return {
        "success": False,
        "explanation": "AI explanation is temporarily unavailable.",
        "error": last_error if last_error else "API request failed"
    }
