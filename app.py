import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import os
import time

# -------------------------
# App configuration & CSS
# -------------------------
st.set_page_config(page_title="Oral Cancer Predictor", page_icon="🦷", layout="centered")

# --- ENHANCED CSS FOR A BEAUTIFUL, WARMER LOOK ---
st.markdown(
    """
    <style>
    /* Main Streamlit container and background: Soft Beige/White Gradient */
    .main {
        background: linear-gradient(145deg, #fefefe, #f0f0e8); /* Soft, subtle gradient */
    }
    .stApp {max-width: 980px; margin: 0 auto;}
    
    /* Card for content blocks: Crisp white with refined shadow */
    .card {
        background: white; 
        padding: 25px; 
        border-radius: 12px; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.1); /* Stronger, professional shadow */
        border: 1px solid #e0e0e0; /* Subtle border for definition */
    }
    
    /* Muted text for descriptions */
    .muted {color:#757575; font-size:0.9rem}

    /* Primary Accent Color (Deep Teal/Ocean Blue) */
    h1, h2, h3 {
        color: #00897b; /* Deep Teal/Ocean Blue for headers */
        font-weight: 600;
    }
    
    /* Streamlit's progress bar (Customizing the accent) */
    .stProgress > div > div > div > div {
        background-color: #00897b; /* Matching Teal color */
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #d84315; /* Contrasting Terracotta/Orange for high impact metric */
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #455a64; /* Darker slate gray for label */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helper utilities
# -------------------------

# Define the expected model filename consistently
DEFAULT_MODEL_NAME = "oral_cancer_efficientnet.keras"

@st.cache_resource
def load_model(model_path=DEFAULT_MODEL_NAME):
    """Load Keras model safely with compile=False."""
    # Check for both defined name and the common fallback .h5
    if not os.path.exists(model_path):
        if os.path.exists("model.h5"):
            model_path = "model.h5"
        else:
            raise FileNotFoundError(f"Model file not found. Expected: '{DEFAULT_MODEL_NAME}' or 'model.h5'.")
            
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_path}': {e}")


def preprocess_image(pil_img, target_size=(224, 224)):
    """Resizes, converts to RGB, and normalizes the image for the model."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    
    img = ImageOps.fit(pil_img, target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, preproc_img):
    """Runs prediction and maps probability to a human-readable label."""
    preds = model.predict(preproc_img)
    
    if preds.shape[-1] == 1:
        prob = float(preds[0][0])
    else:
        prob = float(preds[0][-1])

    if prob >= 0.70:
        label = "HIGH RISK"
    elif prob >= 0.40:
        label = "MODERATE RISK"
    else:
        label = "LOW RISK"

    return prob, label

# -------------------------
# Main App Layout
# -------------------------

# --- HEADER ---
with st.container():
    st.markdown("<div class='card' style='padding: 20px; text-align: center;'>", unsafe_allow_html=True)
    st.title("🦷 Oral Cancer Risk Predictor")
    st.markdown("<p class='muted'>Analyze intra-oral images and incorporate patient risk factors for a combined risk assessment.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Try loading model once at the top
model = None
model_load_error = None
try:
    model = load_model()
except Exception as e:
    model_load_error = str(e)

# --- LAYOUT ---
col_left, col_right = st.columns([2.2, 1])

# --- COLUMN LEFT: IMAGE UPLOAD & QUESTIONNAIRE ---
with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("1) Image & Clinical Risk Factors")
    
    uploaded_file = st.file_uploader("Upload an intra-oral image (jpg / png)", type=["png", "jpg", "jpeg"])
    st.caption("Tip: Take a clear close-up under good lighting. Avoid heavy shadows or filters.")

    st.markdown("---")

    # --- QUESTIONNAIRE ---
    st.subheader("2) Risk Questionnaire")
    
    # Define required placeholder
    REQUIRED_PLACEHOLDER = "--- Select an option ---"

    # Encapsulate all inputs in a single form for easier validation and submission handling
    with st.form(key='risk_form'):
        
        # Demographics
        with st.expander("Patient demographics", expanded=False):
            age = st.number_input("Age", min_value=1, max_value=120, value=45, help="Patient age in years")
            gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"]) 

        # Tobacco & smoking
        with st.expander("🚬 Tobacco / Smoking history", expanded=True):
            smoking_status = st.selectbox("Current smoking status", ["No", "Yes — regularly (daily)", "Yes — occasionally (social)", "Former smoker (quit)"])
            if smoking_status.startswith("Yes"):
                smoke_years = st.number_input("If yes, for how many years?", min_value=0, max_value=100, value=5)
                tobacco_type = st.multiselect("Type(s) of tobacco used", ["Cigarettes", "Smokeless tobacco (Gutka, Khaini)", "E-cigarette / vaping", "Other"], default=[]) 
            else:
                smoke_years = 0
                tobacco_type = []

        # Alcohol
        with st.expander("Alcohol & Substance use (required)", expanded=True):
            # Added REQUIRED_PLACEHOLDER for validation
            alcohol = st.selectbox("Alcohol consumption frequency *", [REQUIRED_PLACEHOLDER, "Never", "Occasional (monthly)", "Weekly", "Daily / Heavy"])
            other_substances = st.text_input("Other substance use (optional)")

        # Oral disease / symptoms history
        with st.expander("⚕️ Oral health & prior conditions", expanded=True):
            # Added REQUIRED_PLACEHOLDER for validation
            prior_lesions = st.selectbox("Previous oral lesions/diagnosed disease? *", [REQUIRED_PLACEHOLDER, "No", "Yes — non-cancerous lesions", "Yes — precancerous lesion / dysplasia", "Yes — previous oral cancer"])
            
            lesion_when = None
            prior_treatment = None
            if prior_lesions in ["Yes — non-cancerous lesions", "Yes — precancerous lesion / dysplasia", "Yes — previous oral cancer"]:
                lesion_when = st.text_input("If yes, when was it diagnosed/noticed? *")
                prior_treatment = st.text_input("Was treatment given? (e.g., surgery, biopsy) *")

            # Added selection check for current symptoms
            current_symptoms = st.multiselect("Current symptoms (select all that apply) *", ["Non-healing ulcer", "Red/white patch", "Pain or discomfort", "Difficulty swallowing or speaking", "Bleeding", "Lump or thickening", "Loose teeth or numbness", "No symptoms"], default=[])
            # Added REQUIRED_PLACEHOLDER for validation
            symptom_duration = st.selectbox("Duration of symptoms *", [REQUIRED_PLACEHOLDER, "Less than 2 weeks", "2–6 weeks", "6–12 weeks", "More than 3 months"])
        
        # Family history
        with st.expander("🧬 Family history & other risks", expanded=True):
            family_history = st.selectbox("Family history of head & neck cancer?", ["No", "Yes — first degree relative", "Unknown"]) 
            # Added REQUIRED_PLACEHOLDER for validation
            hpv_history = st.selectbox("Known HPV infection history? *", [REQUIRED_PLACEHOLDER, "No", "Yes", "Unknown"])

        # Predict button - must be inside the st.form block
        predict_button = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True) # Close the card

# --- COLUMN RIGHT: RESULTS & GUIDANCE ---
with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction & Clinical Guidance 📝")

    if model_load_error:
        st.error(f"⚠️ Model load failed: {model_load_error}")
        st.info(f"Please check that '{DEFAULT_MODEL_NAME}' or 'model.h5' is in the application folder.")
    elif uploaded_file is None:
        st.info("Upload image and fill questionnaire to begin.")
    
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        # Fixed Deprecation: use_column_width is replaced by use_container_width
        st.image(image, caption="Uploaded Image", use_container_width=True) 
        
        if predict_button:
            
            # --- INPUT VALIDATION CHECK ---
            if alcohol == REQUIRED_PLACEHOLDER or prior_lesions == REQUIRED_PLACEHOLDER or symptom_duration == REQUIRED_PLACEHOLDER or hpv_history == REQUIRED_PLACEHOLDER or not current_symptoms:
                st.error("Please fill in all required fields marked with * in the questionnaire.")
            else:
                # Add a slight delay for better UX
                with st.spinner('Analyzing image and calculating risk...'):
                    time.sleep(1) 
                
                try:
                    pre = preprocess_image(image)
                    prob, label = predict(model, pre)

                    # --- 1. MODEL-ONLY OUTPUT ---
                    pct = prob * 100
                    st.markdown("---")
                    st.markdown("**1. Image Model Probability (CNN)**")
                    st.metric(label=f"Predicted Class: **{label}**", value=f"{pct:.2f}%")
                    st.progress(int(pct))

                    # --- 2. CLINICAL RISK ADJUSTMENT ---
                    bonus_risk = 0.0
                    # Check for critical risk factors
                    if "Smokeless" in " ".join(tobacco_type): bonus_risk += 0.08
                    if prior_lesions in ["Yes — precancerous lesion / dysplasia", "Yes — previous oral cancer"]: bonus_risk += 0.15
                    if alcohol == "Daily / Heavy": bonus_risk += 0.05
                    if family_history.startswith("Yes"): bonus_risk += 0.04
                    if "Non-healing ulcer" in current_symptoms and symptom_duration not in ["Less than 2 weeks", REQUIRED_PLACEHOLDER]: bonus_risk += 0.08
                    
                    combined_prob = min(0.999, prob + bonus_risk)
                    combined_pct = combined_prob * 100

                    st.markdown("---")
                    st.markdown("**2. Combined Estimated Risk**")
                    st.write(f"Adjustment from Risk Factors: **+{bonus_risk*100:.2f}%**")
                    st.write(f"**Final Estimated Risk:** **{combined_pct:.2f}%**")

                    # --- 3. FINAL GUIDANCE ---
                    st.markdown("---")
                    st.markdown("**3. Recommended Action**")
                    
                    if combined_prob >= 0.70:
                        st.error("🚨 URGENT: High risk estimated. Immediate referral to an **Oral Pathologist/Specialist** is strongly recommended for definitive diagnosis and **biopsy**.")
                    elif combined_prob >= 0.40:
                        st.warning("⚠️ MONITOR: Moderate risk. Referral for specialist consultation and close follow-up (2-4 weeks) is recommended. Biopsy consideration.")
                    else:
                        st.success("✅ ROUTINE: Low risk estimated. Recommend routine monitoring and lifestyle modification advice.")

                    st.caption("Disclaimer: This tool is for educational/screening purposes only and is not a substitute for clinical assessment.")

                except Exception as e:
                    st.error(f"An unexpected error occurred during prediction: {e}")

    st.markdown("</div>", unsafe_allow_html=True)