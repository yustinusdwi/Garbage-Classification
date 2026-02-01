import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import base64
from io import BytesIO
from skimage.feature import graycomatrix, graycoprops

# Page setup
st.set_page_config(layout="wide")

# Load models once
@st.cache_resource
def load_models():
    xgb = joblib.load("XGBoost.pkl")
    rf = joblib.load("Random_Forest.pkl")
    gb = joblib.load("Gradient_Boosting.pkl")
    scaler = joblib.load("scaler.pkl")  
    return xgb, rf, gb, scaler

xgb_model, rf_model, gb_model, scaler = load_models()

# Background removal
def remove_background(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, img.shape[1]-10, img.shape[0]-10)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_clean = img * mask2[:, :, np.newaxis]
    return img_clean

# Color histogram feature
def color_histogram(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Shape features (Hu Moments)
def shape_features(img_gray):
    moments = cv2.moments(img_gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

# Texture features (GLCM)
def texture_features(img_gray):
    glcm = graycomatrix(img_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    return [
        graycoprops(glcm, 'contrast')[0][0],
        graycoprops(glcm, 'correlation')[0][0],
        graycoprops(glcm, 'energy')[0][0],
        graycoprops(glcm, 'homogeneity')[0][0]
    ]

# Combined feature extraction
def extract_all_features(pil_image):
    img = np.array(pil_image.convert("RGB"))  # Convert PIL to OpenCV RGB
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    img_clean = remove_background(img_cv2)
    img_gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    features = []
    features.extend(color_histogram(img_clean))
    features.extend(shape_features(img_gray))
    features.extend(texture_features(img_gray))
    return features

# Sidebar
with st.sidebar:
    st.markdown("### Masukkan gambar")
    uploaded_file = st.file_uploader(
        "Tarik dan jatuhkan file di sini atau klik untuk memilih file",  
        type=["jpg", "jpeg", "png"]
    )
    
    start_button = st.button("MULAI")

# Title
st.markdown("<h1 style='text-align: center;'>Klasifikasi Cerdas Sampah Padat</h1>", unsafe_allow_html=True)

# Base64 image converter
def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    return img_b64

# Rendering uploaded image
st.markdown("<h2 style='text-align: center;'>Gambar Input</h2>", unsafe_allow_html=True)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_b64 = get_image_base64(image)
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{img_b64}' style='max-width: 100%; height: auto;'>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("Tidak ada gambar yang diunggah. Silakan unggah file gambar.")

# Prediction section
st.markdown("---")
st.markdown("<h2 style='text-align: center;'>Prediksi</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

if start_button and uploaded_file is not None:
    features = extract_all_features(image)

    # Terapkan scaling pada fitur sebelum prediksi
    scaled_features = scaler.transform([features])  

    # Prediksi dengan model
    pred_xgb = xgb_model.predict(scaled_features)[0]
    pred_rf = rf_model.predict(scaled_features)[0]
    pred_gb = gb_model.predict(scaled_features)[0]

    # Kategori mapping
    kategori_dict = {
        0: 'metal',
        1: 'clothes',
        2: 'white-glass',
        3: 'green-glass',
        4: 'cardboard',
        5: 'brown-glass',
        6: 'trash',
        7: 'shoes',
        8: 'battery',
        9: 'paper',
        10: 'biological',
        11: 'plastic'
    }

    # Mapping hasil prediksi ke kategori teks
    kategori_xgb = kategori_dict.get(pred_xgb, 'Tidak diketahui')
    kategori_rf = kategori_dict.get(pred_rf, 'Tidak diketahui')
    kategori_gb = kategori_dict.get(pred_gb, 'Tidak diketahui')

    with col1:
        st.markdown(f"<h4 style='text-align: center;'>XGBoost: <b>{kategori_xgb}</b></h4>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h4 style='text-align: center;'>Random Forest: <b>{kategori_rf}</b></h4>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<h4 style='text-align: center;'>Gradient Boosting: <b>{kategori_gb}</b></h4>", unsafe_allow_html=True)


elif uploaded_file is not None:
    st.warning("Klik 'MULAI' untuk memulai klasifikasi.")
else:
    st.error("Tidak ada gambar yang diunggah. Silakan unggah file gambar.")
