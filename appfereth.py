import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import joblib
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import StandardScaler # Diperlukan untuk Pipeline

# --- KONFIGURASI DAN KONSTANTA (REPLIKASI DARI SCRIPT TRAINING) ---
EMOTION_LABELS = {0: 'fear', 1: 'surprised', 2: 'angry', 3: 'sad', 4: 'disgusted', 5: 'happy'}
ETHNICITY_LABELS = {0: 'Ambon (A)', 1: 'Toraja (T)', 2: 'Kaukasia (K)', 3: 'Jepang (J)'}

EMOTION_MODEL_FILE = "BestSvmEmotionModel_LandTexNoSmote.joblib"
ETHNICITY_MODEL_FILE = "BestSvmEthnicityModel_LandTexNoSmote.joblib"

CNN_INPUT_SIZE = (160, 160)
CNN_POOLING = 'avg'
CNN_LAYER_TRAINABLE = False
FACE_CROP_PAD = 0.2

# --- FUNGSI UTILITY EKSTRAKSI FITUR (REPLIKASI LENGKAP) ---

# Replikasi: LandmarkExtractor
class LandmarkExtractor:
    def __init__(self):
        mp_face = mp.solutions.face_mesh
        # Perhatikan: Refine_landmarks=True
        self.face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    def process_raw(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(img_rgb)

    def extract_both(self, img_bgr):
        result = self.process_raw(img_bgr)
        if not result.multi_face_landmarks:
            return None, None
        lm_raw = np.array([[p.x, p.y, p.z] for p in result.multi_face_landmarks[0].landmark])
        lm_norm = lm_raw.copy()
        # Normalisasi: dikurangi mean, dibagi std (seperti dalam skrip training)
        lm_norm = lm_norm - lm_norm.mean(axis=0)
        lm_norm = lm_norm / (lm_norm.std(axis=0) + 1e-6)
        return lm_norm, lm_raw

# Replikasi: crop_face_from_raw_landmarks
def crop_face_from_raw_landmarks(img_bgr, lm_raw, pad=FACE_CROP_PAD):
    h, w = img_bgr.shape[:2]
    xs = (lm_raw[:,0] * w).astype(np.float32)
    ys = (lm_raw[:,1] * h).astype(np.float32)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_pad = pad * (x_max - x_min)
    y_pad = pad * (y_max - y_min)
    x1 = max(0, int(x_min - x_pad))
    y1 = max(0, int(y_min - y_pad))
    x2 = min(w, int(x_max + x_pad))
    y2 = min(h, int(y_max + y_pad))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop

# Replikasi: CNNEmbedder
class CNNEmbedder:
    def __init__(self, input_size=CNN_INPUT_SIZE, pooling=CNN_POOLING, trainable=CNN_LAYER_TRAINABLE):
        self.input_size = input_size
        base = MobileNetV2(
            include_top=False, 
            weights='imagenet', 
            input_shape=(input_size[0], input_size[1], 3), 
            pooling=pooling
        )
        base.trainable = trainable
        self.model = base

    @st.cache_data
    def compute(_self, img_bgr): # Menggunakan _self untuk menghindari konflik cache
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, _self.input_size, interpolation=cv2.INTER_AREA)
        arr = img_to_array(img_resized)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        # Gunakan tf.function untuk memuat model dan melakukan inferensi
        @tf.function(jit_compile=True)
        def predict_fn(data):
            return _self.model(data)
            
        emb = predict_fn(arr)
        return emb.numpy().flatten()

# Replikasi: Feature Engineering Functions
def angle_between(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.arccos(cosang)

def triangle_area(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    # Asumsi 2D, hanya menggunakan komponen x dan y (z diabaikan untuk area 2D)
    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return area

def extract_features_basic(lm):
    def dist(a, b): return np.linalg.norm(lm[a] - lm[b])
    feats = [
        dist(33, 133), dist(362, 263), dist(61, 291), dist(13, 14),
        dist(159, 145), dist(386, 374), dist(10, 152),
    ]
    ear_left = (dist(159, 145) + dist(160, 144)) / (2.0 * dist(33, 133) + 1e-6)
    ear_right = (dist(386, 374) + dist(387, 373)) / (2.0 * dist(362, 263) + 1e-6)
    mar = dist(13, 14) / (dist(61, 291) + 1e-6)
    brow_left = dist(70, 105)
    brow_right = dist(336, 334)
    feats.extend([ear_left, ear_right, mar, brow_left, brow_right])
    return np.array(feats)

def extract_features_symmetry_ratio(lm):
    def dist(a, b): return np.linalg.norm(lm[a] - lm[b])
    ear_left = (dist(159, 145) + dist(160, 144)) / (2.0 * dist(33, 133) + 1e-6)
    ear_right = (dist(386, 374) + dist(387, 373)) / (2.0 * dist(362, 263) + 1e-6)
    mar = dist(13, 14) / (dist(61, 291) + 1e-6)
    face_len = dist(10, 152) + 1e-6
    brow_left = dist(70, 105)
    brow_right = dist(336, 334)
    left_mask = lm[:,0] < 0
    right_mask = lm[:,0] >= 0
    sym_x = abs(np.mean(np.abs(lm[left_mask,0])) - np.mean(np.abs(lm[right_mask,0]))) if left_mask.any() and right_mask.any() else 0.0
    sym_y = abs(np.mean(lm[left_mask,1]) - np.mean(lm[right_mask,1])) if left_mask.any() and right_mask.any() else 0.0
    ear_sym = ear_left / (ear_right + 1e-6)
    mar_norm = mar / face_len
    brow_asym = abs(brow_left - brow_right)
    avg_ear = (ear_left + ear_right) / 2.0
    mar_over_ear = mar / (avg_ear + 1e-6)
    mar_over_face = mar / (face_len + 1e-6)
    ear_diff = abs(ear_left - ear_right)
    return np.array([
        ear_sym, mar_norm, brow_asym, avg_ear, mar,
        sym_x, sym_y, mar_over_ear, mar_over_face, ear_diff
    ])

def extract_features_angles_areas(lm):
    idx = {
        'mouth_l': 61, 'mouth_r': 291, 'lip_up': 13,
        'nose': 1, 'eye_l_o': 33, 'eye_l_i': 133, 'eye_r_o': 362, 'eye_r_i': 263
    }
    p = {k: lm[v] for k, v in idx.items()}
    ang_mouth_nose_l = angle_between(p['mouth_l'], p['nose'], p['lip_up'])
    ang_mouth_nose_r = angle_between(p['mouth_r'], p['nose'], p['lip_up'])
    ang_eye_left = angle_between(p['eye_l_o'], p['eye_l_i'], p['mouth_l'])
    ang_eye_right = angle_between(p['eye_r_o'], p['eye_r_i'], p['mouth_r'])
    area_eye_left = triangle_area(p['eye_l_o'], p['eye_l_i'], p['nose'])
    area_eye_right = triangle_area(p['eye_r_o'], p['eye_r_i'], p['nose'])
    area_mouth = triangle_area(p['mouth_l'], p['mouth_r'], p['lip_up'])
    return np.array([
        ang_mouth_nose_l, ang_mouth_nose_r,
        ang_eye_left, ang_eye_right,
        area_eye_left, area_eye_right, area_mouth
    ])

def extract_class_specific_features(lm):
    def dist(a, b): return np.linalg.norm(lm[a] - lm[b])
    face_len = dist(10, 152) + 1e-6
    mouth_left_nose = dist(61, 1) / face_len
    mouth_right_nose = dist(291, 1) / face_len
    mouth_corner_asym = abs(mouth_left_nose - mouth_right_nose)
    eye_left_center = (lm[33] + lm[133]) / 2.0
    eye_right_center = (lm[362] + lm[263]) / 2.0
    brow_left_center = (lm[70] + lm[105]) / 2.0
    brow_right_center = (lm[336] + lm[334]) / 2.0
    brow_lift_left = np.linalg.norm(brow_left_center - eye_left_center) / face_len
    brow_lift_right = np.linalg.norm(brow_right_center - eye_right_center) / face_len
    brow_lift_asym = abs(brow_lift_left - brow_lift_right)
    lip_up = 13
    lip_low = 14
    mouth_open_ratio = dist(lip_up, lip_low) / (dist(61, 291) + 1e-6)
    return np.array([
        mouth_left_nose, mouth_right_nose, mouth_corner_asym,
        brow_lift_left, brow_lift_right, brow_lift_asym,
        mouth_open_ratio
    ])

def build_feature_vector(lm, cnn_emb=None):
    parts = [
        lm.flatten(),
        extract_features_basic(lm),
        extract_features_symmetry_ratio(lm),
        extract_features_angles_areas(lm),
        extract_class_specific_features(lm)
    ]
    if cnn_emb is not None:
        parts.append(cnn_emb)
    return np.concatenate(parts).astype(np.float64) # Pastikan tipe data konsisten

def softmax(x):
    """Menghitung softmax untuk mengubah skor mentah (decision function) menjadi confidence/probabilitas."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# --- FUNGSI UTAMA PEMUATAN MODEL ---

@st.cache_resource
def load_and_init_components():
    """Memuat model dan inisialisasi ekstraktror/embedder hanya sekali."""
    try:
        # 1. Muat Model Joblib (Pipeline: StandardScaler + SVC)
        # Model ini berisi Pipeline yang harus memiliki fungsi .decision_function
        emotion_model = joblib.load(EMOTION_MODEL_FILE)
        ethnicity_model = joblib.load(ETHNICITY_MODEL_FILE)
        
        # 2. Inisialisasi Ekstraktor/Embedder
        extractor = LandmarkExtractor()
        embedder = CNNEmbedder()

        st.success("✅ Model dan Komponen berhasil dimuat.")
        return emotion_model, ethnicity_model, extractor, embedder
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}. Pastikan file `.joblib` ada di direktori yang sama di GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal memuat komponen. Error: {e}")
        st.stop()

# Muat komponen utama
emotion_model, ethnicity_model, extractor, embedder = load_and_init_components()

# --- FUNGSI PREDIKSI TERPADU ---

def predict_image(image_pil):
    """Menjalankan seluruh pipeline prediksi untuk kedua model."""
    
    # 1. Konversi PIL ke BGR (untuk OpenCV dan MediaPipe)
    image_np = np.array(image_pil)
    # Gunakan cv2.COLOR_RGB2BGR karena PIL membaca sebagai RGB, tetapi cv2.imread secara default BGR
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 2. Ekstraksi Landmark & Cropping
    lm_norm, lm_raw = extractor.extract_both(img_bgr)
    
    if lm_norm is None:
        return None, None, "Landmark wajah tidak terdeteksi. Pastikan wajah terlihat jelas.", None

    crop = crop_face_from_raw_landmarks(img_bgr, lm_raw, pad=FACE_CROP_PAD)
    if crop is None or crop.size == 0:
        return None, None, "Gagal memotong wajah (crop area kosong).", None
    
    # 3. CNN Embedding
    cnn_emb = embedder.compute(crop)
    
    # 4. Membangun Vektor Fitur
    X_single = build_feature_vector(lm_norm, cnn_emb).reshape(1, -1)
    
    # 5. Prediksi dan Confidence (Menggunakan decision_function dan Softmax)
    
    # MODEL EMOSI
    emotion_scores = emotion_model.decision_function(X_single)
    emotion_confidence = softmax(emotion_scores)
    emotion_pred_idx = np.argmax(emotion_confidence)
    
    # MODEL ETNISITAS
    ethnicity_scores = ethnicity_model.decision_function(X_single)
    ethnicity_confidence = softmax(ethnicity_scores)
    ethnicity_pred_idx = np.argmax(ethnicity_confidence)

    # 6. Format Hasil
    emotion_result = {
        "label": EMOTION_LABELS[emotion_pred_idx],
        "confidence": emotion_confidence[0][emotion_pred_idx]
    }
    ethnicity_result = {
        "label": ETHNICITY_LABELS[ethnicity_pred_idx],
        "confidence": ethnicity_confidence[0][ethnicity_pred_idx]
    }

    return emotion_result, ethnicity_result, None, lm_raw


# --- TAMPILAN UTAMA STREAMLIT ---

st.set_page_config(
    page_title="Pure Landmark-Based Emotion & Ethnicity Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Logo dan Judul
col_logo, col_title = st.columns([1, 4])

with col_logo:
    # Ganti URL ini dengan URL logo proyek Anda atau gunakan gambar lokal
    st.image(
        "https://placehold.co/100x100/A0E7E5/1F2937?text=ML+Logo",
        width=100
    )

with col_title:
    st.markdown(
        "<h1 style='color: #2F4F4F; font-size: 2.5rem;'>Emotion & Ethnicity Classifier v8</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "Aplikasi prediksi ganda berbasis gabungan Landmark, Fitur Lanjutan, dan CNN Embedding (SVC Model)."
    )

st.divider()

# 2. Input Gambar (Upload dan Webcam)
st.header("Input Gambar Wajah")

tab_upload, tab_webcam = st.tabs(["🖼️ Unggah File", "📸 Kamera Langsung"])

input_image = None
image_source = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Unggah gambar wajah (JPG/PNG)",
        type=["jpg", "jpeg", "png", "tiff"]
    )
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_source = 'upload'

with tab_webcam:
    st.info("Kamera mungkin memerlukan beberapa detik untuk memuat. Pastikan wajah Anda berada di tengah frame.")
    camera_input = st.camera_input("Ambil Foto Wajah Langsung")
    if camera_input:
        input_image = Image.open(camera_input).convert("RGB")
        image_source = 'camera'

# --- LOGIKA PREDIKSI ---

if input_image:
    st.divider()
    st.subheader("Gambar Siap Diproses")
    
    # Tampilkan gambar yang diambil/diunggah
    st.image(input_image, caption=f"Gambar dari {image_source}", use_column_width=True)

    if st.button("🚀 Jalankan Prediksi Ganda", type="primary"):
        with st.spinner('Mengekstrak fitur, menjalankan CNN embedding, dan memprediksi...'):
            try:
                emotion_res, ethnicity_res, error_msg, lm_raw = predict_image(input_image)

                if error_msg:
                    st.error(f"❌ Kesalahan Pemrosesan: {error_msg}")
                    st.warning("Coba gambar lain atau pastikan wajah terlihat penuh dan jelas.")
                else:
                    st.subheader("✅ Hasil Prediksi")

                    col1, col2 = st.columns(2)

                    # Hasil Model EMOSI
                    with col1:
                        st.markdown("<h3 style='color: #1E90FF;'>Model Emosi</h3>", unsafe_allow_html=True)
                        st.metric(
                            label="Prediksi Emosi:",
                            value=emotion_res['label'].upper(),
                            delta=f"{emotion_res['confidence']*100:.2f}% (Keyakinan)",
                            delta_color="normal"
                        )
                        
                        st.markdown(f"**Keyakinan (Confidence):** `{emotion_res['confidence']:.4f}`")
                    
                    # Hasil Model ETNISITAS
                    with col2:
                        st.markdown("<h3 style='color: #3CB371;'>Model Etnisitas</h3>", unsafe_allow_html=True)
                        st.metric(
                            label="Prediksi Etnisitas:",
                            value=ethnicity_res['label'].upper(),
                            delta=f"{ethnicity_res['confidence']*100:.2f}% (Keyakinan)",
                            delta_color="normal"
                        )
                        st.markdown(f"**Keyakinan (Confidence):** `{ethnicity_res['confidence']:.4f}`")

                    st.balloons()

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menjalankan pipeline. Detail: {e}")
                # st.exception(e) 

else:
    st.info("Silakan unggah atau ambil gambar wajah untuk memulai analisis.")

st.divider()
st.caption("Pastikan file model `.joblib` tersedia di root repositori untuk Streamlit Share. Dibuat dengan Streamlit & Gemini.")