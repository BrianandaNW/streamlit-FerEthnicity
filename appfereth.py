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
from sklearn.preprocessing import StandardScaler
# 🌟 BARU: Import komponen Streamlit-WebRTC
from streamlit_webrtc import webrtc_stream, VideoProcessorBase, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av # Library untuk memproses frame video

# --- KONFIGURASI DAN KONSTANTA (TIDAK BERUBAH) ---
EMOTION_LABELS = {0: 'fear', 1: 'surprised', 2: 'angry', 3: 'sad', 4: 'disgusted', 5: 'happy'}
ETHNICITY_LABELS = {0: 'Ambon (A)', 1: 'Toraja (T)', 2: 'Kaukasia (K)', 3: 'Jepang (J)'}

EMOTION_MODEL_FILE = "BestSvmEmotionModel_LandTexNoSmote.joblib"
ETHNICITY_MODEL_FILE = "BestSvmEthnicityModel_LandTexNoSmote.joblib"

CNN_INPUT_SIZE = (160, 160)
CNN_POOLING = 'avg'
CNN_LAYER_TRAINABLE = False
FACE_CROP_PAD = 0.2

# --- FUNGSI UTILITY EKSTRAKSI FITUR (TIDAK BERUBAH) ---
# LandmarkExtractor, crop_face_from_raw_landmarks, CNNEmbedder, 
# extract_features_basic, extract_features_symmetry_ratio, 
# extract_features_angles_areas, extract_class_specific_features,
# build_feature_vector, softmax, load_and_init_components (DIPINDAH KE DALAM KELAS)
# Karena kode di atas panjang dan tidak ada perubahan signifikan pada logika,
# saya akan menyalinnya (kecuali @st.cache_data pada CNNEmbedder.compute yang perlu disesuaikan).

# Replikasi LandmarkExtractor
class LandmarkExtractor:
    def __init__(self):
        mp_face = mp.solutions.face_mesh
        self.face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        
    def process_raw(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(img_rgb)

    def extract_both(self, img_bgr):
        # MediaPipe untuk streaming lebih baik dipanggil di luar init jika perlu
        # Kita panggil di dalam transformer frame
        pass # Logika ekstraksi dipindahkan ke VideoTransformer

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
        return None, (0, 0, 0, 0)
    return crop, (x1, y1, x2, y2) # Mengembalikan bounding box

# Replikasi CNNEmbedder
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

    # Hapus @st.cache_data karena dipanggil di dalam loop real-time
    def compute(self, img_bgr): 
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.input_size, interpolation=cv2.INTER_AREA)
        arr = img_to_array(img_resized)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        # Gunakan tf.function untuk memuat model dan melakukan inferensi
        @tf.function(jit_compile=True)
        def predict_fn(data):
            return self.model(data)
            
        emb = predict_fn(arr)
        return emb.numpy().flatten()
        
# Replikasi: Feature Engineering Functions (Tetap sama)
def angle_between(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.arccos(cosang)

def triangle_area(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
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
    return np.concatenate(parts).astype(np.float64)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# --- FUNGSI UTAMA PEMUATAN MODEL (DIPINDAHKAN KE DALAM KELAS BARU) ---
@st.cache_resource
def load_and_init_components():
    """Memuat model dan inisialisasi ekstraktror/embedder hanya sekali."""
    try:
        # 1. Muat Model Joblib (Pipeline: StandardScaler + SVC)
        emotion_model = joblib.load(EMOTION_MODEL_FILE)
        ethnicity_model = joblib.load(ETHNICITY_MODEL_FILE)
        
        # 2. Inisialisasi Ekstraktor/Embedder
        # LandmarkExtractor akan diinisialisasi di dalam VideoTransformer
        embedder = CNNEmbedder()

        st.success("✅ Model dan Komponen berhasil dimuat.")
        return emotion_model, ethnicity_model, embedder
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}. Pastikan file `.joblib` ada di direktori yang sama di GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal memuat komponen. Error: {e}")
        st.stop()

# Muat komponen utama yang digunakan bersama
emotion_model, ethnicity_model, embedder = load_and_init_components()

# --- 🌟 BARU: KELAS VIDEO TRANSFORMER UNTUK REAL-TIME ---
class FaceClassifierTransformer(VideoTransformerBase):
    def __init__(self):
        # Inisialisasi MediaPipe di sini agar hanya dilakukan sekali per sesi transformer
        self.mp_face = mp.solutions.face_mesh
        # Gunakan static_image_mode=False untuk streaming
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.emotion_model = emotion_model
        self.ethnicity_model = ethnicity_model
        self.embedder = embedder

    def transform(self, frame: av.VideoFrame):
        # Konversi AVFrame ke numpy array (BGR)
        img_bgr = frame.to_ndarray(format="bgr24")
        h, w = img_bgr.shape[:2]
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            lm_raw = np.array([[p.x, p.y] for p in results.multi_face_landmarks[0].landmark])
            
            # Normalisasi Landmark
            lm_norm = lm_raw.copy()
            lm_norm = lm_norm - lm_norm.mean(axis=0)
            lm_norm = lm_norm / (lm_norm.std(axis=0) + 1e-6)
            
            # Cropping Wajah untuk CNN Embedding
            crop, bbox = crop_face_from_raw_landmarks(img_bgr, lm_raw, pad=FACE_CROP_PAD)
            x1, y1, x2, y2 = bbox
            
            if crop is not None:
                try:
                    # 1. CNN Embedding
                    cnn_emb = self.embedder.compute(crop)
                    
                    # 2. Vektor Fitur
                    X_single = build_feature_vector(lm_norm, cnn_emb).reshape(1, -1)
                    
                    # 3. Prediksi Emosi
                    e_scores = self.emotion_model.decision_function(X_single)
                    e_conf = softmax(e_scores)
                    e_pred_idx = np.argmax(e_conf)
                    
                    # 4. Prediksi Etnisitas
                    eth_scores = self.ethnicity_model.decision_function(X_single)
                    eth_conf = softmax(eth_scores)
                    eth_pred_idx = np.argmax(eth_conf)
                    
                    # Format Hasil
                    emotion_label = EMOTION_LABELS[e_pred_idx].upper()
                    ethnicity_label = ETHNICITY_LABELS[eth_pred_idx].upper()
                    
                    # --- DRAWING BOUNDING BOX DAN LABEL ---
                    color = (0, 255, 0) # Hijau
                    thickness = 2
                    
                    # Gambar Bounding Box
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

                    # Teks Emosi (di atas)
                    text_emotion = f"Emosi: {emotion_label} ({e_conf[0][e_pred_idx]*100:.1f}%)"
                    cv2.putText(img_bgr, text_emotion, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    
                    # Teks Etnisitas (di bawah emosi)
                    text_ethnicity = f"Etnis: {ethnicity_label} ({eth_conf[0][eth_pred_idx]*100:.1f}%)"
                    cv2.putText(img_bgr, text_ethnicity, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                except Exception as e:
                    # Menangani error prediksi/cropping yang jarang terjadi
                    cv2.putText(img_bgr, "ERROR: " + str(e), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Jika landmark terdeteksi, tetapi cropping gagal atau error lain, biarkan frame berjalan
            pass

        # Kembalikan frame yang telah dimodifikasi (BGR)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# --- TAMPILAN UTAMA STREAMLIT ---

st.set_page_config(
    page_title="Facial Landmark And Texture Embedding Based Emotion & Ethnicity Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Logo dan Judul
# ... (Bagian logo dan judul tetap sama, saya tidak menyalinnya untuk brevity) ...
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image(
        "https://www.sandia.gov/app/uploads/sites/177/2022/04/MLDL_logo_2.jpg",
        width=100
    )
with col_title:
    st.markdown(
        "<h1 style='color: #2F4F4F; font-size: 2.5rem; margin-top: 0px;'>Emotion & Ethnicity Classifier </h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "Geometric Feature Landmark Based And Deep Feature Embedding - Emotion And Etnicity Classifier."
    )

st.divider()

# 2. Input Gambar (Upload dan Webcam)
st.header("Input Gambar Wajah")

# Hapus tab_upload dan tab_webcam (st.camera_input)
# Ganti dengan Streamlit-WebRTC untuk Real-Time

# 🌟 MODIFIKASI: Menggunakan Streamlit-WebRTC
st.subheader("📸 Analisis Wajah Secara Real-Time")
st.info("Pastikan kamera Anda aktif. Hasil klasifikasi (Emosi & Etnis) akan ditampilkan langsung pada *bounding box*.")

# Konfigurasi WebRTC
# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_stream(
    key="realtime_detection",
    mode=WebRtcMode.SENDRECV,
    # rtc_configuration=RTC_CONFIGURATION, # Opsional, untuk mengatasi masalah jaringan
    video_processor_factory=FaceClassifierTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


st.divider()

# Bagian lama untuk Upload File (Dihapus karena fokus Real-Time, tetapi jika mau dipertahankan,
# Anda bisa memindahkannya kembali di bawah WebRTC, tetapi logika 'if input_image' harus dihapus/diubah).

st.caption("Pastikan file model `.joblib` tersedia di root repositori untuk Streamlit Share. Dibuat dengan Streamlit & Gemini.")
