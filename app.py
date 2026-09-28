import os
import cv2
import av
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="FaceGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# LOAD CSS
# ==========================================

BASE_DIR = os.path.dirname(__file__)

CSS_PATH = os.path.join(
    BASE_DIR,
    "style.css"
)

if os.path.exists(CSS_PATH):

    with open(
        CSS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


# ==========================================
# TITLE
# ==========================================

st.title("🛡️ FaceGuard AI")

st.subheader("AI-Powered Face Mask Detection")

st.write(
    "Detect faces and classify them as "
    "With Mask or Without Mask using AI."
)


# ==========================================
# FILE PATHS
# ==========================================

MASK_MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_mask.keras"
)

YUNET_MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_detection_yunet_2023mar.onnx"
)


# ==========================================
# LOAD MASK MODEL
# ==========================================

@st.cache_resource
def load_mask_model():

    if not os.path.exists(MASK_MODEL_PATH):

        raise FileNotFoundError(
            f"Mask model not found: {MASK_MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MASK_MODEL_PATH
    )

    return model


# ==========================================
# LOAD YUNET
# ==========================================

@st.cache_resource
def load_face_detector():

    if not os.path.exists(YUNET_MODEL_PATH):

        raise FileNotFoundError(
            f"YuNet model not found: {YUNET_MODEL_PATH}"
        )

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        0.45,
        0.3,
        5000
    )

    return detector


# ==========================================
# LOAD MODELS
# ==========================================

try:

    model = load_mask_model()

    yunet = load_face_detector()

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("🛡️ FaceGuard AI")

    st.write(
        "AI-Powered Face Mask Detection"
    )

    st.divider()

    st.write("### Features")

    st.write("👤 Face Detection")
    st.write("🟩 Bounding Boxes")
    st.write("😷 With Mask Detection")
    st.write("🚫 Without Mask Detection")
    st.write("📊 Confidence Score")
    st.write("👥 Multiple Face Detection")
    st.write("🎥 Live Webcam Detection")


# ==========================================
# LIVE DETECTION PROCESSOR
# ==========================================

class MaskDetectionProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (320, 320),
            0.45,
            0.3,
            5000
        )

        self.model = model


    def recv(self, frame):

        # ==================================
        # READ FRAME
        # ==================================

        img = frame.to_ndarray(
            format="bgr24"
        )

        h, w = img.shape[:2]


        # ==================================
        # FACE DETECTION
        # ==================================

        self.detector.setInputSize(
            (w, h)
        )

        _, faces = self.detector.detect(
            img
        )


        # ==================================
        # PROCESS FACES
        # ==================================

        if faces is not None:

            for i, face in enumerate(faces):

                x, y, fw, fh = face[:4]

                x = int(x)
                y = int(y)
                fw = int(fw)
                fh = int(fh)

                x2 = x + fw
                y2 = y + fh


                # ==============================
                # KEEP BOX INSIDE IMAGE
                # ==============================

                x = max(0, x)
                y = max(0, y)

                x2 = min(w, x2)
                y2 = min(h, y2)


                # ==============================
                # PADDING
                # ==============================

                pad_x = int(fw * 0.25)
                pad_y = int(fh * 0.35)

                crop_x1 = max(
                    0,
                    x - pad_x
                )

                crop_y1 = max(
                    0,
                    y - pad_y
                )

                crop_x2 = min(
                    w,
                    x2 + pad_x
                )

                crop_y2 = min(
                    h,
                    y2 + pad_y
                )


                # ==============================
                # FACE CROP
                # ==============================

                face_crop = img[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2
                ]


                if face_crop.size == 0:

                    continue


                # ==============================
                # PREPARE IMAGE
                # ==============================

                face_rgb = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB
                )

                face_resized = cv2.resize(
                    face_rgb,
                    (224, 224)
                )

                input_image = np.expand_dims(
                    face_resized,
                    axis=0
                )


                # ==============================
                # PREDICTION
                # ==============================

                prediction = self.model.predict(
                    input_image,
                    verbose=0
                )[0][0]


                # ==============================
                # CLASSIFICATION
                # ==============================

                if prediction > 0.5:

                    label = "Without Mask"
                    confidence = prediction

                else:

                    label = "With Mask"
                    confidence = 1 - prediction


                confidence_percent = (
                    confidence * 100
                )


                # ==============================
                # DRAW FACE BOX
                # ==============================

                cv2.rectangle(
                    img,
                    (x, y),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )


                # ==============================
                # LABEL
                # ==============================

                text = (
                    f"Face {i + 1} | "
                    f"{label} | "
                    f"{confidence_percent:.1f}%"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2

                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(
                    text,
                    font,
                    font_scale,
                    thickness
                )


                # ==============================
                # LABEL POSITION
                # ==============================

                label_x = x
                label_y = y - 10


                if (
                    label_y -
                    text_height -
                    baseline < 0
                ):

                    label_y = (
                        y +
                        text_height +
                        10
                    )


                # Keep label inside image

                label_x = max(
                    0,
                    min(
                        label_x,
                        w - text_width - 10
                    )
                )


                # ==============================
                # LABEL BACKGROUND
                # ==============================

                bg_x1 = label_x

                bg_y1 = (
                    label_y -
                    text_height -
                    baseline
                )

                bg_x2 = (
                    label_x +
                    text_width +
                    10
                )

                bg_y2 = label_y + 5

                bg_y1 = max(
                    0,
                    bg_y1
                )

                bg_y2 = min(
                    h,
                    bg_y2
                )


                cv2.rectangle(
                    img,
                    (
                        bg_x1,
                        bg_y1
                    ),
                    (
                        bg_x2,
                        bg_y2
                    ),
                    (0, 0, 0),
                    -1
                )


                # ==============================
                # DRAW TEXT
                # ==============================

                cv2.putText(
                    img,
                    text,
                    (
                        label_x + 5,
                        label_y
                    ),
                    font,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                    cv2.LINE_AA
                )


        # ==================================
        # RETURN FRAME
        # ==================================

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ==========================================
# DETECTION MODE
# ==========================================

mode = st.radio(
    "Choose Detection Mode",
    [
        "📷 Image Detection",
        "🎥 Live Detection"
    ],
    horizontal=True
)


# ==========================================
# IMAGE DETECTION
# ==========================================

if mode == "📷 Image Detection":

    st.header("📷 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        # ==================================
        # READ IMAGE
        # ==================================

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(
            image
        )

        frame = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        h, w = frame.shape[:2]

        result = frame.copy()


        # ==================================
        # YUNET DETECTION
        # ==================================

        yunet.setInputSize(
            (w, h)
        )

        _, faces = yunet.detect(
            frame
        )


        # ==================================
        # NO FACE
        # ==================================

        if faces is None:

            st.warning(
                "⚠️ No face detected."
            )

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )


        # ==================================
        # FACES DETECTED
        # ==================================

        else:

            face_count = len(
                faces
            )

            st.success(
                f"👥 {face_count} face(s) detected"
            )

            results = []


            # ==================================
            # PROCESS EACH FACE
            # ==================================

            for i, face in enumerate(faces):

                x, y, fw, fh = face[:4]

                x = int(x)
                y = int(y)
                fw = int(fw)
                fh = int(fh)

                x2 = x + fw
                y2 = y + fh


                # ==============================
                # KEEP BOX INSIDE IMAGE
                # ==============================

                x = max(0, x)
                y = max(0, y)

                x2 = min(w, x2)
                y2 = min(h, y2)


                # ==============================
                # PADDING
                # ==============================

                pad_x = int(fw * 0.25)
                pad_y = int(fh * 0.35)

                crop_x1 = max(
                    0,
                    x - pad_x
                )

                crop_y1 = max(
                    0,
                    y - pad_y
                )

                crop_x2 = min(
                    w,
                    x2 + pad_x
                )

                crop_y2 = min(
                    h,
                    y2 + pad_y
                )


                # ==============================
                # FACE CROP
                # ==============================

                face_crop = frame[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2
                ]


                if face_crop.size == 0:

                    continue


                # ==============================
                # PREPARE IMAGE
                # ==============================

                face_rgb = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB
                )

                face_resized = cv2.resize(
                    face_rgb,
                    (224, 224)
                )

                input_image = np.expand_dims(
                    face_resized,
                    axis=0
                )


                # ==============================
                # PREDICTION
                # ==============================

                prediction = model.predict(
                    input_image,
                    verbose=0
                )[0][0]


                # ==============================
                # CLASSIFICATION
                # ==============================

                if prediction > 0.5:

                    label = "Without Mask"
                    confidence = prediction

                else:

                    label = "With Mask"
                    confidence = 1 - prediction


                confidence_percent = (
                    confidence * 100
                )


                # ==============================
                # DRAW BOX
                # ==============================

                cv2.rectangle(
                    result,
                    (x, y),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )


                # ==============================
                # LABEL
                # ==============================

                text = (
                    f"Face {i + 1} | "
                    f"{label}: "
                    f"{confidence_percent:.1f}%"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(
                    text,
                    font,
                    font_scale,
                    thickness
                )


                label_x = x
                label_y = y - 10


                if (
                    label_y -
                    text_height -
                    baseline < 0
                ):

                    label_y = (
                        y +
                        text_height +
                        10
                    )


                label_x = max(
                    0,
                    min(
                        label_x,
                        w - text_width - 10
                    )
                )


                # ==============================
                # LABEL BACKGROUND
                # ==============================

                bg_x1 = label_x

                bg_y1 = (
                    label_y -
                    text_height -
                    baseline
                )

                bg_x2 = (
                    label_x +
                    text_width +
                    10
                )

                bg_y2 = label_y + 5

                bg_y1 = max(
                    0,
                    bg_y1
                )

                bg_y2 = min(
                    h,
                    bg_y2
                )


                cv2.rectangle(
                    result,
                    (
                        bg_x1,
                        bg_y1
                    ),
                    (
                        bg_x2,
                        bg_y2
                    ),
                    (0, 0, 0),
                    -1
                )


                # ==============================
                # TEXT
                # ==============================

                cv2.putText(
                    result,
                    text,
                    (
                        label_x + 5,
                        label_y
                    ),
                    font,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                    cv2.LINE_AA
                )


                # ==============================
                # STORE RESULT
                # ==============================

                results.append({
                    "Face": i + 1,
                    "Result": label,
                    "Confidence": confidence_percent
                })


            # ==================================
            # RESULT IMAGE
            # ==================================

            st.header(
                "🔍 Detection Result"
            )

            result_rgb = cv2.cvtColor(
                result,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                result_rgb,
                caption="FaceGuard AI Detection",
                use_container_width=True
            )


            # ==================================
            # DETAILS
            # ==================================

            st.header(
                "📊 Detection Details"
            )


            for result_item in results:

                st.write(
                    f"### 👤 Face "
                    f"{result_item['Face']}"
                )

                st.write(
                    f"**Result:** "
                    f"{result_item['Result']}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{result_item['Confidence']:.1f}%"
                )

                st.divider()


# ==========================================
# LIVE DETECTION
# ==========================================

else:

    st.header(
        "🎥 Live Face Mask Detection"
    )

    st.write(
        "Start your webcam to detect faces "
        "and classify face masks in real time."
    )

    st.info(
        "📌 Allow camera permission when your browser asks."
    )


    webrtc_streamer(
        key="faceguard-live",

        video_processor_factory=MaskDetectionProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


# ==========================================
# FOOTER
# ==========================================

st.write("---")

st.caption(
    "🛡️ FaceGuard AI | "
    "AI-Powered Face Mask Detection | "
    "Made with OpenCV & EfficientNetB0"
)