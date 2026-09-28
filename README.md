# 🛡️ FaceGuard AI

**AI-Powered Face Mask Detection System**

FaceGuard AI is an AI-powered face mask detection application that detects human faces and classifies them as **With Mask** or **Without Mask** in images and through a live webcam.

## 🚀 Features

* 👤 Face Detection
* 😷 With Mask Detection
* 🚫 Without Mask Detection
* 🟩 Real-time Bounding Boxes
* 📊 Confidence Score
* 👥 Multiple Face Detection
* 📷 Image Detection
* 🎥 Live Webcam Detection
* 💻 Professional Streamlit Interface

## 🧠 Technologies Used

* Python
* TensorFlow
* EfficientNetB0
* OpenCV
* YuNet Face Detector
* Streamlit
* Streamlit-WebRTC
* NumPy
* Pillow

## 📊 Model Performance

The EfficientNetB0 classification model achieved approximately **98% validation accuracy** on the face-mask dataset.

## 🔍 How It Works

1. The user uploads an image or starts the webcam.
2. YuNet detects faces in the frame.
3. Each detected face is cropped and processed.
4. EfficientNetB0 classifies the face as:

   * **With Mask**
   * **Without Mask**
5. The result and confidence score are displayed with a bounding box.

## 📁 Project Structure

```text
FaceGuard-AI/
│
├── app.py
├── style.css
├── face_mask.keras
├── face_detection_yunet_2023mar.onnx
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/FaceGuard-AI.git
cd FaceGuard-AI
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🌐 Deployment

FaceGuard AI can be deployed using **Streamlit Community Cloud**.

## 👨‍💻 Developer

**Muhammad Saim**

Made with Python, TensorFlow, OpenCV & Streamlit.

## 📌 Disclaimer

This project is developed for educational and demonstration purposes. Detection results may vary depending on image quality, lighting, camera a
