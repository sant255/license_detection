import os
import io
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from utils.detection import load_model, detect_plates_from_image, annotate_image
from utils.ocr import read_plate_text_from_image, clean_plate_text
from utils.db import save_record, fetch_recent, fetch_stats, get_image_bytes

# Configs from env
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "models/yolov8n.pt")
SAVE_IMAGES_DIR = os.getenv("SAVE_IMAGES_DIR", "saved_images")
os.makedirs(SAVE_IMAGES_DIR, exist_ok=True)

# Load model on startup (can pass device param if needed)
@st.experimental_singleton
def get_model():
    return load_model(YOLO_WEIGHTS)

model = get_model()

st.set_page_config(page_title="License Plate Recognition", layout="wide")
st.title("AI License Plate Recognition System 🚗🔎")

st.sidebar.header("Options")
conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
process_mode = st.sidebar.radio("Input mode", ["Image Upload", "Video Upload", "Webcam Stream"])
st.sidebar.markdown("---")
if st.sidebar.button("Refresh Analytics"):
    pass

st.sidebar.markdown("**Database & Storage**")
st.sidebar.write("MongoDB required; set `MONGO_URI` in .env")

# Main UI
col1, col2 = st.columns([2,1])

with col1:
    st.header("Input & Detection")
    if process_mode == "Image Upload":
        uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Cannot read image.")
            else:
                with st.spinner("Running detection..."):
                    dets = detect_plates_from_image(img, model=model, conf=conf_thres)
                    annotated = annotate_image(img, dets)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.write(f"Detected {len(dets)} candidate(s).")
                    for i, d in enumerate(dets):
                        st.subheader(f"Candidate {i+1}")
                        crop = d['crop']
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Cropped plate")
                        text = read_plate_text_from_image(crop)
                        st.write("OCR:", text if text else "No text detected")
                        # Save record to DB
                        # Save image bytes
                        try:
                            filename = f"{int(time.time())}_{i}.jpg"
                            out_path = os.path.join(SAVE_IMAGES_DIR, filename)
                            cv2.imwrite(out_path, crop)
                            with open(out_path, "rb") as f:
                                img_bytes = f.read()
                            ts = datetime.utcnow()
                            save_record(text, ts, image_bytes=img_bytes, image_filename=filename, meta={"source":"image_upload"})
                            st.success("Saved to database.")
                        except Exception as e:
                            st.error(f"DB save error: {e}")

    elif process_mode == "Video Upload":
        uploaded = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])
        if uploaded is not None:
            tfile = uploaded.name
            bytes_data = uploaded.read()
            tmp_path = os.path.join(SAVE_IMAGES_DIR, f"video_{int(time.time())}_{tfile}")
            with open(tmp_path, "wb") as f:
                f.write(bytes_data)
            # Process video frames (sample every N frames)
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            sample_rate = int(max(1, fps // 2))  # sample at half fps (tuneable)
            frame_idx = 0
            frames_with_detections = 0
            stframe = st.empty()
            progress = st.progress(0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % sample_rate != 0:
                    continue
                dets = detect_plates_from_image(frame, model=model, conf=conf_thres)
                if dets:
                    frames_with_detections += 1
                    annotated = annotate_image(frame, dets)
                    stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                    for i, d in enumerate(dets):
                        crop = d['crop']
                        text = read_plate_text_from_image(crop)
                        # save crop and record
                        filename = f"video_{int(time.time())}_{frame_idx}_{i}.jpg"
                        out_path = os.path.join(SAVE_IMAGES_DIR, filename)
                        cv2.imwrite(out_path, crop)
                        with open(out_path, "rb") as f:
                            img_bytes = f.read()
                        ts = datetime.utcnow()
                        save_record(text, ts, image_bytes=img_bytes, image_filename=filename, meta={"source":"video_upload", "frame": frame_idx})
                progress.progress(min(100, int(frame_idx / total * 100)))
            cap.release()
            st.success(f"Processing complete. {frames_with_detections} frames had detections.")

    else:  # Webcam
        st.info("Use your webcam (Streamlit camera). Note: server-hosted apps may not have a webcam.")
        cam_img = st.camera_input("Capture photo from webcam")
        if cam_img:
            file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            dets = detect_plates_from_image(img, model=model, conf=conf_thres)
            annotated = annotate_image(img, dets)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
            for i, d in enumerate(dets):
                crop = d['crop']
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Cropped plate")
                text = read_plate_text_from_image(crop)
                st.write("OCR:", text if text else "No text detected")
                # save to db
                filename = f"webcam_{int(time.time())}_{i}.jpg"
                out_path = os.path.join(SAVE_IMAGES_DIR, filename)
                cv2.imwrite(out_path, crop)
                with open(out_path, "rb") as f:
                    img_bytes = f.read()
                ts = datetime.utcnow()
                save_record(text, ts, image_bytes=img_bytes, image_filename=filename, meta={"source":"webcam"})

with col2:
    st.header("Analytics & Search")
    try:
        stats = fetch_stats()
        top = stats.get("top", [])
        if top:
            df_top = pd.DataFrame(top)
            df_top.columns = ["plate_number", "count"] if "_id" not in df_top.columns else ["_id", "count"]
            # adjust column names
            if "_id" in df_top.columns:
                df_top = df_top.rename(columns={"_id":"plate_number"})
            st.subheader("Most frequent plates")
            fig = px.bar(df_top.head(20), x='plate_number', y='count', labels={'plate_number':'Plate','count':'Count'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No records yet. Process some images or upload data.")

        daily = stats.get("daily", [])
        if daily:
            df_daily = pd.DataFrame(daily)
            df_daily.columns = ["date","count"] if "_id" not in df_daily.columns else ["_id","count"]
            if "_id" in df_daily.columns:
                df_daily = df_daily.rename(columns={"_id":"date"})
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            st.subheader("Daily counts")
            fig2 = px.line(df_daily.sort_values("date"), x='date', y='count', markers=True)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Search / Recent")
        query = st.text_input("Search plate (partial allowed)")
        recent = fetch_recent(200)
        if recent:
            df = pd.DataFrame(recent)
            # convert timestamp to readable
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            if query:
                q = query.strip().upper()
                df_search = df[df['plate_number'].fillna("").str.contains(q, na=False)]
                st.dataframe(df_search[['plate_number','timestamp','image_filename','meta']].sort_values('timestamp', ascending=False))
            else:
                st.dataframe(df[['plate_number','timestamp','image_filename','meta']].sort_values('timestamp', ascending=False).head(50))
        else:
            st.info("No recent records found.")

    except Exception as e:
        st.error(f"Analytics error: {e}")

st.markdown("---")
st.caption("© LPR System — Not for legal enforcement. Use for demonstration and controlled deployments only.")