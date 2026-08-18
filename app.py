import os
os.environ["FLAGS_enable_pir_api"] = "0"

import streamlit as st
import cv2
import re
import json
import tempfile
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

st.set_page_config(
    page_title="Indian Number Plate Recognition",
    page_icon="🚗"
)

st.title("🚗 Indian Number Plate Recognition")
st.write("Upload a vehicle image or video to detect and read number plates.")


# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():

    # Number-plate detector bundled with this project
    plate_yolo = YOLO("/content/vehicle_dataset/best.pt")

    # COCO-pretrained detector for automatic vehicle/person classes.
    # It detects: car, motorcycle, person, bus and truck.
    vehicle_yolo = YOLO("yolo11n.pt")

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False
    )

    return plate_yolo, vehicle_yolo, ocr


plate_yolo, vehicle_yolo, ocr = load_models()


# =========================
# OCR FUNCTION
# =========================

def read_plate(plate):

    if plate is None or plate.size == 0:
        return "", 0.0, None

    plate = cv2.resize(
        plate,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite("/tmp/plate.jpg", otsu)

    ocr_result = ocr.predict("/tmp/plate.jpg")

    best_text = ""
    best_conf = 0.0

    for res in ocr_result:

        data = res.json

        if isinstance(data, str):
            data = json.loads(data)

        data = data["res"]

        texts = data["rec_texts"]
        scores = data["rec_scores"]

        if len(texts) == 0:
            continue

        text = " ".join(texts)

        text = re.sub(
            r"[^A-Z0-9]",
            "",
            text.upper()
        )

        confidence = float(sum(scores) / len(scores))

        if confidence > best_conf:
            best_text = text
            best_conf = confidence

    return best_text, best_conf, otsu


# =========================
# DETECT + READ PLATE
# =========================


VEHICLE_NAMES = {
    0: "Pedestrian",
    2: "Car",
    3: "Motorcycle",
    5: "Heavy Vehicle",
    7: "Heavy Vehicle"
}

def detect_vehicles(image):
    """Detect Car, Bike, Pedestrian and Heavy Vehicle using COCO YOLO."""
    results = vehicle_yolo(
        image,
        conf=0.30,
        classes=[0, 2, 3, 5, 7],
        verbose=False
    )

    detections = []

    boxes = results[0].boxes
    if boxes is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    for box, class_id, conf in zip(xyxy, cls, confs):
        x1, y1, x2, y2 = box.astype(int)
        detections.append({
            "type": VEHICLE_NAMES.get(class_id, "Unknown"),
            "confidence": float(conf),
            "box": (x1, y1, x2, y2)
        })

    return detections


def match_plate_to_vehicle(plate_box, vehicle_detections):
    """Match a plate to the vehicle whose box contains/nearest its center."""
    x1, y1, x2, y2 = plate_box
    px = (x1 + x2) / 2
    py = (y1 + y2) / 2

    containing = []
    for v in vehicle_detections:
        vx1, vy1, vx2, vy2 = v["box"]
        if vx1 <= px <= vx2 and vy1 <= py <= vy2:
            area = max(1, (vx2-vx1) * (vy2-vy1))
            containing.append((area, v))

    if containing:
        containing.sort(key=lambda x: x[0])
        return containing[0][1]["type"]

    # Fallback: nearest vehicle center.
    best = None
    best_dist = float("inf")
    for v in vehicle_detections:
        vx1, vy1, vx2, vy2 = v["box"]
        cx = (vx1 + vx2) / 2
        cy = (vy1 + vy2) / 2
        dist = (cx - px) ** 2 + (cy - py) ** 2
        if dist < best_dist:
            best_dist = dist
            best = v

    return best["type"] if best else "Unknown"


def process_frame(image, draw_box=True):

    # Detect vehicle/person classes first.
    vehicle_detections = detect_vehicles(image)

    # Draw vehicle boxes and labels.
    if draw_box:
        for v in vehicle_detections:
            x1, y1, x2, y2 = v["box"]
            label = f'{v["type"]} ({v["confidence"] * 100:.0f}%)'

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 180, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 8, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 180, 0),
                2
            )

    # Detect number plates.
    results = plate_yolo(
        image,
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes.xyxy.cpu().numpy()
    detections = []

    for box in boxes:

        x1, y1, x2, y2 = box.astype(int)

        h, w = image.shape[:2]

        mx = int((x2 - x1) * 0.05)
        my = int((y2 - y1) * 0.05)

        x1 = max(0, x1 - mx)
        x2 = min(w, x2 + mx)
        y1 = max(0, y1 - my)
        y2 = min(h, y2 + my)

        plate = image[y1:y2, x1:x2]

        text, confidence, processed_plate = read_plate(plate)

        vehicle_type = match_plate_to_vehicle(
            (x1, y1, x2, y2),
            vehicle_detections
        )

        if draw_box:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            if text:
                cv2.putText(
                    image,
                    f"{vehicle_type}: {text}",
                    (x1, min(y2 + 25, h - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

        detections.append({
            "plate": text,
            "confidence": confidence,
            "vehicle_type": vehicle_type,
            "processed_plate": processed_plate
        })

    return image, detections, vehicle_detections


# =========================
# MODE SELECTION
# =========================

mode = st.radio(
    "Select input type:",
    ["🖼️ Image", "🎥 Video"],
    horizontal=True
)


# ==========================================================
# IMAGE MODE
# ==========================================================

if mode == "🖼️ Image":

    uploaded_file = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        file_bytes = uploaded_file.read()

        image = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("🔍 Detect Number Plate"):

            with st.spinner("Detecting and reading plate..."):

                result_image, text, confidence, plate = process_frame(
                    image.copy()
                )

            st.image(
                cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                caption="Detection Result",
                use_container_width=True
            )

            if plate is not None:
                st.image(
                    plate,
                    caption="Detected Plate",
                    clamp=True
                )

            if text:

                st.success(
                    f"🚗 Number Plate: {text}"
                )

                st.info(
                    f"📊 OCR Confidence: "
                    f"{confidence * 100:.2f}%"
                )

            else:

                st.warning(
                    "⚠️ Plate detected, but OCR "
                    "could not read the number."
                )


# ==========================================================
# VIDEO MODE
# ==========================================================

else:

    uploaded_video = st.file_uploader(
        "Upload vehicle video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:

        suffix = os.path.splitext(
            uploaded_video.name
        )[1]

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        input_file.write(uploaded_video.read())
        input_file.close()

        input_path = input_file.name

        st.video(input_path)

        if st.button("🎥 Process Video"):

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():

                st.error("❌ Could not open video.")

            else:

                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps <= 0:
                    fps = 25

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                output_path = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ).name

                # MP4 output
                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                out = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                progress = st.progress(0)
                status = st.empty()

                detected_plates = []

                frame_number = 0

                # Process every 5th frame
                frame_skip = 5

                last_plate = ""
                last_confidence = 0.0

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1

                    # Detection only every 5th frame
                    if frame_number % frame_skip == 0:

                        processed_frame, text, confidence, _ = (
                            process_frame(
                                frame.copy()
                            )
                        )

                        frame = processed_frame

                        if text:

                            last_plate = text
                            last_confidence = confidence

                            if text not in detected_plates:
                                detected_plates.append(text)

                    # Keep last result visible
                    if last_plate:

                        cv2.putText(
                            frame,
                            f"Plate: {last_plate} "
                            f"({last_confidence * 100:.1f}%)",
                            (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                    out.write(frame)

                    progress_value = (
                        frame_number /
                        max(total_frames, 1)
                    )

                    progress.progress(
                        min(progress_value, 1.0)
                    )

                    status.text(
                        f"Processing frame "
                        f"{frame_number}/{total_frames}"
                    )

                cap.release()
                out.release()

                progress.empty()
                status.empty()

                st.success(
                    "✅ Video processing complete!"
                )

                st.subheader("🎬 Processed Video")

                st.video(output_path)

                if detected_plates:

                    st.subheader(
                        "🚗 Detected Number Plates"
                    )

                    for plate_text in detected_plates:

                        st.success(
                            f"Plate: {plate_text}"
                        )

                else:

                    st.warning(
                        "⚠️ No readable number plate "
                        "was found in the video."
                    )
