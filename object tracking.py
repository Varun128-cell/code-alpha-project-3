import streamlit as st
import tempfile
import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8m.pt")

def apply_thermal(frame):
    thermal = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    return thermal

def process_video(input_path, output_path, mode="normal", frame_skip=1):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))

    tracker = DeepSort(max_age=30)
    frame_idx = 0

    progress = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        results = model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if label != "person" or conf < 0.3:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = track.to_ltrb()

            x1, y1 = max(0, int(l)), max(0, int(t))
            x2, y2 = min(int(l + w), frame.shape[1] - 1), min(int(t + h), frame.shape[0] - 1)

            if mode == "thermal":
                cropped_person = frame[y1:y2, x1:x2]
                if cropped_person.size > 0:
                    frame[y1:y2, x1:x2] = apply_thermal(cropped_person)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            elif mode == "blur":
                cropped_person = frame[y1:y2, x1:x2]
                if cropped_person.size > 0:
                    sub_face = cv2.GaussianBlur(cropped_person, (23, 23), 30)
                    frame[y1:y2, x1:x2] = sub_face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

        progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    out.release()
    progress.empty()

st.set_page_config(page_title="🎯 Smart Human Tracker", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(90deg, #e0eafc 0%, #cfdef3 100%);
    }
    .stSidebar {
        background-color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: red !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Smart Human Tracker")
st.markdown("Upload a video and track people with **Normal**, **Thermal**, or **Blurred** views! ✨")

st.sidebar.header("🛠 Settings")
tracking_mode = st.sidebar.selectbox("Tracking Mode", ["Normal", "Thermal", "Blur Background"])
frame_skip = st.sidebar.slider("Frame Skip (Higher = Faster)", 1, 10, 1)
uploaded_video = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video:
    st.video(uploaded_video)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_video.read())
        temp_input_path = temp_input.name

    temp_output_path = os.path.join(tempfile.gettempdir(), "output_tracked.mp4")

    mode_map = {
        "Normal": "normal",
        "Thermal": "thermal",
        "Blur Background": "blur"
    }
    mode = mode_map[tracking_mode]

    with st.spinner(f"🔄 Processing your video with **{tracking_mode}** mode... Please wait!"):
        process_video(temp_input_path, temp_output_path, mode=mode, frame_skip=frame_skip)

    st.success("✅ Processing complete! See the output below:")
    st.video(temp_output_path)

    with open(temp_output_path, "rb") as file:
        st.download_button("⬇️ Download Processed Video", file, "tracked_output.mp4", mime="video/mp4")
