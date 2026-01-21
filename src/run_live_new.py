# run_live.py
import cv2
import time

from preprocessing.roi_extractor import extract_rois
from inference.eye_infer import load_eye_model, infer_eye
from inference.mouth_infer import load_yawn_model, infer_yawn
from inference.eye_fusion import fuse_eye_signals
from temporal.eye_temporal import EyeTemporalTracker
from temporal.mouth_temporal import YawnTemporalTracker
from threading import Thread
# from web_app import app,frame
import web_app_new
import shared_state
import shared_frame
import copy


# Checkpoint paths
EYE_CKPT = r"C:\Users\vivek\Desktop\project folder\src\training\training model\eye_model\training\checkpoints\new_changed_better_data4layer_20eoches.pth"
YAWN_CKPT = r"C:\Users\vivek\Desktop\project folder\src\training\training model\yawn_model\checkpoints\best one_yawn_5layers_10epochs .pth"

print("Loading models...")
eye_model = load_eye_model(EYE_CKPT)
yawn_model = load_yawn_model(YAWN_CKPT)
print("Models loaded!")

eye_tracker = EyeTemporalTracker(close_th=0.4, open_th=0.3, drowsy_duration=1.5)
yawn_tracker = YawnTemporalTracker()

# cap = cv2.VideoCapture(r"C:\Users\vivek\Desktop\project folder\WIN_20251229_13_44_36_Pro.mp4")
# cap = cv2.VideoCapture(0)
# cap=web_app.frames

# if not cap.isOpened():
#     print("Error: Cannot open video!")
#     exit()

print("Starting...")
frame_count = 0


def update_shared_state(eye_signal, yawn_signal, eye_state, drowsy_alert):
    if eye_signal is not None:
        shared_state.shared_state["eye_closed_prob"] = eye_signal["prob"]
    else:
        shared_state.shared_state["eye_closed_prob"] = 1
    if yawn_signal is not None:
        shared_state.shared_state["yawn_prob"] = yawn_signal["prob"]
    else:
        shared_state.shared_state["yawn_prob"] = 0

    shared_state.shared_state["eye_state"] = eye_state
    if drowsy_alert:
        shared_state.shared_state["drowsy"] = True

    else:
        shared_state.shared_state["drowsy"] = False
    # "level of drowsyness"=


def start_flask():
    web_app_new.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

flask_thread = Thread(target=start_flask, daemon=True)
flask_thread.start()

# buffer to start flask 
time.sleep(2)
print("Flask server started on http://0.0.0.0:5000")

while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break
    
    if web_app_new.frames is None:
        time.sleep(0.01)
        continue

    # getting current frame 
    with web_app_new.frames_lock:
        current_frame = web_app_new.frames.copy()

    frame_count += 1
    t = time.time()

    # getting rois
    rois = extract_rois(current_frame)

    # sending rois to models and getting probability of closure and confidence score
    left_eye = infer_eye(rois.get("left_eye"), eye_model)
    right_eye = infer_eye(rois.get("right_eye"), eye_model)
    # eye signal combining of signal of both eyes
    eye_signal = fuse_eye_signals(left_eye, right_eye)
    # yawn model geetting rois
    yawn_signal = infer_yawn(rois.get("mouth"), yawn_model)
    # Temporal tracking
    eye_drowsy = eye_tracker.update(eye_signal, t)
    yawn_event = yawn_tracker.update(yawn_signal, t)
    #alert if any of two shows alert 
    drowsy_alert = eye_drowsy or yawn_event
    alert_level = 0
    # if drowsy_alert is True:
    #     alert_level +=1
        
    

    eye_state = eye_tracker.eye_state

    update_shared_state(eye_signal, yawn_signal, eye_state, drowsy_alert)

    #  print every frames
    if frame_count % 30 == 0:
        if eye_signal:
            eye_prob = eye_signal["prob"]
            print(
                "Frame "
                + str(frame_count)
                + " | Eye prob: "
                + str(round(eye_prob, 2))
                + " | State: "
                + eye_tracker.eye_state
            )
        else:
            print("Frame " + str(frame_count) + " | Eye: None")

    # Display text on frame
    y = 30
    if eye_signal:
        eye_text = "Eye p(closed): " + str(round(eye_signal["prob"], 2))
        cv2.putText(
            current_frame, eye_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        y = y + 25

    if yawn_signal:
        yawn_text = "Yawn p: " + str(round(yawn_signal["prob"], 2))
        cv2.putText(
            current_frame,
            yawn_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        y = y + 25

    state_text = "Eye state: " + eye_state
    if str(eye_tracker.eye_state) == str("OPEN"):
        cv2.putText(
            current_frame,
            state_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
    else:
        cv2.putText(
            current_frame,
            state_text, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
    y = y + 30

    if drowsy_alert:
        cv2.putText(
            current_frame,
            "DROWSINESS ALERT",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
        )

    # Update shared frame for web streaming
    shared_frame.shared_frame = current_frame.copy()

    cv2.imshow("Driver Monitor", current_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cap.release()
cv2.destroyAllWindows()
print("Done! Processed " + str(frame_count) + " frames")