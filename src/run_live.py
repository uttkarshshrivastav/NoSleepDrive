# run_live.py
import cv2
import time

from preprocessing.roi_extractor import extract_rois
from inference.eye_infer import load_eye_model, infer_eye
from inference.mouth_infer import load_yawn_model, infer_yawn
from inference.eye_fusion import fuse_eye_signals
from temporal.eye_temporal import EyeTemporalTracker
from temporal.mouth_temporal import YawnTemporalTracker

# Checkpoint paths
EYE_CKPT = r"C:\Users\vivek\Desktop\project folder\src\training\training model\eye_model\training\checkpoints\new_changed_better_data4layer_20eoches.pth"
YAWN_CKPT = r"C:\Users\vivek\Desktop\project folder\src\training\training model\yawn_model\checkpoints\best one_yawn_5layers_10epochs .pth"

print("Loading models...")
eye_model = load_eye_model(EYE_CKPT)
yawn_model = load_yawn_model(YAWN_CKPT)
print("Models loaded!")

eye_tracker = EyeTemporalTracker(
    close_th=0.4,
    open_th=0.3,
    drowsy_duration=1.5
)
yawn_tracker = YawnTemporalTracker()

#cap = cv2.VideoCapture(r"C:\Users\vivek\Desktop\project folder\WIN_20251229_13_44_36_Pro.mp4")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()

print("Starting...")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count = frame_count + 1
    t = time.time()

    # getting rois
    rois = extract_rois(frame)

    # sending rois to models and getting probability of closure and confidence score 
    left_eye = infer_eye(rois.get("left_eye"), eye_model)
    right_eye = infer_eye(rois.get("right_eye"), eye_model)
    #eye signal combining of signal of both eyes 
    eye_signal = fuse_eye_signals(left_eye, right_eye)

    # yawn model geetting rois 
    yawn_signal = infer_yawn(rois.get("mouth"), yawn_model)

    # Temporal tracking
    eye_drowsy = eye_tracker.update(eye_signal, t)
    yawn_event = yawn_tracker.update(yawn_signal, t)

    # Alert
    drowsy_alert = eye_drowsy or yawn_event

    #  print every frames
    if frame_count % 30 == 0:
        if eye_signal:
            eye_prob = eye_signal['prob']
            print("Frame " + str(frame_count) + " | Eye prob: " + str(round(eye_prob, 2)) + " | State: " + eye_tracker.eye_state)
        else:
            print("Frame " + str(frame_count) + " | Eye: None")

    # Display text on frame
    y = 30
    if eye_signal:
        eye_text = "Eye p(closed): " + str(round(eye_signal['prob'], 2))
        cv2.putText(frame, eye_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y = y + 25

    if yawn_signal:
        yawn_text = "Yawn p: " + str(round(yawn_signal['prob'], 2))
        cv2.putText(frame, yawn_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y = y + 25
    state_text = "Eye state: " + eye_tracker.eye_state
    if (str(eye_tracker.eye_state) == str("OPEN")):
        cv2.putText(frame, state_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else :
        cv2.putText(frame, state_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y = y + 30

    if drowsy_alert:
        cv2.putText(frame, "DROWSINESS ALERT", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done! Processed " + str(frame_count) + " frames")