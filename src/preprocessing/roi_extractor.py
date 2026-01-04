import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,  # can be increased later for multi-face support
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# landmark indices
left_eye_index = [33, 133, 160, 159, 158, 144, 145, 153]
right_eye_index = [362, 263, 387, 386, 385, 373, 374, 380]
mouth_index = [61, 291, 81, 178, 13, 14, 402, 308]


def extract_rois(frame):
    # default values if roi is not detected
    left_eye_roi = None
    right_eye_roi = None
    mouth_roi = None

    # converting input from bgr to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # processing frame through face mesh
    face_mesh_coordinates = face_mesh.process(frame_rgb)

    # if face not detected this prevents breaking of code
    if not face_mesh_coordinates.multi_face_landmarks:
        return {
            "left_eye": None,
            "right_eye": None,
            "mouth": None
        }

    
    faces = face_mesh_coordinates.multi_face_landmarks
    face_landmarks = faces[0]

    landmarks = face_landmarks.landmark
    h, w, _ = frame.shape

    # LEFT EYE ROI
    left_eye_points = []
    for index in left_eye_index:
        lm = landmarks[index]
        left_eye_points.append((int(lm.x * w), int(lm.y * h)))

    xle = [p[0] for p in left_eye_points]
    yle = [p[1] for p in left_eye_points]

    x_min, x_max = min(xle), max(xle)
    y_min, y_max = min(yle), max(yle)

    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)

    left_eye_roi = frame[y_min:y_max, x_min:x_max]
    if left_eye_roi.size == 0:
        left_eye_roi = None
    else:
        left_eye_roi = cv2.resize(left_eye_roi, (64, 64))

    # RIGHT EYE ROI 
    right_eye_points = []
    for index in right_eye_index:
        lm = landmarks[index]
        right_eye_points.append((int(lm.x * w), int(lm.y * h)))

    xre = [p[0] for p in right_eye_points]
    yre = [p[1] for p in right_eye_points]

    x_min, x_max = min(xre), max(xre)
    y_min, y_max = min(yre), max(yre)

    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)

    right_eye_roi = frame[y_min:y_max, x_min:x_max]
    if right_eye_roi.size == 0:
        right_eye_roi = None
    else:
        right_eye_roi = cv2.resize(right_eye_roi, (64, 64))

    # MOUTH ROI
    mouth_points = []
    for index in mouth_index:
        lm = landmarks[index]
        mouth_points.append((int(lm.x * w), int(lm.y * h)))

    xm = [p[0] for p in mouth_points]
    ym = [p[1] for p in mouth_points]

    x_min, x_max = min(xm), max(xm)
    y_min, y_max = min(ym), max(ym)

    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)

    mouth_roi = frame[y_min:y_max, x_min:x_max]
    if mouth_roi.size == 0:
        mouth_roi = None
    else:
        mouth_roi = cv2.resize(mouth_roi, (128, 128))

    # returning all the rois
    return {
        "left_eye": left_eye_roi,
        "right_eye": right_eye_roi,
        "mouth": mouth_roi
    }
