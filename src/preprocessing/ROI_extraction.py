# importing the libraries
import cv2
import mediapipe as mp
import time 
import os

SAVE_ROIS = True            # turn OFF when not saving
SAVE_EVERY_N_FRAMES = 20    # save 1 frame every N frames
frame_count = 0

os.makedirs("saved_rois/left_eye", exist_ok=True)
os.makedirs("saved_rois/right_eye", exist_ok=True)
os.makedirs("saved_rois/mouth", exist_ok=True)


# importing FaceMesh from MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# setting parameters for FaceMesh 
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,         
    max_num_faces=1,                  
    min_detection_confidence=0.6,     
    min_tracking_confidence=0.6       
)

# to video capture from OpenCV
video = cv2.VideoCapture(0)
 #indexes of prdefined topology areas around the eye and mouth region
left_eye_index= [33, 133, 160, 159, 158, 144, 145, 153]
right_eye_index=[362, 263, 387, 386, 385, 373, 374, 380]
mouth_index=[61, 291, 81, 178, 13, 14, 402, 308]


prev_time=0
# reading the video stream and to check frames are coming or not
while True:
    ret, frame_captured = video.read()
    if not ret:
        break

    # converting  input from bgr to rgb
    frame_rgb = cv2.cvtColor(frame_captured, cv2.COLOR_BGR2RGB)

    # applying Facemesh on the RGB frame
    face_mesh_coordinates = face_mesh.process(frame_rgb)
    #  if face not detected this prevents breaking of code and checking the reading of frames
    if not face_mesh_coordinates.multi_face_landmarks:
        cv2.imshow("FaceMesh Live", frame_captured)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue   # skip this frame safely



     # iteration on every face detected (acts like an array if multiple faces)
    for face_landmarks in face_mesh_coordinates.multi_face_landmarks:

            # putting the landmarks into a variable
            # landmarks contains 468 points on the face
            # each landmark has x, y, z values (normalized)  

            landmarks = face_landmarks.landmark

            # getting frame dimensions
            h, w, _ = frame_captured.shape
           

            # iterating through all 468 landmarks
            for lm in landmarks:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h) 

                # drawing each landmark point on the face
              #  cv2.circle(frame_captured, (x_px, y_px), 1, (0, 255, 0), -1)
            #left eye ke lia alag se
            left_eye_points=[]
            for index in left_eye_index:
                lm=landmarks[index]
                x_px=int(lm.x *w)
                y_px=int (lm.y*h)
                left_eye_points.append((x_px,y_px))
                #drawing blue over left eye 
                #cv2.circle(frame_captured,(x_px,y_px),1,(255,0,0), -1)


            #taking out coordinates of each point 
            xle = [p[0] for p in left_eye_points]
            yle = [p[1] for p in left_eye_points]
            x_min = min(xle)
            x_max = max(xle)
            y_min = min(yle)
            y_max = max(yle)

             #applying padding to the region for some surrounding region of the roi
            padding = 10

            x_min -= padding
            x_max += padding
            y_min -= padding
            y_max += padding
            #preventing padding from breaking the code 
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max) 


            #drawing a bounding box around roi
            cv2.rectangle(
            frame_captured,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 255),
            1
            )

            left_eye_roi = frame_captured[y_min:y_max, x_min:x_max]
            if left_eye_roi.size==0:
                left_eye_roi=None

            if left_eye_roi is not None:
                left_eye_roi=cv2.resize(left_eye_roi,(64,64)) 
            
            if left_eye_roi is not None:
                cv2.imshow("Left Eye 64x64", left_eye_roi)
 
        

              

              # RIGHT EYE START HERE
            right_eye_points=[]
            for index in right_eye_index:
                lm=landmarks[index]
                x_px=int(lm.x *w)
                y_px=int (lm.y*h)
                right_eye_points.append((x_px,y_px))
                #drawing blue over right eye 
                #cv2.circle(frame_captured,(x_px,y_px),1,(255,0,0), -1)


                #taking out coordinates of each point 
            xre = [p[0] for p in right_eye_points]
            yre = [p[1] for p in right_eye_points]
            x_min = min(xre)
            x_max = max(xre)
            y_min = min(yre)
            y_max = max(yre)

             #applying padding to the region for some surrounding region of the roi
            padding = 10

            x_min -= padding
            x_max += padding
            y_min -= padding
            y_max += padding
            #preventing padding from breaking the code 
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max) 


            #drawing a bounding box around roi
            cv2.rectangle(
            frame_captured,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 255),
            1
            )


            right_eye_roi = frame_captured[y_min:y_max, x_min:x_max]
            
            
            if right_eye_roi.size==0:
                right_eye_roi=None

            if right_eye_roi is not None:
                right_eye_roi=cv2.resize(right_eye_roi,(64,64)) 
            
            if right_eye_roi is not None:
                cv2.imshow("right Eye 64x64",right_eye_roi)

                

            



          #MOUTH PART START HERE 
            mouth_points=[]
            for index in mouth_index:
                lm=landmarks[index]
                x_px=int(lm.x*w)
                y_px=int(lm.y*h)
                mouth_points.append((x_px,y_px))
                #drawing blue over mouth 
                #cv2.circle(frame_captured,(x_px,y_px),1,(0,0,255), -1)


                #taking out coordinates of each point 
            xm = [p[0] for p in mouth_points]
            ym = [p[1] for p in mouth_points]
            x_min = min(xm)
            x_max = max(xm)
            y_min = min(ym)
            y_max = max(ym)

             #applying padding to the region for some surrounding region of the roi
            padding = 10

            x_min -= padding
            x_max += padding
            y_min -= padding
            y_max += padding
            #preventing padding from breaking the code 
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max) 


            #drawing a bounding box around roi
            cv2.rectangle(
            frame_captured,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 255),
            1
            )


            mouth_roi = frame_captured[y_min:y_max, x_min:x_max]
            if mouth_roi.size==0:
                mouth_roi=None

            if mouth_roi is not None:
                mouth_roi=cv2.resize(mouth_roi,(128,128)) 
            
            if mouth_roi is not None:
                cv2.imshow("Left Eye 64x64", mouth_roi)

            #cheking the working 

            if left_eye_roi is not None:
                print("Left eye shape:", left_eye_roi.shape)

            if right_eye_roi is not None:
                print("right eye shape:",right_eye_roi.shape)

            if mouth_roi is not None:
                print("Mouth shape:", mouth_roi.shape)
            #checking fps
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(
            frame_captured,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
            )
            # ---------- STEP 11: CLEAN OUTPUT CONTRACT ----------
            rois = {
            "left_eye": left_eye_roi,
            "right_eye": right_eye_roi,
            "mouth": mouth_roi
            }

    # ---------- STEP 11: OPTIONAL ROI SAVING ----------
    if SAVE_ROIS and frame_count % SAVE_EVERY_N_FRAMES == 0:
        ts = int(time.time() * 1000)

    if rois["left_eye"] is not None:
        cv2.imwrite(f"saved_rois/left_eye/{ts}.jpg", rois["left_eye"])

    if rois["right_eye"] is not None:
        cv2.imwrite(f"saved_rois/right_eye/{ts}.jpg", rois["right_eye"])

    if rois["mouth"] is not None:
        cv2.imwrite(f"saved_rois/mouth/{ts}.jpg", rois["mouth"])


            
     # displaying the output frame on which the landmarks will be shown 
    cv2.imshow("FaceMesh Live", frame_captured)
    #  ESC key to break out of the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break
    frame_count += 1


# releasing the camera resource
video.release()


# destroying all OpenCV windows varns system crash
cv2.destroyAllWindows()