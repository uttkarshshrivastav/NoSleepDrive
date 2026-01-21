
from flask import Flask, jsonify, render_template, request, Response
from shared_state import shared_state
import shared_frame
import cv2
import time
import numpy as np
import threading

app = Flask(__name__)

# Global variable to store latest frame from web upload
frames = None
frames_lock = threading.Lock()


@app.route("/status")
def status():
    return jsonify(shared_state)

# @app.route("/")
# def index():
#     return render_template("state_change.html")

# def generate_frames():
#     while True:
#         if shared_frame is None:
#             time.sleep(0.01)
#             continue

#         ret, buffer = cv2.imencode(".jpg", shared_frame.shared_frame)
#         if not ret:
#             continue

#         frame_bytes = buffer.tobytes()

#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" + 
#             frame_bytes +
#             b"\r\n"
#         )


# @app.route("/video")
# def video():
#     return Response(
#         generate_frames(),
#         mimetype="multipart/x-mixed-replace; boundary=frame"    
        
#     )

# When someone visits the homepage "/"
@app.route("/")
def index():
    # Send them the index.html file
    return render_template("read.html")

# When someone sends an image to "analyze_frame"
@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    global frames
    
    # Checking if the sent file called frame
    if "frame" not in request.files:
        return jsonify({"error": "No frame"}), 400
    
    # Get the uploaded file
    file = request.files["frame"]
    img_bytes = file.read()
    
    # Convert bytes to a numpy array and to colour image to 3 channel image
    np_buf = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    with frames_lock:
        frames = frame.copy()
    
    # Get image dimensions
    height, width, channels = frame.shape
  
    return jsonify(shared_state)

# enabling debuging 
if __name__ == "__main__":
    app.run(debug=True)