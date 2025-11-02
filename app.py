from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask_cors import CORS
import threading
import time
import os

app = Flask(__name__)
CORS(app)

# Model Configuration
MODEL_PATH = "models/attention_model.h5"
IMG_SIZE = 224

# Load ML Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ ML Model loaded successfully from {MODEL_PATH}")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️  Running without ML model - will use mock predictions")
    model = None
    MODEL_LOADED = False

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection

# Global state for monitoring
monitoring_active = False
monitoring_lock = threading.Lock()
camera = None
camera_lock = threading.Lock()

def get_camera():
    """Get or initialize camera safely"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("❌ Cannot open camera")
                return None
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            print("✅ Camera initialized")
        return camera

def release_camera():
    """Release camera safely"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print("📷 Camera released")

def preprocess(img):
    """Preprocess image for model prediction"""
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def predict_engagement(frame, face_detector):
    """Predict engagement status with face detection"""
    if model is None:
        # Fallback to mock prediction if model not loaded
        import random
        prob = random.uniform(0.3, 0.9)
        label = "Engaged" if prob >= 0.5 else "Not Engaged"
        confidence = prob if prob >= 0.5 else 1 - prob
        color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
        return label, confidence, color, None, "MOCK"
    
    # Face detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    
    roi = frame
    bbox = None
    source = "Full Frame"
    
    if results.detections:
        d = results.detections[0].location_data.relative_bounding_box
        h, w = frame.shape[:2]
        
        # Calculate bounding box with padding
        x1 = int(d.xmin * w)
        y1 = int(d.ymin * h)
        bw = int(d.width * w)
        bh = int(d.height * h)
        
        # Add padding
        pad_w = int(0.2 * bw)
        pad_h = int(0.5 * bh)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x1 + bw + 2*pad_w)
        y2 = min(h, y1 + bh + 2*pad_h)
        
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            bbox = (x1, y1, x2, y2)
            source = "Face Crop"
    
    # Predict
    try:
        input_img = preprocess(roi)
        if input_img is not None:
            prob = float(model.predict(input_img, verbose=0)[0][0])
        else:
            prob = 0.5
    except Exception as e:
        print(f"Prediction error: {e}")
        prob = 0.5
    
    label = "Engaged" if prob >= 0.5 else "Not Engaged"
    confidence = prob if prob >= 0.5 else 1 - prob
    color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
    
    return label, confidence, color, bbox, source

def gen_frames():
    """Generate video frames with ML predictions"""
    face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
    cam = get_camera()
    
    if cam is None:
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        _, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    fps = 0
    prev_time = time.time()
    frame_count = 0
    
    while True:
        with monitoring_lock:
            if not monitoring_active:
                break
        
        success, frame = cam.read()
        if not success or frame is None:
            print("Failed to read frame")
            time.sleep(0.1)
            continue
        
        # Get prediction with face detection
        label, confidence, color, bbox, source = predict_engagement(frame, face_detector)
        
        # Calculate FPS
        current_time = time.time()
        if current_time - prev_time > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / (current_time - prev_time))
        prev_time = current_time
        frame_count += 1
        
        # Draw bounding box if face detected
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text overlay
        cv2.putText(frame, f"Status: {label}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | {source}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Model status indicator
        mode_text = "ML MODEL ACTIVE" if MODEL_LOADED else "MOCK MODE"
        mode_color = (0, 255, 0) if MODEL_LOADED else (0, 165, 255)
        cv2.putText(frame, mode_text, (420, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    face_detector.close()
    print(f"Video stream ended. Total frames: {frame_count}")

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "message": "Student Concentration Detection API",
        "status": "active",
        "model_loaded": MODEL_LOADED,
        "model_path": MODEL_PATH if MODEL_LOADED else None,
        "monitoring": monitoring_active,
        "mode": "ML" if MODEL_LOADED else "Mock"
    })

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring endpoint"""
    global monitoring_active
    
    with monitoring_lock:
        if monitoring_active:
            return jsonify({
                "status": "already_active",
                "message": "Monitoring is already active"
            })
        
        monitoring_active = True
    
    # Initialize camera
    cam = get_camera()
    if cam is None:
        monitoring_active = False
        return jsonify({
            "status": "error",
            "message": "Cannot access camera"
        }), 500
    
    return jsonify({
        "status": "success",
        "message": f"Monitoring started ({'ML Model' if MODEL_LOADED else 'Mock Mode'})",
        "model_loaded": MODEL_LOADED
    })

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop monitoring endpoint"""
    global monitoring_active
    
    with monitoring_lock:
        monitoring_active = False
    
    time.sleep(0.5)
    release_camera()
    
    return jsonify({
        "status": "success",
        "message": "Monitoring stopped successfully"
    })

@app.route('/monitoring_status')
def monitoring_status():
    """Get current monitoring status"""
    return jsonify({
        "monitoring": monitoring_active,
        "model_loaded": MODEL_LOADED,
        "mode": "ML" if MODEL_LOADED else "Mock"
    })

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    global monitoring_active
    
    with monitoring_lock:
        if not monitoring_active:
            monitoring_active = True
    
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Single frame prediction"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    try:
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        
        face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
        label, confidence, _, _, source = predict_engagement(img, face_detector)
        face_detector.close()
        
        return jsonify({
            "status": label,
            "confidence": float(confidence),
            "source": source,
            "model_loaded": MODEL_LOADED
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     print("="*70)
#     print("🚀 Student Concentration Detection Server")
#     print("="*70)
    
#     if MODEL_LOADED:
#         print(f"✅ ML Model: {MODEL_PATH}")
#         print(f"✅ Model loaded successfully - REAL predictions enabled")
#     else:
#         print(f"⚠️  ML Model: NOT LOADED")
#         print(f"⚠️  Using mock predictions - Install TensorFlow and train model")
    
#     print("")
#     print(f"🌐 Server running on: http://0.0.0.0:5050")
#     print(f"📱 For Flutter app: http://YOUR_IP:5050")
#     print("")
#     print("Endpoints:")
#     print("  GET  / - Health check")
#     print("  POST /start_monitoring - Start monitoring")
#     print("  POST /stop_monitoring - Stop monitoring")
#     print("  GET  /monitoring_status - Check status")
#     print("  GET  /video_feed - Video stream")
#     print("  POST /predict_frame - Single frame prediction")
#     print("="*70)
#     print("")
    
#     app.run(host='0.0.0.0', port=5050, debug=True, threaded=True)


if __name__ == '__main__':
    # Get port from environment variable (Cloud Run provides this)
    port = int(os.environ.get('PORT', 5050))
    
    print("="*70)
    print("🚀 Student Concentration Detection Server")
    print("="*70)
    
    if MODEL_LOADED:
        print(f"✅ ML Model: {MODEL_PATH}")
        print(f"✅ Model loaded successfully - REAL predictions enabled")
    else:
        print(f"⚠️  ML Model: NOT LOADED")
        print(f"⚠️  Using mock predictions - Install TensorFlow and train model")
    
    print("")
    print(f"🌐 Server running on: http://0.0.0.0:{port}")
    print(f"📱 Cloud Run URL will be provided after deployment")
    print("")
    print("Endpoints:")
    print("  GET  / - Health check")
    print("  POST /start_monitoring - Start monitoring")
    print("  POST /stop_monitoring - Stop monitoring")
    print("  GET  /monitoring_status - Check status")
    print("  GET  /video_feed - Video stream")
    print("  POST /predict_frame - Single frame prediction")
    print("="*70)
    print("")
    
    # IMPORTANT: Remove debug=True for production
    # Cloud Run doesn't support debug mode
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# from flask import Flask, Response, jsonify, request
# import cv2
# import numpy as np
# import tensorflow as tf
# from flask_cors import CORS
# import threading
# import time
# import os

# app = Flask(__name__)
# CORS(app)

# # Model Configuration
# MODEL_PATH = "models/attention_model.h5"
# IMG_SIZE = 224

# # Load ML Model
# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
#     print(f"✅ ML Model loaded successfully from {MODEL_PATH}")
#     MODEL_LOADED = True
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     print("⚠️  Running without ML model - will use mock predictions")
#     model = None
#     MODEL_LOADED = False

# # Try to load MediaPipe (optional)
# try:
#     import mediapipe as mp
#     mp_face = mp.solutions.face_detection
#     MEDIAPIPE_AVAILABLE = True
#     print("✅ MediaPipe loaded successfully")
# except ImportError:
#     MEDIAPIPE_AVAILABLE = False
#     print("⚠️  MediaPipe not available - using full frame detection")

# # Global state for monitoring
# monitoring_active = False
# monitoring_lock = threading.Lock()
# camera = None
# camera_lock = threading.Lock()

# def get_camera():
#     """Get or initialize camera safely"""
#     global camera
#     with camera_lock:
#         if camera is None or not camera.isOpened():
#             camera = cv2.VideoCapture(0)
#             if not camera.isOpened():
#                 print("❌ Cannot open camera")
#                 return None
#             camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#             camera.set(cv2.CAP_PROP_FPS, 30)
#             print("✅ Camera initialized")
#         return camera

# def release_camera():
#     """Release camera safely"""
#     global camera
#     with camera_lock:
#         if camera is not None:
#             camera.release()
#             camera = None
#             print("📷 Camera released")

# def preprocess(img):
#     """Preprocess image for model prediction"""
#     try:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#         img = img.astype("float32") / 255.0
#         return np.expand_dims(img, axis=0)
#     except Exception as e:
#         print(f"Preprocessing error: {e}")
#         return None

# def predict_engagement(frame, face_detector=None):
#     """Predict engagement status with optional face detection"""
#     if model is None:
#         # Fallback to mock prediction if model not loaded
#         import random
#         prob = random.uniform(0.3, 0.9)
#         label = "Engaged" if prob >= 0.5 else "Not Engaged"
#         confidence = prob if prob >= 0.5 else 1 - prob
#         color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
#         return label, confidence, color, None, "MOCK"
    
#     roi = frame
#     bbox = None
#     source = "Full Frame"
    
#     # Use MediaPipe if available
#     if MEDIAPIPE_AVAILABLE and face_detector:
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_detector.process(rgb)
        
#         if results.detections:
#             d = results.detections[0].location_data.relative_bounding_box
#             h, w = frame.shape[:2]
            
#             # Calculate bounding box with padding
#             x1 = int(d.xmin * w)
#             y1 = int(d.ymin * h)
#             bw = int(d.width * w)
#             bh = int(d.height * h)
            
#             # Add padding
#             pad_w = int(0.2 * bw)
#             pad_h = int(0.5 * bh)
#             x1 = max(0, x1 - pad_w)
#             y1 = max(0, y1 - pad_h)
#             x2 = min(w, x1 + bw + 2*pad_w)
#             y2 = min(h, y1 + bh + 2*pad_h)
            
#             if x2 > x1 and y2 > y1:
#                 roi = frame[y1:y2, x1:x2]
#                 bbox = (x1, y1, x2, y2)
#                 source = "Face Crop"
    
#     # Predict
#     try:
#         input_img = preprocess(roi)
#         if input_img is not None:
#             prob = float(model.predict(input_img, verbose=0)[0][0])
#         else:
#             prob = 0.5
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         prob = 0.5
    
#     label = "Engaged" if prob >= 0.5 else "Not Engaged"
#     confidence = prob if prob >= 0.5 else 1 - prob
#     color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
    
#     return label, confidence, color, bbox, source

# def gen_frames():
#     """Generate video frames with ML predictions"""
#     face_detector = None
#     if MEDIAPIPE_AVAILABLE:
#         face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
    
#     cam = get_camera()
    
#     if cam is None:
#         # Generate error frame
#         error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#         cv2.putText(error_frame, "Camera Error", (180, 240),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
#         _, buffer = cv2.imencode('.jpg', error_frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#         return
    
#     fps = 0
#     prev_time = time.time()
#     frame_count = 0
    
#     while True:
#         with monitoring_lock:
#             if not monitoring_active:
#                 break
        
#         success, frame = cam.read()
#         if not success or frame is None:
#             print("Failed to read frame")
#             time.sleep(0.1)
#             continue
        
#         # Get prediction with optional face detection
#         label, confidence, color, bbox, source = predict_engagement(frame, face_detector)
        
#         # Calculate FPS
#         current_time = time.time()
#         if current_time - prev_time > 0:
#             fps = 0.9 * fps + 0.1 * (1.0 / (current_time - prev_time))
#         prev_time = current_time
#         frame_count += 1
        
#         # Draw bounding box if face detected
#         if bbox:
#             x1, y1, x2, y2 = bbox
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
#         # Draw background for text
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (0, 0), (640, 110), (0, 0, 0), -1)
#         cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
#         # Draw text overlay
#         cv2.putText(frame, f"Status: {label}", (20, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(frame, f"FPS: {fps:.1f} | {source}", (20, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         # Model status indicator
#         mode_text = "ML MODEL ACTIVE" if MODEL_LOADED else "MOCK MODE"
#         mode_color = (0, 255, 0) if MODEL_LOADED else (0, 165, 255)
#         cv2.putText(frame, mode_text, (420, 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
#         # Encode frame
#         _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#         frame_bytes = buffer.tobytes()
        
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
#     if face_detector and MEDIAPIPE_AVAILABLE:
#         face_detector.close()
#     print(f"Video stream ended. Total frames: {frame_count}")

# @app.route('/')
# def index():
#     """Health check endpoint"""
#     return jsonify({
#         "message": "Student Concentration Detection API",
#         "status": "active",
#         "model_loaded": MODEL_LOADED,
#         "mediapipe_available": MEDIAPIPE_AVAILABLE,
#         "model_path": MODEL_PATH if MODEL_LOADED else None,
#         "monitoring": monitoring_active,
#         "mode": "ML" if MODEL_LOADED else "Mock",
#         "environment": os.environ.get('RENDER', 'local')
#     })

# @app.route('/start_monitoring', methods=['POST'])
# def start_monitoring():
#     """Start monitoring endpoint"""
#     global monitoring_active
    
#     with monitoring_lock:
#         if monitoring_active:
#             return jsonify({
#                 "status": "already_active",
#                 "message": "Monitoring is already active"
#             })
        
#         monitoring_active = True
    
#     # Initialize camera
#     cam = get_camera()
#     if cam is None:
#         monitoring_active = False
#         return jsonify({
#             "status": "error",
#             "message": "Cannot access camera"
#         }), 500
    
#     return jsonify({
#         "status": "success",
#         "message": f"Monitoring started ({'ML Model' if MODEL_LOADED else 'Mock Mode'})",
#         "model_loaded": MODEL_LOADED
#     })

# @app.route('/stop_monitoring', methods=['POST'])
# def stop_monitoring():
#     """Stop monitoring endpoint"""
#     global monitoring_active
    
#     with monitoring_lock:
#         monitoring_active = False
    
#     time.sleep(0.5)
#     release_camera()
    
#     return jsonify({
#         "status": "success",
#         "message": "Monitoring stopped successfully"
#     })

# @app.route('/monitoring_status')
# def monitoring_status():
#     """Get current monitoring status"""
#     return jsonify({
#         "monitoring": monitoring_active,
#         "model_loaded": MODEL_LOADED,
#         "mediapipe_available": MEDIAPIPE_AVAILABLE,
#         "mode": "ML" if MODEL_LOADED else "Mock"
#     })

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route"""
#     global monitoring_active
    
#     with monitoring_lock:
#         if not monitoring_active:
#             monitoring_active = True
    
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/predict_frame', methods=['POST'])
# def predict_frame():
#     """Single frame prediction"""
#     if model is None:
#         return jsonify({"error": "Model not loaded"}), 500
    
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files['file']
    
#     try:
#         npimg = np.frombuffer(file.read(), np.uint8)
#         img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
#         if img is None:
#             return jsonify({"error": "Invalid image"}), 400
        
#         face_detector = None
#         if MEDIAPIPE_AVAILABLE:
#             face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
        
#         label, confidence, _, _, source = predict_engagement(img, face_detector)
        
#         if face_detector and MEDIAPIPE_AVAILABLE:
#             face_detector.close()
        
#         return jsonify({
#             "status": label,
#             "confidence": float(confidence),
#             "source": source,
#             "model_loaded": MODEL_LOADED
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     # Get port from environment variable (Railway/Render sets this)
#     port = int(os.environ.get('PORT', 5050))
    
#     # Check if running in production
#     is_production = os.environ.get('RENDER', False) or os.environ.get('RAILWAY_ENVIRONMENT', False)
#     debug_mode = not is_production
    
#     print("="*70)
#     print("🚀 Student Concentration Detection Server")
#     print("="*70)
    
#     if MODEL_LOADED:
#         print(f"✅ ML Model: {MODEL_PATH}")
#         print(f"✅ Model loaded successfully - REAL predictions enabled")
#     else:
#         print(f"⚠️  ML Model: NOT LOADED")
#         print(f"⚠️  Using mock predictions")
    
#     if MEDIAPIPE_AVAILABLE:
#         print(f"✅ MediaPipe: Available (Face detection enabled)")
#     else:
#         print(f"⚠️  MediaPipe: Not available (Using full frame)")
    
#     print("")
#     print(f"🌐 Environment: {'PRODUCTION' if is_production else 'LOCAL DEVELOPMENT'}")
#     print(f"🌐 Server running on: http://0.0.0.0:{port}")
#     if not is_production:
#         print(f"📱 For Flutter app: http://YOUR_IP:{port}")
#     print("")
#     print("Endpoints:")
#     print("  GET  / - Health check")
#     print("  POST /start_monitoring - Start monitoring")
#     print("  POST /stop_monitoring - Stop monitoring")
#     print("  GET  /monitoring_status - Check status")
#     print("  GET  /video_feed - Video stream")
#     print("  POST /predict_frame - Single frame prediction")
#     print("="*70)
#     print("")
    
#     # Run with appropriate settings
#     app.run(
#         host='0.0.0.0',
#         port=port,
#         debug=debug_mode,
#         threaded=True
#     )