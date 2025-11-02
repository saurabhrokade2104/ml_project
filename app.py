#original

# from flask import Flask, Response, jsonify, request
# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
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

# # Initialize MediaPipe Face Detection
# mp_face = mp.solutions.face_detection

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

# def predict_engagement(frame, face_detector):
#     """Predict engagement status with face detection"""
#     if model is None:
#         # Fallback to mock prediction if model not loaded
#         import random
#         prob = random.uniform(0.3, 0.9)
#         label = "Engaged" if prob >= 0.5 else "Not Engaged"
#         confidence = prob if prob >= 0.5 else 1 - prob
#         color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
#         return label, confidence, color, None, "MOCK"
    
#     # Face detection
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_detector.process(rgb)
    
#     roi = frame
#     bbox = None
#     source = "Full Frame"
    
#     if results.detections:
#         d = results.detections[0].location_data.relative_bounding_box
#         h, w = frame.shape[:2]
        
#         # Calculate bounding box with padding
#         x1 = int(d.xmin * w)
#         y1 = int(d.ymin * h)
#         bw = int(d.width * w)
#         bh = int(d.height * h)
        
#         # Add padding
#         pad_w = int(0.2 * bw)
#         pad_h = int(0.5 * bh)
#         x1 = max(0, x1 - pad_w)
#         y1 = max(0, y1 - pad_h)
#         x2 = min(w, x1 + bw + 2*pad_w)
#         y2 = min(h, y1 + bh + 2*pad_h)
        
#         if x2 > x1 and y2 > y1:
#             roi = frame[y1:y2, x1:x2]
#             bbox = (x1, y1, x2, y2)
#             source = "Face Crop"
    
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
#     face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
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
        
#         # Get prediction with face detection
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
    
#     face_detector.close()
#     print(f"Video stream ended. Total frames: {frame_count}")

# @app.route('/')
# def index():
#     """Health check endpoint"""
#     return jsonify({
#         "message": "Student Concentration Detection API",
#         "status": "active",
#         "model_loaded": MODEL_LOADED,
#         "model_path": MODEL_PATH if MODEL_LOADED else None,
#         "monitoring": monitoring_active,
#         "mode": "ML" if MODEL_LOADED else "Mock"
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
        
#         face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
#         label, confidence, _, _, source = predict_engagement(img, face_detector)
#         face_detector.close()
        
#         return jsonify({
#             "status": label,
#             "confidence": float(confidence),
#             "source": source,
#             "model_loaded": MODEL_LOADED
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

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

from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import os
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "models/attention_model.h5"
IMG_SIZE = 224

# Detect environment
IS_PRODUCTION = os.environ.get('RENDER') or os.environ.get('RAILWAY_ENVIRONMENT')
CAMERA_AVAILABLE = False  # Will be detected at runtime

# ============================================================================
# LOAD ML MODEL - MANDATORY
# ============================================================================
MODEL_LOADED = False
model = None

def load_model():
    """Load ML model with multiple attempts"""
    global model, MODEL_LOADED
    
    print("🔄 Attempting to load ML model...")
    
    # Try different possible paths
    possible_paths = [
        MODEL_PATH,
        os.path.join(os.getcwd(), MODEL_PATH),
        os.path.join(os.getcwd(), "attention_model.h5"),
        "attention_model.h5"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"📂 Found model at: {path}")
                model = tf.keras.models.load_model(path)
                MODEL_LOADED = True
                print(f"✅ ML Model loaded successfully!")
                print(f"📊 Model input shape: {model.input_shape}")
                print(f"📊 Model output shape: {model.output_shape}")
                return True
            except Exception as e:
                print(f"❌ Error loading model from {path}: {e}")
                continue
    
    print("=" * 70)
    print("⚠️  WARNING: ML MODEL NOT FOUND!")
    print("=" * 70)
    print(f"Expected model path: {MODEL_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    if os.path.exists('models'):
        print(f"Files in models directory: {os.listdir('models')}")
    print("")
    print("To fix this:")
    print("1. Create 'models' folder in project root")
    print("2. Place 'attention_model.h5' inside models/")
    print("3. Redeploy the application")
    print("=" * 70)
    return False

# Load model at startup
load_model()

# ============================================================================
# MEDIAPIPE SETUP (Optional but recommended)
# ============================================================================
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully - Face detection enabled")
except Exception as e:
    print(f"⚠️  MediaPipe not available: {e}")
    print("⚠️  Will use full frame for predictions")
    MEDIAPIPE_AVAILABLE = False
    mp_face = None

# ============================================================================
# GLOBAL STATE
# ============================================================================
monitoring_active = False
monitoring_lock = threading.Lock()
camera = None
camera_lock = threading.Lock()
frame_count = 0

# ============================================================================
# CAMERA MANAGEMENT
# ============================================================================
def detect_camera_availability():
    """Detect if a real camera is available"""
    global CAMERA_AVAILABLE
    
    if IS_PRODUCTION:
        print("🌐 Production environment detected - using mock camera (no physical camera on server)")
        CAMERA_AVAILABLE = False
        return False
    
    try:
        test_cam = cv2.VideoCapture(0)
        if test_cam.isOpened():
            ret, frame = test_cam.read()
            test_cam.release()
            if ret and frame is not None:
                print("✅ Real camera detected and working")
                CAMERA_AVAILABLE = True
                return True
        test_cam.release()
    except Exception as e:
        print(f"⚠️  Camera detection failed: {e}")
    
    print("⚠️  No camera available - will use mock camera for demonstration")
    CAMERA_AVAILABLE = False
    return False

def get_camera():
    """Get or initialize camera safely"""
    global camera, CAMERA_AVAILABLE
    
    if not CAMERA_AVAILABLE:
        return None  # Will use mock camera
    
    with camera_lock:
        if camera is None or not camera.isOpened():
            try:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    print("❌ Cannot open camera")
                    CAMERA_AVAILABLE = False
                    return None
                
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                print("✅ Camera initialized successfully")
            except Exception as e:
                print(f"❌ Camera initialization error: {e}")
                CAMERA_AVAILABLE = False
                return None
        
        return camera

def release_camera():
    """Release camera safely"""
    global camera
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
                print("📷 Camera released")
            except:
                pass
            camera = None

# ============================================================================
# ML PREDICTION FUNCTIONS - USING REAL MODEL
# ============================================================================
def preprocess(img):
    """Preprocess image for model prediction"""
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

def predict_engagement(frame, face_detector=None):
    """
    Predict engagement status using ML MODEL
    NO MOCK PREDICTIONS - Only real ML model predictions
    """
    if not MODEL_LOADED or model is None:
        # Return error state if model not loaded
        return "MODEL ERROR", 0.0, (0, 0, 255), None, "NO MODEL"
    
    # Default values
    roi = frame
    bbox = None
    source = "Full Frame"
    
    # Face detection (if MediaPipe available) - improves accuracy
    if MEDIAPIPE_AVAILABLE and face_detector is not None:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)
            
            if results.detections:
                d = results.detections[0].location_data.relative_bounding_box
                h, w = frame.shape[:2]
                
                # Calculate bounding box with padding
                x1 = int(max(0, d.xmin * w))
                y1 = int(max(0, d.ymin * h))
                bw = int(d.width * w)
                bh = int(d.height * h)
                
                # Add padding for better context
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
        except Exception as e:
            print(f"⚠️ Face detection error: {e}")
    
    # ML Model Prediction - REAL MODEL ONLY
    try:
        input_img = preprocess(roi)
        if input_img is None:
            return "PREPROCESS ERROR", 0.0, (0, 0, 255), bbox, source
        
        # Get prediction from model
        prediction = model.predict(input_img, verbose=0)
        prob = float(prediction[0][0])
        
        # Interpret prediction
        # Assuming model outputs: 1 = Engaged, 0 = Not Engaged
        label = "Engaged" if prob >= 0.5 else "Not Engaged"
        confidence = prob if prob >= 0.5 else 1 - prob
        color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
        
        return label, confidence, color, bbox, source
        
    except Exception as e:
        print(f"❌ ML Prediction error: {e}")
        return "PREDICTION ERROR", 0.0, (0, 0, 255), bbox, source

# ============================================================================
# MOCK CAMERA GENERATION (For Render - Uses REAL ML Model)
# ============================================================================
def generate_mock_frames():
    """
    Generate synthetic video frames when no real camera is available
    BUT STILL USES REAL ML MODEL for predictions
    """
    global monitoring_active, frame_count
    
    if not MODEL_LOADED:
        # Generate error frame if model not loaded
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ML MODEL NOT LOADED", (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(error_frame, "Please upload attention_model.h5", (80, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(error_frame, "to models/ directory", (120, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        
        while monitoring_active:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
        return
    
    print("📹 Starting MOCK camera with REAL ML MODEL predictions...")
    frame_count = 0
    start_time = time.time()
    
    # Animation variables
    angle = 0
    color_shift = 0
    
    # Initialize face detector if available
    face_detector = None
    if MEDIAPIPE_AVAILABLE:
        try:
            face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
        except:
            pass
    
    while True:
        with monitoring_lock:
            if not monitoring_active:
                break
        
        try:
            # Create realistic synthetic frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Animated gradient background (more realistic)
            for i in range(480):
                color_val = int(120 + 40 * np.sin(i / 60 + color_shift))
                frame[i, :] = [color_val - 20, color_val, color_val + 20]
            
            # Draw realistic "student" face
            center_x = 320 + int(40 * np.sin(angle * 0.3))
            center_y = 240 + int(25 * np.cos(angle * 0.5))
            
            # Face oval (skin tone)
            cv2.ellipse(frame, (center_x, center_y), (90, 110), 0, 0, 360, (180, 150, 120), -1)
            
            # Hair
            cv2.ellipse(frame, (center_x, center_y - 50), (95, 70), 0, 0, 180, (40, 30, 20), -1)
            
            # Eyes (changes based on time - simulating attention)
            time_mod = int((time.time() - start_time) % 10)
            if time_mod < 6:  # Concentrated - eyes open
                eye1_x, eye2_x = center_x - 30, center_x + 30
                eye_y = center_y - 20
                # Eye whites
                cv2.ellipse(frame, (eye1_x, eye_y), (12, 15), 0, 0, 360, (255, 255, 255), -1)
                cv2.ellipse(frame, (eye2_x, eye_y), (12, 15), 0, 0, 360, (255, 255, 255), -1)
                # Pupils
                cv2.circle(frame, (eye1_x, eye_y), 6, (50, 30, 10), -1)
                cv2.circle(frame, (eye2_x, eye_y), 6, (50, 30, 10), -1)
            else:  # Distracted - looking away
                eye1_x, eye2_x = center_x - 30, center_x + 30
                eye_y = center_y - 20
                # Eye whites
                cv2.ellipse(frame, (eye1_x, eye_y), (12, 15), 0, 0, 360, (255, 255, 255), -1)
                cv2.ellipse(frame, (eye2_x, eye_y), (12, 15), 0, 0, 360, (255, 255, 255), -1)
                # Pupils looking away
                cv2.circle(frame, (eye1_x + 4, eye_y + 3), 6, (50, 30, 10), -1)
                cv2.circle(frame, (eye2_x + 4, eye_y + 3), 6, (50, 30, 10), -1)
            
            # Eyebrows
            cv2.ellipse(frame, (eye1_x, center_y - 35), (15, 8), 0, 0, 180, (40, 30, 20), 2)
            cv2.ellipse(frame, (eye2_x, center_y - 35), (15, 8), 0, 0, 180, (40, 30, 20), 2)
            
            # Nose
            cv2.ellipse(frame, (center_x, center_y + 10), (8, 15), 0, 0, 360, (160, 130, 100), -1)
            
            # Mouth
            if time_mod < 6:  # Concentrated - neutral/slight smile
                cv2.ellipse(frame, (center_x, center_y + 40), (25, 12), 0, 0, 180, (100, 70, 70), 2)
            else:  # Distracted - different expression
                cv2.ellipse(frame, (center_x, center_y + 45), (20, 8), 0, 180, 360, (100, 70, 70), 2)
            
            # Get REAL ML MODEL prediction on this synthetic frame
            label, confidence, color, bbox, source = predict_engagement(frame, face_detector)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Draw overlay background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            
            # Draw ML prediction results
            cv2.putText(frame, f"Status: {label}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {frame_count}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Source: {source}", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            # ML Model indicator
            cv2.putText(frame, "ML MODEL ACTIVE", (420, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Camera type indicator
            cv2.putText(frame, "MOCK CAMERA", (420, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            cv2.putText(frame, "(Real ML Predictions)", (400, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
            
            # Timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (20, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Platform info
            platform = "Render/Cloud" if IS_PRODUCTION else "Local"
            cv2.putText(frame, f"Platform: {platform}", (450, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Encode frame
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if not ret:
                print("❌ Failed to encode frame")
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield MJPEG frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"📊 Processed {frame_count} frames | FPS: {fps:.1f} | Last prediction: {label} ({confidence*100:.1f}%)")
            
            # Update animation
            angle += 0.04
            color_shift += 0.015
            
            # Control frame rate (~30 FPS)
            time.sleep(0.033)
            
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
            time.sleep(0.1)
    
    if face_detector:
        try:
            face_detector.close()
        except:
            pass
    
    print(f"🏁 Mock camera stopped. Total frames: {frame_count}")

# ============================================================================
# REAL CAMERA GENERATION (For Local - Uses REAL ML Model)
# ============================================================================
def generate_real_frames():
    """Generate frames from real camera with REAL ML predictions"""
    global monitoring_active, frame_count
    
    if not MODEL_LOADED:
        # Generate error frame if model not loaded
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ML MODEL NOT LOADED", (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(error_frame, "Cannot process video without model", (60, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        
        while monitoring_active:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
        return
    
    print("📹 Starting REAL camera with ML MODEL predictions...")
    
    # Initialize face detector
    face_detector = None
    if MEDIAPIPE_AVAILABLE:
        try:
            face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
            print("✅ Face detector initialized")
        except Exception as e:
            print(f"⚠️ Face detector init failed: {e}")
    
    cam = get_camera()
    
    if cam is None:
        print("❌ Camera not available")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(error_frame, "Cannot access camera", (140, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
        
        try:
            success, frame = cam.read()
            if not success or frame is None:
                print("⚠️ Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Get REAL ML MODEL prediction
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
            
            # Draw overlay background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            
            # Draw ML prediction results
            cv2.putText(frame, f"Status: {label}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f} | {source}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Frame: {frame_count}", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            # ML Model indicator
            cv2.putText(frame, "ML MODEL ACTIVE", (420, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Camera type indicator
            cv2.putText(frame, "REAL CAMERA", (420, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (20, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Encode frame
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if not ret:
                print("❌ Failed to encode frame")
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield MJPEG frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')
            
            if frame_count % 100 == 0:
                print(f"📊 Processed {frame_count} frames | FPS: {fps:.1f} | Last: {label} ({confidence*100:.1f}%)")
        
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            time.sleep(0.1)
    
    if face_detector:
        try:
            face_detector.close()
        except:
            pass
    
    print(f"🏁 Real camera stopped. Total frames: {frame_count}")

# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def index():
    """Health check and API info"""
    return jsonify({
        "message": "Student Concentration Detection API",
        "status": "active",
        "environment": "production" if IS_PRODUCTION else "local",
        "model_loaded": MODEL_LOADED,
        "model_path": MODEL_PATH if MODEL_LOADED else None,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "camera_available": CAMERA_AVAILABLE,
        "monitoring": monitoring_active,
        "mode": "ML Model" if MODEL_LOADED else "ERROR - No Model",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/start_monitoring', methods=['POST', 'GET'])
def start_monitoring():
    """Start monitoring session"""
    global monitoring_active
    
    if not MODEL_LOADED:
        return jsonify({
            "status": "error",
            "message": "ML Model not loaded. Cannot start monitoring.",
            "details": f"Expected model at: {MODEL_PATH}",
            "monitoring": False
        }), 500
    
    with monitoring_lock:
        if monitoring_active:
            return jsonify({
                "status": "already_active",
                "message": "Monitoring is already active",
                "monitoring": True
            }), 200
        
        monitoring_active = True
    
    # Test camera if available
    camera_status = "mock"
    if CAMERA_AVAILABLE:
        cam = get_camera()
        if cam is not None:
            camera_status = "real"
        else:
            camera_status = "mock (camera unavailable)"
    
    print(f"✅ Monitoring started - Camera: {camera_status}, Model: ML")
    
    return jsonify({
        "status": "success",
        "message": f"Monitoring started with ML model ({camera_status} camera)",
        "monitoring": True,
        "model_loaded": MODEL_LOADED,
        "camera_type": camera_status,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/stop_monitoring', methods=['POST', 'GET'])
def stop_monitoring():
    """Stop monitoring session"""
    global monitoring_active, frame_count
    
    with monitoring_lock:
        monitoring_active = False
    
    total_frames = frame_count
    frame_count = 0
    
    time.sleep(0.5)
    release_camera()
    
    print(f"✅ Monitoring stopped - Total frames processed: {total_frames}")
    
    return jsonify({
        "status": "success",
        "message": "Monitoring stopped successfully",
        "monitoring": False,
        "total_frames": total_frames,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/monitoring_status')
def monitoring_status():
    """Get current monitoring status"""
    return jsonify({
        "monitoring": monitoring_active,
        "model_loaded": MODEL_LOADED,
        "camera_available": CAMERA_AVAILABLE,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "mode": "ML Model" if MODEL_LOADED else "ERROR",
        "frames_processed": frame_count,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route with REAL ML MODEL predictions
    """
    global monitoring_active
    
    if not MODEL_LOADED:
        return jsonify({
            "status": "error",
            "message": "ML Model not loaded. Cannot generate video feed.",
            "details": f"Place model at: {MODEL_PATH}"
        }), 500
    
    if not monitoring_active:
        print("⚠️ Video feed requested but monitoring not started")
        return jsonify({
            "status": "error",
            "message": "Monitoring not started. Call /start_monitoring first"
        }), 400
    
    camera_type = "REAL" if CAMERA_AVAILABLE else "MOCK"
    print(f"🎥 Video feed requested - Camera: {camera_type}, Model: ML")
    
    try:
        if CAMERA_AVAILABLE:
            generator = generate_real_frames()
        else:
            generator = generate_mock_frames()
        
        return Response(
            generator,
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Connection': 'keep-alive'
            }
        )
    except Exception as e:
        print(f"❌ Video feed error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Video feed failed: {str(e)}"
        }), 500

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Single frame prediction endpoint using REAL ML MODEL"""
    if not MODEL_LOADED:
        return jsonify({
            "error": "ML Model not loaded",
            "details": f"Expected model at: {MODEL_PATH}"
        }), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    try:
        # Read and decode image
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Initialize face detector if available
        face_detector = None
        if MEDIAPIPE_AVAILABLE:
            try:
                face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
            except:
                pass
        
        # Get REAL ML MODEL prediction
        label, confidence, _, _, source = predict_engagement(img, face_detector)
        
        if face_detector:
            try:
                face_detector.close()
            except:
                pass
        
        return jsonify({
            "status": label,
            "confidence": float(confidence),
            "source": source,
            "model_loaded": MODEL_LOADED,
            "model_type": "ML Model (Real)",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"❌ Frame prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "degraded",
        "monitoring": monitoring_active,
        "model_loaded": MODEL_LOADED,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "camera_available": CAMERA_AVAILABLE,
        "environment": "production" if IS_PRODUCTION else "local",
        "frames_processed": frame_count,
        "timestamp": datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": [
            "/", "/start_monitoring", "/stop_monitoring",
            "/video_feed", "/monitoring_status", "/predict_frame", "/health"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "details": str(e)
    }), 500

# ============================================================================
# STARTUP
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Student Concentration Detection Server")
    print("=" * 70)
    
    # Detect environment
    if IS_PRODUCTION:
        print(f"🌐 Environment: PRODUCTION (Render/Railway)")
    else:
        print(f"🏠 Environment: LOCAL")
    
    # Model status - CRITICAL
    print("")
    if MODEL_LOADED:
        print(f"✅ ML MODEL: LOADED SUCCESSFULLY")
        print(f"📂 Model Path: {MODEL_PATH}")
        print(f"🧠 Using REAL ML predictions")
    else:
        print(f"❌ ML MODEL: NOT LOADED")
        print(f"❌ Expected at: {MODEL_PATH}")
        print(f"❌ Server will NOT work without model!")
        print(f"")
        print(f"🔧 TO FIX:")
        print(f"   1. Create 'models' folder in project root")
        print(f"   2. Place 'attention_model.h5' in models/")
        print(f"   3. Redeploy application")
        print(f"")
    
    # Detect camera
    print("")
    detect_camera_availability()
    if CAMERA_AVAILABLE:
        print(f"📷 Camera: REAL CAMERA DETECTED ✅")
        print(f"   Will use real camera for predictions")
    else:
        print(f"📷 Camera: MOCK CAMERA (No physical camera)")
        print(f"   Will generate synthetic frames with ML predictions")
    
    # MediaPipe status
    print("")
    if MEDIAPIPE_AVAILABLE:
        print(f"👤 MediaPipe: AVAILABLE ✅")
        print(f"   Face detection enabled for better accuracy")
    else:
        print(f"👤 MediaPipe: NOT AVAILABLE")
        print(f"   Will use full frame for predictions")
    
    print("")
    print("=" * 70)
    port = int(os.environ.get('PORT', 8080))
    print(f"🌐 Server starting on: 0.0.0.0:{port}")
    print("")
    print("Available Endpoints:")
    print("  GET  /                 - API info & health check")
    print("  POST /start_monitoring - Start monitoring (requires ML model)")
    print("  POST /stop_monitoring  - Stop monitoring session")
    print("  GET  /monitoring_status - Check monitoring status")
    print("  GET  /video_feed       - Video stream with ML predictions")
    print("  POST /predict_frame    - Single frame ML prediction")
    print("  GET  /health           - Detailed health check")
    print("=" * 70)
    
    if not MODEL_LOADED:
        print("")
        print("⚠️  WARNING: Starting without ML model - limited functionality")
        print("")
    
    print("")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )