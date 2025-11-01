import cv2
import time
import threading
import numpy as np
import tensorflow as tf
import mediapipe as mp
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="models/attention_model.h5")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--cam", type=int, default=0)
parser.add_argument("--min_detection_confidence", type=float, default=0.4)
args = parser.parse_args()

MODEL_PATH = args.model
IMG_SIZE = args.img_size
CAM_ID = args.cam

class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")
        self.ret, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            ret, frame = self.cap.read()
            with self.read_lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            ret = self.ret
        return ret, frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found at", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=args.min_detection_confidence)

    cap = VideoCaptureAsync(CAM_ID).start()
    time.sleep(0.5)

    fps = 0.0
    prev = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(img_rgb)

            # default: predict on full frame if no face found
            roi = frame
            label_source = "Full Frame"

            if results.detections:
                # use first detected face
                d = results.detections[0].location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x1 = int(d.xmin * w)
                y1 = int(d.ymin * h)
                bw = int(d.width * w)
                bh = int(d.height * h)
                # expand bbox slightly
                pad_w = int(0.2 * bw)
                pad_h = int(0.5 * bh)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(w, x1 + bw + 2*pad_w)
                y2 = min(h, y1 + bh + 2*pad_h)
                roi = frame[y1:y2, x1:x2]
                label_source = "Face Crop"

            # preprocess and predict
            try:
                input_img = preprocess(roi)
                preds = model.predict(input_img, verbose=0)
                prob = float(preds[0][0])
            except Exception as e:
                prob = 0.5

            label = "Engaged" if prob >= 0.5 else "Not Engaged"
            confidence = prob if prob >= 0.5 else 1 - prob

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev)) if now!=prev else fps
            prev = now

            # overlay
            text = f"{label} ({confidence*100:.1f}%) | {label_source} | FPS: {fps:.1f}"
            color = (0,255,0) if label=="Engaged" else (0,0,255)
            # draw background rectangle for text
            cv2.rectangle(frame, (5,5), (480,40), (0,0,0), -1)
            cv2.putText(frame, text, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # if face crop, draw bbox
            if results.detections:
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            cv2.imshow("Engagement Detection (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        face_detector.close()

if __name__ == "__main__":
    main()