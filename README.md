# Student Attention — Realtime (with Mediapipe face detection)

This project contains a ready-to-run pipeline for real-time student engagement detection using your uploaded dataset.
Dataset (unzipped) was placed under `dataset/` from your uploaded archive.

## Structure
```
Student_Attention_Realtime/
├── dataset/    # your extracted dataset (train/ test or Engaged/Not Engaged folders)
├── models/     # saved models (training will save here)
├── train.py
├── realtime_infer.py
├── requirements.txt
└── README.md
```

## Quick setup (Windows / Mac / Linux)
1. Create & activate a virtual environment:
   - Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - Mac/Linux:
     ```
     python -m venv venv
     source venv/bin/activate
     ```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Train (example):
```
python train.py --data_dir "dataset" --epochs 12 --batch_size 32 --img_size 224 --save_path "models/attention_model.h5"
```
If your dataset has `train/` and `test/` folders, the script will use them. If your dataset directly contains class folders (e.g., `Engaged`, `Not Engaged`) the script will also work.

4. Run real-time inference:
```
python realtime_infer.py --model "models/attention_model.h5" --img_size 224
```
This opens your webcam and uses Mediapipe to detect face and crop before prediction (falls back to full frame if no face found).
Press `q` to quit.

## Notes & tips
- If you have a GPU and want faster training, install GPU TensorFlow per your system's CUDA/cuDNN.
- Lighting variety in training data helps robustness.
- You can change `--min_detection_confidence` for Mediapipe (default 0.4) to tune detection sensitivity.