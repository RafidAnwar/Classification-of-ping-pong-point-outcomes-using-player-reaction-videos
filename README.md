# Classification of Ping-Pong Point Outcomes Using Player Reaction Videos

---
## Usage Instructions

- **Python Requirements**  
  - Python 3.10+  
  - Essential libraries: see `requirements.txt` for all dependencies 

- **Model Requirements**
  - YOLOv11 pose model: `yolo11m-pose.pt` (download from [Ultralytics lab](https://docs.ultralytics.com/tasks/pose/))
  - Emo-AffectNet pretrained model by Ryumina et al.: [https://github.com/ElenaRyumina/EMO-AffectNetModel](https://github.com/ElenaRyumina/EMO-AffectNetModel)
  - Dataset: 600+ annotated table tennis reaction video clips can be collected from KISMED research group.
    
- **Execution Order**  
  - Run scripts sequentially for complete pipeline execution.
---

## Pipeline Structure

### 1. video annotation.py
**YOLOv11 Pose Annotation**: Detects & visualizes **17 body keypoints** on table tennis reaction videos.

### 2. landmark detection check.py
**Quality Filtering**: Retains videos with **≥80% keypoint detection** across all 17 landmarks.

### 3. fea_extract.py
**Feature Extraction**: 
- **Pose**: 68-dim features including the keypoints relative position and distances, angles, velocities and acceleration.
- **Face**: **512-dim Emo-AffectNet** emotion vectors

### 4. lstm.py
classification using LSTM based model architecture

### 5. xlstm.py
classification using xLSTM based model architecture

### 6. transformer.py
Transformer model with multihead self-attention for classification.
---

