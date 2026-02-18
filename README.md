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
Detects & visualizes **17 body keypoints** on table tennis reaction videos using YOLO11 pose estimation model.

### 2. landmark detection check.py
Retains videos with **≥80% keypoint detection** across all 17 landmarks for all of the 600+ annotated videos.

### 3. fea_extract.py
extraction of 68-dim features from the 17 keypoints of pose landmarks (keypoints relative position, distances, angles, velocities and acceleration) and 512 dimesnion feature vector from the emo-affectnet model for the facial features

### 4. lstm.py
classification using LSTM based model architecture

### 5. xlstm.py
classification using xLSTM based model architecture

### 6. transformer.py
Transformer model with multihead self-attention for classification.

### 7.creating_database.py
Creating sql database of different model and feature based data for analysis.

### 8.power_BI_csv.py
extracting important CSV files from the database to be used in power BI for detailed reporting.


