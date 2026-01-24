from ultralytics import YOLO
import cv2
import numpy as np

pose_model = YOLO('yolo11m-pose.pt')
face_model = YOLO('yolov12n-face.pt')

# Input and output paths
input_video_path = 'G_Mt_Wcq_9_8.mp4'
output_video_path = 'output_pose_face_combined_2.mp4'

# Open video capture and get properties
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

face_keypoint_indices = [0, 1, 2, 3, 4]

skeleton_connections = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # legs and torso
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO11 Pose inference
    pose_results = pose_model(frame, verbose=False)

    # Run YOLOv8 Face Detection inference
    face_results = face_model(frame, verbose=False)

    annotated_frame = frame.copy()

    if len(pose_results) > 0 and len(pose_results[0].keypoints) > 0:
        keypoints = pose_results[0].keypoints.xy.cpu().numpy()
        if len(keypoints) > 0:
            first_person_kpts = keypoints[0]

            for idx, (x, y) in enumerate(first_person_kpts):
                if idx not in face_keypoint_indices and x > 0 and y > 0:
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Draw skeleton connections (excluding any connection involving face keypoints)
            for start_idx, end_idx in skeleton_connections:
                if start_idx not in face_keypoint_indices and end_idx not in face_keypoint_indices:
                    x1, y1 = first_person_kpts[start_idx]
                    x2, y2 = first_person_kpts[end_idx]
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Draw face rectangles
    if len(face_results) > 0:
        for result in face_results:
            for box in result.boxes:
                if box.cls == 0:  # Assuming class 0 is face
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print('No face detected')

    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
