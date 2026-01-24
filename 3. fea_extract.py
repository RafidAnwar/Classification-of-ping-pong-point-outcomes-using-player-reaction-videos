##
import cv2
import numpy as np
import glob
import os
import scipy.io
# Required for face feature extraction
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import img_to_array


INPUT_VIDEO_DIRECTORY = './dataset/'
OUTPUT_MAT_FILE = 'trainset.mat'
# Pre-trained models
POSE_MODEL_PATH = 'yolo11m-pose.pt'
FACE_MODEL_PATH = 'yolov12n-face.pt'
EMO_AFFECTNET_MODEL_PATH = './emo_affectnet_model.h5'


LANDMARK_INDICES = list(range(5, 15))

feature_model = None

def pre_processing(img):
    """Preprocesses a face image for the Emo-AffectNet model."""
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)

    # Mean subtraction (channel-wise)
    img[..., 0] -= 91.4953  # Blue
    img[..., 1] -= 103.8827  # Green
    img[..., 2] -= 131.0912  # Red

    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cosine_angle = dot_product / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cosine_angle))


def process_video_and_extract_features(video_path, pose_model, face_model, feature_model_keras):

    print(f"Processing: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)

    kpt_seq = []
    face_features_seq = []

    # Define a placeholder for missing face features (1x512 vector of zeros)
    missing_face_feat = np.zeros((1, 512), dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pose_results = pose_model(frame, verbose=False)

        if not pose_results or not pose_results[0].keypoints.xy.shape[0]:
            continue

        landmarks = pose_results[0].keypoints.xy.cpu().numpy()
        kpt = landmarks[0]

        hip_center = (kpt[11] + kpt[12]) / 2.0
        shoulder_center = (kpt[5] + kpt[6]) / 2.0
        kpt_centered = kpt - hip_center
        torso_length = np.linalg.norm(shoulder_center - hip_center)

        if torso_length < 1e-6:
            kpt_norm = kpt_centered
        else:
            kpt_norm = kpt_centered / torso_length

        face_results = face_model(frame, verbose=False)
        current_face_feat = missing_face_feat

        face_detected = False
        if face_results and face_results[0].boxes.shape[0] > 0:
            # Only process the first detected face
            box = face_results[0].boxes[0]
            if box.cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    face = frame[y1:y2, x1:x2]
                    face = pre_processing(face)
                    feat = feature_model_keras.predict(face, verbose=0)
                    current_face_feat = feat
                    face_detected = True

        kpt_seq.append(kpt_norm)
        face_features_seq.append(current_face_feat)

    cap.release()

    if not kpt_seq:
        return None

    face_features = np.vstack(face_features_seq)

    all_coords = np.array(kpt_seq)
    frame_count = len(all_coords)

    # 1. RAW COORDINATES (Landmarks 5-14)
    coords_subset = all_coords[:, LANDMARK_INDICES, :]
    raw_features = coords_subset.reshape(frame_count, -1)

    # 2. RELATIVE POSITION & ANGLE FEATURES (28 features)
    relative_and_angle_features = []

    for frame_kpts in all_coords:
        ls, rs, le, re, lw, rw, lh, rh = frame_kpts[5:13]

        # Angle Features (4 features)
        angle_l_elbow = calculate_angle(ls, le, lw)
        angle_r_elbow = calculate_angle(rs, re, rw)
        angle_l_shoulder = calculate_angle(le, ls, lh)
        angle_r_shoulder = calculate_angle(re, rs, rh)
        angles = np.array([angle_l_elbow, angle_r_elbow, angle_l_shoulder, angle_r_shoulder])

        # Relative Position & Distance Features (24 features)
        rel_lw_ls, dist_lw_ls = lw - ls, np.linalg.norm(lw - ls)
        rel_rw_rs, dist_rw_rs = rw - rs, np.linalg.norm(rw - rs)
        rel_le_ls, dist_le_ls = le - ls, np.linalg.norm(le - ls)
        rel_re_rs, dist_re_rs = re - rs, np.linalg.norm(re - rs)
        rel_lw_lh, dist_lw_lh = lw - lh, np.linalg.norm(lw - lh)
        rel_rw_rh, dist_rw_rh = rw - rh, np.linalg.norm(rw - rh)
        rel_le_lh, dist_le_lh = le - lh, np.linalg.norm(le - lh)
        rel_re_rh, dist_re_rh = re - rh, np.linalg.norm(re - rh)

        frame_features = np.concatenate([
            angles,
            rel_lw_ls, [dist_lw_ls], rel_rw_rs, [dist_rw_rs],
            rel_le_ls, [dist_le_ls], rel_re_rs, [dist_re_rs],
            rel_lw_lh, [dist_lw_lh], rel_rw_rh, [dist_rw_rh],
            rel_le_lh, [dist_le_lh], rel_re_rh, [dist_re_rh],
        ])

        relative_and_angle_features.append(frame_features)

    relative_and_angle_features = np.array(relative_and_angle_features)

    # 3. VELOCITY & ACCELERATION FEATURES (20 features)
    vel = np.diff(coords_subset, axis=0)
    accel = np.diff(vel, axis=0)

    velocity_norm = np.linalg.norm(vel, axis=2)
    acceleration_norm = np.linalg.norm(accel, axis=2)

    velocity_padded = np.pad(velocity_norm, ((0, 1), (0, 0)), constant_values=0)
    acceleration_padded = np.pad(acceleration_norm, ((0, 2), (0, 0)), constant_values=0)

    velocity_acceleration_features = np.concatenate([velocity_padded, acceleration_padded], axis=1)

    # 4. COMBINE ALL FEATURES: 20 (Raw) + 28 (Relative/Dist/Angle) + 20 (Vel/Accel) + 512 (Face) = 580 features
    combined_features = np.concatenate([
        raw_features,
        relative_and_angle_features,
        velocity_acceleration_features,
        face_features
    ], axis=1)

    print(f"Extraction complete. Frames: {frame_count}, Total Feature Vector Size: {combined_features.shape[1]}")

    return {
        'video_name': os.path.basename(video_path),
        'features': combined_features
    }


# --- Main Execution ---

def main():
    # --- Check for Required Files ---
    required_files = {
        POSE_MODEL_PATH: "YOLO Pose Model",
        FACE_MODEL_PATH: "YOLO Face Model",
        EMO_AFFECTNET_MODEL_PATH: "Keras Feature Model"
    }


    pose_model = YOLO(POSE_MODEL_PATH)
    face_model = YOLO(FACE_MODEL_PATH)
    # Load the Keras model and select the feature extraction layer
    keras_model = load_model(EMO_AFFECTNET_MODEL_PATH)
    feature_model_keras = Model(inputs=keras_model.input, outputs=keras_model.get_layer('dense_4').output)

    # Find all video files in the specified directory
    video_files = glob.glob(os.path.join(INPUT_VIDEO_DIRECTORY, '*.mp4'))
    video_files.extend(glob.glob(os.path.join(INPUT_VIDEO_DIRECTORY, '*.avi')))

    all_features_list = []
    all_labels_list = []
    all_video_tags_list = []
    video_tag_counter = 0

    for video_path in video_files:
        # Extract features for each video
        result = process_video_and_extract_features(video_path, pose_model, face_model, feature_model_keras)

        if result:
            features = result['features']
            frame_count = features.shape[0]
            video_name = result['video_name']

            first_char = video_name[0].upper()
            label = -1
            if first_char == 'G':
                label = 0
            elif first_char == 'V':
                label = 1
            else:
                print(f"Warning: Unknown prefix for {video_name}. Skipping video.")
                continue

            labels_for_video = np.full((frame_count, 1), label, dtype=np.int8)
            tags_for_video = np.full((frame_count, 1), video_tag_counter, dtype=np.int32)

            all_features_list.append(features)
            all_labels_list.append(labels_for_video)
            all_video_tags_list.append(tags_for_video)

            video_tag_counter += 1

    if all_features_list:
        final_features_matrix = np.concatenate(all_features_list, axis=0)
        final_labels_vector = np.concatenate(all_labels_list, axis=0)
        final_video_tags_vector = np.concatenate(all_video_tags_list, axis=0)

        print("\n--- Final Data Structure ---")
        print(f"Total Frames Processed: {final_features_matrix.shape[0]}")
        print(f"Features Matrix Shape: {final_features_matrix.shape}")
        #print(f"Labels Vector Shape: {final_labels_vector.shape}, wins = {len(np.sum(final_labels_vector==0))}, loss = {len(np.sum(final_labels_vector==1))}")
        print(f"Video Tags Vector Shape: {final_video_tags_vector.shape}")

        scipy.io.savemat(OUTPUT_MAT_FILE, {
            'features_matrix': final_features_matrix,
            'labels_vector': final_labels_vector,
            'video_tags_vector': final_video_tags_vector
        })
        print(f"\nSuccessfully saved unified features to '{OUTPUT_MAT_FILE}'")
    else:
        print("No features were successfully extracted from any video. Output file was not created.")


if __name__ == '__main__':
    main()