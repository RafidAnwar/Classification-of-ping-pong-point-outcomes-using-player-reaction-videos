import cv2
import mediapipe as mp
import os
import glob
from enum import IntEnum

# --- Model Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


VIDEO_DIR = './videos'
MISSING_PERCENTAGE_THRESHOLD = 40.0


REQUIRED_LANDMARK_INDICES = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
]

VISIBILITY_THRESH = 0.2

def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 0

    total_frames = 0
    missing_data_frames = 0

    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=2,
                      enable_segmentation=False,
                      min_detection_confidence=0.2,
                      min_tracking_confidence=0.2) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(image_rgb)

            is_pose_missing = True

            # Check for missing POSE Landmarks (Shoulder, Hip, Hand)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                found_all_required = True

                for idx in REQUIRED_LANDMARK_INDICES:
                    lm = landmarks[idx]
                    if lm.visibility < VISIBILITY_THRESH:
                        found_all_required = False
                        break

                if found_all_required:
                    is_pose_missing = False

            # Count if POSE is missing
            if is_pose_missing:
                missing_data_frames += 1

    cap.release()

    # Calculate the percentage
    if total_frames > 0:
        missing_percentage = (missing_data_frames / total_frames) * 100
    else:
        missing_percentage = 0

    return missing_data_frames, total_frames, missing_percentage

def main():
    print("--- MediaPipe Pose-Only Batch Analysis Started ---")
    print(f"Scanning directory: **{VIDEO_DIR}**")
    print(f"Reporting videos with missing pose percentage > **{MISSING_PERCENTAGE_THRESHOLD:.1f}%**\n")

    video_files = glob.glob(os.path.join(VIDEO_DIR, '*.mp4')) + \
                  glob.glob(os.path.join(VIDEO_DIR, '*.avi')) + \
                  glob.glob(os.path.join(VIDEO_DIR, '*.mov'))


    total_videos_found = len(video_files)

    if not video_files:
        print(f"No video files found in the directory: {VIDEO_DIR}")
        return

    high_missing_videos = []

    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        print(f"Processing: {video_filename}...")


        missing_frames, total_frames, missing_percentage = analyze_video(video_path)

        if total_frames > 0 and missing_percentage > MISSING_PERCENTAGE_THRESHOLD:
            high_missing_videos.append((video_filename, missing_frames, total_frames, missing_percentage))

    videos_reported = len(high_missing_videos)
    videos_excluded = total_videos_found - videos_reported

    ## --- Print Filtered Results ---
    print("\n" + "=" * 70)
    print("📊 Analysis Summary")
    print("=" * 70)

    if high_missing_videos:
        high_missing_videos.sort(key=lambda x: x[3], reverse=True)

        for name, missing, total, percentage in high_missing_videos:
            print(f"**{name}**")
            print(f"  Frames Missing Pose Data: {missing} / {total}")
            print(f"  Missing Percentage: **{percentage:.2f}%**")
            print("-" * 30)
    else:
        print("No videos exceeded the missing frame percentage threshold.")


    print(f"FINAL VIDEO FILTER REPORT")
    print(f"Total videos scanned:{total_videos_found}")
    print(f"Videos reported:{videos_reported}")


if __name__ == "__main__":
    main()