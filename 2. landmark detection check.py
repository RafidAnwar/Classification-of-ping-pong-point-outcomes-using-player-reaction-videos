from ultralytics import YOLO
import cv2
import os
import glob
import math

try:
    pose_model = YOLO('yolo11m-pose.pt')
except Exception as e:
    print(f"Error initializing models: {e}")
    print("Please ensure 'yolo11m-pose.pt' file is available.")
    exit()

VIDEO_DIR = './videos'
MISSING_PERCENTAGE_THRESHOLD = 30.0

REQUIRED_POSE_KEYPOINT_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return 0, 0, 0

    total_frames = 0
    missing_data_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        pose_results = pose_model(frame, verbose=False, max_det=1)

        is_pose_missing = True

        if len(pose_results) > 0 and len(pose_results[0].keypoints) > 0:
            keypoints_xy = pose_results[0].keypoints.xy.cpu().numpy()

            if len(keypoints_xy) > 0:
                first_person_kpts = keypoints_xy[0]

                found_all_required_kpts = True
                for idx in REQUIRED_POSE_KEYPOINT_INDICES:
                    if idx >= len(first_person_kpts) or first_person_kpts[idx][0] == 0.0 or first_person_kpts[idx][
                        1] == 0.0:
                        found_all_required_kpts = False
                        break

                if found_all_required_kpts:
                    is_pose_missing = False

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
    print("--- YOLO Pose-Only Batch Analysis Started ---")
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

        # Analyze the video
        missing_frames, total_frames, missing_percentage = analyze_video(video_path)

        if total_frames > 0 and missing_percentage > MISSING_PERCENTAGE_THRESHOLD:
            high_missing_videos.append((video_filename, missing_frames, total_frames, missing_percentage))

    # Calculate the final counts
    videos_reported = len(high_missing_videos)
    videos_excluded = total_videos_found - videos_reported


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
    print(f"Total videos scanned: {total_videos_found}")
    print(f"Videos reported:{videos_reported}")

if __name__ == "__main__":
    main()