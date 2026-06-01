import cv2
import time
import math
import os
import urllib.request
from djitellopy import Tello
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def calculate_angle(a, b, c):
    """Calculates the angle between three points (x, y)."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# 1. Model Download Handler (For Python 3.13)
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)

# 2. Initialize MediaPipe
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.6
)
detector = vision.PoseLandmarker.create_from_options(options)

# 3. Initialize Drone
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()
tello.takeoff()
tello.move_up(80) 

last_action_time = time.time()
cooldown = 4.0 # Seconds to wait after an action before detecting another

try:
    while True:
        frame_read = tello.get_frame_read()
        frame = frame_read.frame
        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape

        # Default hover
        tello.send_rc_control(0, 0, 0, 0)

        # Process Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect_for_video(mp_image, int(time.time() * 1000))

        if results.pose_landmarks and (time.time() - last_action_time > cooldown):
            landmarks = results.pose_landmarks[0]
            
            # Extract relevant points (Left and Right: Shoulder, Hip, Wrist)
            # MediaPipe landmarks: 11=L_Shoulder, 12=R_Shoulder, 15=L_Wrist, 16=R_Wrist, 23=L_Hip, 24=R_Hip
            l_sh = [landmarks[11].x * w, landmarks[11].y * h]
            r_sh = [landmarks[12].x * w, landmarks[12].y * h]
            l_wr = [landmarks[15].x * w, landmarks[15].y * h]
            r_wr = [landmarks[16].x * w, landmarks[16].y * h]
            l_hp = [landmarks[23].x * w, landmarks[23].y * h]
            r_hp = [landmarks[24].x * w, landmarks[24].y * h]

            # Calculate angles at the shoulder (Hip -> Shoulder -> Wrist)
            angle_left = calculate_angle(l_hp, l_sh, l_wr)
            angle_right = calculate_angle(r_hp, r_sh, r_wr)

            # Draw visual skeleton lines for debugging
            cv2.line(frame, (int(l_hp[0]), int(l_hp[1])), (int(l_sh[0]), int(l_sh[1])), (255, 0, 0), 2)
            cv2.line(frame, (int(l_sh[0]), int(l_sh[1])), (int(l_wr[0]), int(l_wr[1])), (255, 0, 0), 2)
            cv2.line(frame, (int(r_hp[0]), int(r_hp[1])), (int(r_sh[0]), int(r_sh[1])), (0, 0, 255), 2)
            cv2.line(frame, (int(r_sh[0]), int(r_sh[1])), (int(r_wr[0]), int(r_wr[1])), (0, 0, 255), 2)

            cv2.putText(frame, f"L: {int(angle_left)} R: {int(angle_right)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # GESTURE LOGIC
            is_left_up = angle_left > 140
            is_right_up = angle_right > 140
            is_left_out = 70 < angle_left < 110
            is_right_out = 70 < angle_right < 110

            # 1. Both Hands Raised -> Land
            if is_left_up and is_right_up:
                print("Gesture: BOTH HANDS UP -> Landing")
                cv2.putText(frame, "LANDING...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("Tello Feed", frame)
                cv2.waitKey(1)
                break 

            # 2. Arms at 180 degrees (T-Pose) -> 360 Turn
            elif is_left_out and is_right_out:
                print("Gesture: 180 DEGREE ARMS -> 360 Turn")
                cv2.putText(frame, "360 TURN!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                cv2.imshow("Tello Feed", frame)
                cv2.waitKey(1)
                tello.rotate_clockwise(360)
                last_action_time = time.time()

            # 3. Arms at 90 degrees (L-Shape: One up, One out) -> Flip
            elif (is_left_up and is_right_out) or (is_left_out and is_right_up):
                print("Gesture: 90 DEGREE ARMS -> Flip")
                cv2.putText(frame, "FLIP!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
                cv2.imshow("Tello Feed", frame)
                cv2.waitKey(1)
                tello.flip_back()
                last_action_time = time.time()

        cv2.imshow("Tello Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    print("Cleaning up and landing...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    detector.close()
    cv2.destroyAllWindows()