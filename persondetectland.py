import cv2
import sys
import time
import os
import urllib.request
from djitellopy import Tello
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. MODEL CHECK (Ensures you have the file before trying to fly)
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Model missing. Downloading...")
    try:
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete!")
    except:
        print("ERROR: Could not download. Disconnect from Tello, connect to normal Wi-Fi, run once, then reconnect to Tello.")
        sys.exit(1)

# 2. INITIALIZE MEDIAPIPE
print("Loading Model...")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# 3. INITIALIZE DRONE
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

if tello.get_battery() < 10:
    print("Battery too low. Exiting.")
    sys.exit(1)

tello.streamon()
tello.takeoff()
tello.move_up(50) # Move to chest height

print("Drone is up. Searching for a person...")

try:
    while True:
        # Get frame
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (640, 480))
        
        # Convert to MediaPipe format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        
        # Run Detection
        results = detector.detect_for_video(mp_image, timestamp_ms)
        
        # Check if a person is found
        if results.pose_landmarks:
            cv2.putText(frame, "PERSON DETECTED! LANDING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Tello Camera", frame)
            cv2.waitKey(1)
            
            print("Person detected! Initiating landing sequence.")
            break # Break the loop to trigger landing
            
        else:
            # If no person, slowly spin to look around
            cv2.putText(frame, "SEARCHING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            tello.send_rc_control(0, 0, 0, 30) # 30 = spin speed
            
        # Display Video
        cv2.imshow("Tello Camera", frame)
        
        # Manual Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual quit triggered.")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 4. LAND AND CLEANUP (Always runs, even if it crashes)
    print("Landing...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    detector.close()
    cv2.destroyAllWindows()