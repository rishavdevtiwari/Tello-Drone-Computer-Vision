# MediaPipe Tasks API Orbit Tracker for DJI Tello Drone

import cv2
import time
import urllib.request
import os
from djitellopy import Tello
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class TelloPersonTracker:
    def __init__(self):
        # 1. Offline Model Download Handler
        self.model_path = 'pose_landmarker_lite.task'
        if not os.path.exists(self.model_path):
            print(f"Model missing. Attempting to download {self.model_path}...")
            try:
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                urllib.request.urlretrieve(url, self.model_path)
                print("Download complete!")
            except Exception as e:
                print("\n" + "="*60)
                print("[NETWORK ERROR] Could not download the MediaPipe model.")
                print("-> FIX: Disconnect from the Tello Wi-Fi, connect to normal")
                print("   internet Wi-Fi, run this script once to download the file,")
                print("   then reconnect to the Tello Wi-Fi and fly.")
                print("="*60 + "\n")
                exit()

        # 2. Initialize Drone
        self.tello = Tello()
        
        # 3. Initialize MediaPipe Tasks Pose Model
        print("Loading MediaPipe Tasks Pose model...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        # 4. Tracking State Variables
        self.target_locked = False
        self.first_seen_time = None
        self.required_recognition_time = 2.0  
        self.last_known_x = 0  
        
        # 5. Flight Tuning Parameters
        # Increased p_yaw slightly so it turns fast enough to hold the orbit
        self.p_yaw = 0.4      
        self.p_pitch = 0.3    
        self.p_throttle = 0.4 
        self.orbit_speed = 35 # Speed of the circular strafe maneuver
        
        self.target_area = 80000 
        self.area_range = [70000, 90000] 

    def connect_and_takeoff(self):
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")
        
        if self.tello.get_battery() < 15:
            print("Battery too low for safe flight. Exiting.")
            exit()

        self.tello.streamon()
        self.tello.takeoff()
        self.tello.move_up(80) 

    def get_person_bounding_box(self, result, w, h):
        """Calculates a pseudo-bounding box from the Tasks API landmarks."""
        if not result.pose_landmarks:
            return None, 0
        
        landmarks = result.pose_landmarks[0]
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x1 = int(min(x_coords) * w)
        y1 = int(min(y_coords) * h)
        x2 = int(max(x_coords) * w)
        y2 = int(max(y_coords) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        
        return (x1, y1, bw, bh), area

    def track(self):
        print("Starting tracking loop. Press 'q' in the video window to quit & land.")
        
        try:
            while True:
                # 1. Get Frame
                frame_read = self.tello.get_frame_read()
                frame = frame_read.frame
                frame = cv2.resize(frame, (640, 480))
                h, w, _ = frame.shape
                center_x, center_y = w // 2, h // 2

                # 2. Run Inference
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)
                
                results = self.detector.detect_for_video(mp_image, timestamp_ms)
                
                # 3. Find our target
                person_box, area = self.get_person_bounding_box(results, w, h)

                left_right, forward_backward, up_down, yaw = 0, 0, 0, 0

                if person_box is not None:
                    # --- PERSON DETECTED ---
                    x1, y1, bw, bh = person_box
                    obj_cx = x1 + (bw // 2)
                    obj_cy = y1 + (bh // 2)
                    
                    self.last_known_x = obj_cx

                    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
                    cv2.circle(frame, (obj_cx, obj_cy), 5, (0, 0, 255), cv2.FILLED)

                    if self.first_seen_time is None:
                        self.first_seen_time = time.time()
                        cv2.putText(frame, "Recognizing...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    
                    elif time.time() - self.first_seen_time >= self.required_recognition_time:
                        self.target_locked = True
                        cv2.putText(frame, "TARGET LOCKED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Calculate center tracking 
                        error_x = obj_cx - center_x
                        error_y = obj_cy - center_y

                        yaw = int(error_x * self.p_yaw)
                        up_down = int(-error_y * self.p_throttle)

                        # Distance logic & Orbit trigger
                        if area > self.area_range[1]:      
                            forward_backward = -20  # Move back
                            left_right = 0          # Stop orbiting while adjusting distance
                        elif area < self.area_range[0]:    
                            forward_backward = 20   # Move forward
                            left_right = 0          # Stop orbiting while adjusting distance
                        else:
                            # Distance is perfect. INITIATE ORBIT!
                            left_right = self.orbit_speed 
                            cv2.putText(frame, "ORBITING...", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                else:
                    # --- PERSON LOST ---
                    self.target_locked = False
                    self.first_seen_time = None
                    cv2.putText(frame, "SEARCHING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if self.last_known_x < center_x:
                        yaw = -30 
                    elif self.last_known_x > center_x:
                        yaw = 30  
                    else:
                        yaw = 0   

                # 4. Clamp speeds to safe Tello limits
                yaw = max(-100, min(100, yaw))
                up_down = max(-100, min(100, up_down))
                forward_backward = max(-100, min(100, forward_backward))
                left_right = max(-100, min(100, left_right))

                # 5. Send Commands
                # Format: left_right, forward_backward, up_down, yaw
                self.tello.send_rc_control(left_right, forward_backward, up_down, yaw)

                # 6. Display Video
                cv2.imshow("Tello Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nManual interrupt received.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            print("Landing and cleaning up...")
            self.tello.send_rc_control(0, 0, 0, 0)
            self.tello.land()
            self.tello.streamoff()
            self.detector.close() 
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TelloPersonTracker()
    tracker.connect_and_takeoff()
    tracker.track()