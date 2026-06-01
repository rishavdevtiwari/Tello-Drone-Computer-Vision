# MediaPipe Tasks API Person Tracker & Two-Phase 360 Search for DJI Tello

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
# --- 1. MODEL DOWNLOAD CHECK ---
        # The new API requires a physical model file
        self.model_path = 'pose_landmarker_lite.task'
        if not os.path.exists(self.model_path):
            print(f"Model file missing. Attempting to download {self.model_path}...")
            try:
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                urllib.request.urlretrieve(url, self.model_path)
                print("Download complete!")
            except Exception as e:
                print("\n" + "="*50)
                print("[ERROR] Could not download the MediaPipe model.")
                print("Your computer likely has no internet connection because it is connected to the Tello drone.")
                print("-> FIX: Disconnect from Tello, connect to your normal Wi-Fi, run this script once to download the file, then reconnect to the Tello.")
                print("="*50 + "\n")
                exit()

        # 2. Initialize Drone
        self.tello = Tello()
        
        # 3. Initialize MediaPipe Tasks Pose Model
        print("Loading MediaPipe Tasks Pose model...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1, # Limit to tracking 1 person
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
        self.p_yaw = 0.3      
        self.p_pitch = 0.3    
        self.p_throttle = 0.4 
        self.target_area = 80000 
        self.area_range = [70000, 90000] 

        # 6. Autonomous Search Pattern Variables
        self.time_target_lost = None
        self.search_state = "ROTATE"   
        self.search_step_start = None  

    def connect_and_takeoff(self):
        """Connects to the drone, starts the stream, and takes off."""
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")
        
        if self.tello.get_battery() < 15:
            print("Battery too low for safe flight. Exiting.")
            exit()

        self.tello.streamon()
        self.tello.takeoff()
        self.tello.move_up(80) 

    def get_person_bounding_box(self, result, w, h):
        """Calculates a bounding box from the new Tasks API landmarks."""
        if not result.pose_landmarks:
            return None, 0
        
        # Extract the skeleton joints of the first person detected
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
        """Main tracking & autonomous flight loop."""
        print("Starting loop. Press 'q' in the video window to quit & land.")
        
        try:
            while True:
                # 1. Get Frame
                frame_read = self.tello.get_frame_read()
                frame = frame_read.frame
                frame = cv2.resize(frame, (640, 480))
                h, w, _ = frame.shape
                center_x, center_y = w // 2, h // 2

                # 2. Run Inference using Tasks API
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # The new API requires a specific MediaPipe Image object and a timestamp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)
                
                results = self.detector.detect_for_video(mp_image, timestamp_ms)
                
                # 3. Find our target
                person_box, area = self.get_person_bounding_box(results, w, h)

                left_right, forward_backward, up_down, yaw = 0, 0, 0, 0

                if person_box is not None:
                    # Reset search timers
                    self.time_target_lost = None
                    self.search_step_start = None
                    self.search_state = "ROTATE"
                    
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
                        
                        error_x = obj_cx - center_x
                        error_y = obj_cy - center_y

                        yaw = int(error_x * self.p_yaw)
                        up_down = int(-error_y * self.p_throttle)

                        if area > self.area_range[1]:      
                            forward_backward = -20
                        elif area < self.area_range[0]:    
                            forward_backward = 20
                else:
                    self.target_locked = False
                    self.first_seen_time = None
                    
                    if self.time_target_lost is None:
                        self.time_target_lost = time.time()
                        
                    time_lost = time.time() - self.time_target_lost
                    
                    # PHASE 1: Quick directional scan (4 seconds)
                    if time_lost < 4.0:
                        cv2.putText(frame, "PHASE 1: QUICK PAN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if self.last_known_x < center_x:
                            yaw = -30 
                        elif self.last_known_x > center_x:
                            yaw = 30  
                        else:
                            yaw = 0
                            
                    # PHASE 2: Deep 360 Vertical Search
                    else:
                        if self.search_step_start is None:
                            self.search_step_start = time.time()
                            
                        elapsed_step_time = time.time() - self.search_step_start
                        cv2.putText(frame, f"PHASE 2: 360 {self.search_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                        if self.search_state == "ROTATE":
                            yaw = 45  
                            if elapsed_step_time > 6.0:  
                                self.search_state = "DOWN"
                                self.search_step_start = time.time()
                                
                        elif self.search_state == "DOWN":
                            up_down = -25 
                            if elapsed_step_time > 2.0: 
                                self.search_state = "UP"
                                self.search_step_start = time.time()
                                
                        elif self.search_state == "UP":
                            up_down = 25 
                            if elapsed_step_time > 4.0:
                                self.search_state = "DOWN_CENTER"
                                self.search_step_start = time.time()
                                
                        elif self.search_state == "DOWN_CENTER":
                            up_down = -25 
                            if elapsed_step_time > 2.0:
                                self.search_state = "ROTATE" 
                                self.search_step_start = time.time()

                # Clamp speeds and Send
                yaw = max(-100, min(100, yaw))
                up_down = max(-100, min(100, up_down))
                forward_backward = max(-100, min(100, forward_backward))
                
                self.tello.send_rc_control(0, forward_backward, up_down, yaw)

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
            self.detector.close() # Free memory
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TelloPersonTracker()
    tracker.connect_and_takeoff()
    tracker.track()