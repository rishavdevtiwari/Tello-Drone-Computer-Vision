# YOLOv8 Person Tracker & 360 Vertical Search for DJI Tello Drone
# ultralytics YOLOv8 model to detect and track a person

import cv2
import sys
import time
from djitellopy import Tello
from ultralytics import YOLO

class TelloPersonTracker:
    def __init__(self):
        # 1. Initialize Drone
        self.tello = Tello()
        
        # 2. Initialize YOLO Model
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        
        # 3. Tracking State Variables
        self.target_locked = False
        self.first_seen_time = None
        self.required_recognition_time = 2.0  # Seconds before moving
        
        # 4. Flight Tuning Parameters (Proportional Control)
        self.p_yaw = 0.3      
        self.p_pitch = 0.3    
        self.p_throttle = 0.4 
        
        self.target_area = 80000 
        self.area_range = [70000, 90000] 

        # 5. Autonomous Search Pattern Variables
        self.search_state = "ROTATE"   # States: ROTATE, DOWN, UP, DOWN_CENTER
        self.search_step_start = None  # Tracks how long we've been in the current state

    def connect_and_takeoff(self):
        """Connects to the drone, starts the stream, and takes off."""
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")
        
        if self.tello.get_battery() < 15:
            print("Battery too low for safe flight. Exiting.")
            sys.exit(1)

        self.tello.streamon()
        self.tello.takeoff()
        self.tello.move_up(80) # Move to roughly chest/face height

    def get_largest_person(self, results, frame_center):
        """Finds the largest person in the frame to lock onto."""
        best_box = None
        max_area = 0
        
        for box in results[0].boxes:
            if int(box.cls[0]) == 0: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, w, h)
                    
        return best_box, max_area

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

                # 2. Run Inference
                results = self.model(frame, classes=[0], verbose=False)
                
                # 3. Find our target
                person_box, area = self.get_largest_person(results, (center_x, center_y))

                # Default speeds (Hover)
                left_right, forward_backward, up_down, yaw = 0, 0, 0, 0

                if person_box is not None:
                    # Reset search pattern timers because we found someone
                    self.search_step_start = None
                    self.search_state = "ROTATE"
                    
                    # --- PERSON DETECTED ---
                    x1, y1, bw, bh = person_box
                    obj_cx = x1 + (bw // 2)
                    obj_cy = y1 + (bh // 2)

                    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
                    cv2.circle(frame, (obj_cx, obj_cy), 5, (0, 0, 255), cv2.FILLED)

                    # Recognition Timer (Follow after 2 seconds)
                    if self.first_seen_time is None:
                        self.first_seen_time = time.time()
                        cv2.putText(frame, "Recognizing...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    
                    elif time.time() - self.first_seen_time >= self.required_recognition_time:
                        self.target_locked = True
                        cv2.putText(frame, "TARGET LOCKED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # --- CALCULATE TRACKING MOVEMENT ---
                        error_x = obj_cx - center_x
                        error_y = obj_cy - center_y

                        yaw = int(error_x * self.p_yaw)
                        up_down = int(-error_y * self.p_throttle)

                        if area > self.area_range[1]:      
                            forward_backward = -20
                        elif area < self.area_range[0]:    
                            forward_backward = 20

                else:
                    # --- PERSON LOST / SEARCH PATTERN ---
                    self.target_locked = False
                    self.first_seen_time = None
                    
                    # Start the step timer for the search sequence
                    if self.search_step_start is None:
                        self.search_step_start = time.time()
                        
                    elapsed_time = time.time() - self.search_step_start
                    
                    # Display Current Search State
                    cv2.putText(frame, f"SEARCH: {self.search_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                    if self.search_state == "ROTATE":
                        yaw = 45  # Spin Right
                        # Adjust 6.0 based on your Tello's actual rotation speed to get a perfect 360
                        if elapsed_time > 6.0:  
                            self.search_state = "DOWN"
                            self.search_step_start = time.time()
                            
                    elif self.search_state == "DOWN":
                        up_down = -25 # Fly down
                        if elapsed_time > 2.0: 
                            self.search_state = "UP"
                            self.search_step_start = time.time()
                            
                    elif self.search_state == "UP":
                        up_down = 25 # Fly up (4 seconds to go back past start height to look high)
                        if elapsed_time > 4.0:
                            self.search_state = "DOWN_CENTER"
                            self.search_step_start = time.time()
                            
                    elif self.search_state == "DOWN_CENTER":
                        up_down = -25 # Fly down to return to original height
                        if elapsed_time > 2.0:
                            self.search_state = "ROTATE" # Loop back to spinning
                            self.search_step_start = time.time()

                # 4. Clamp speeds to safe Tello limits (-100 to 100)
                yaw = max(-100, min(100, yaw))
                up_down = max(-100, min(100, up_down))
                forward_backward = max(-100, min(100, forward_backward))

                # 5. Send Commands
                self.tello.send_rc_control(0, forward_backward, up_down, yaw)

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
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TelloPersonTracker()
    tracker.connect_and_takeoff()
    tracker.track()