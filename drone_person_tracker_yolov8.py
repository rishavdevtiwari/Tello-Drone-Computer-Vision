#YOLOv8 Person Tracker for DJI Tello Drone
#ultralytics YOLOv8 model to detect and track a person

import cv2
import sys
import time
from djitellopy import Tello
from ultralytics import YOLO

class TelloPersonTracker:
    def __init__(self):
        # 1. Initialize Drone
        self.tello = Tello()
        
        # 2. Initialize YOLO Model (n = nano, fastest for real-time video)
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        
        # 3. Tracking State Variables
        self.target_locked = False
        self.first_seen_time = None
        self.required_recognition_time = 2.0  # Seconds before moving
        self.last_known_x = 0  # To track which way they went out of frame
        
        # 4. Flight Tuning Parameters (Proportional Control)
        # Adjust these if the drone is too aggressive or too sluggish
        self.p_yaw = 0.3      # Rotation speed multiplier
        self.p_pitch = 0.3    # Forward/Back speed multiplier
        self.p_throttle = 0.4 # Up/Down speed multiplier
        
        # Target sizes for the bounding box (Area = Width * Height)
        # If the box is smaller than this, the drone moves forward
        self.target_area = 80000 
        self.area_range = [70000, 90000] # "Goldilocks" zone (doesn't move forward or back)

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
        
        # YOLO results contain bounding boxes for all detections
        for box in results[0].boxes:
            # Class 0 in COCO dataset is 'person'
            if int(box.cls[0]) == 0: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, w, h)
                    
        return best_box, max_area

    def track(self):
        """Main tracking loop."""
        print("Starting tracking loop. Press 'q' in the video window to quit & land.")
        
        try:
            while True:
                # 1. Get Frame
                frame_read = self.tello.get_frame_read()
                frame = frame_read.frame
                frame = cv2.resize(frame, (640, 480))
                h, w, _ = frame.shape
                center_x, center_y = w // 2, h // 2

                # 2. Run Inference (Detect people)
                # verbose=False stops it from spamming the console every frame
                results = self.model(frame, classes=[0], verbose=False)
                
                # 3. Find our target
                person_box, area = self.get_largest_person(results, (center_x, center_y))

                # Default speeds (Hover)
                left_right, forward_backward, up_down, yaw = 0, 0, 0, 0

                if person_box is not None:
                    # --- PERSON DETECTED ---
                    x1, y1, bw, bh = person_box
                    obj_cx = x1 + (bw // 2)
                    obj_cy = y1 + (bh // 2)
                    
                    # Update last known position for search logic
                    self.last_known_x = obj_cx

                    # Draw visual targeting reticle
                    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
                    cv2.circle(frame, (obj_cx, obj_cy), 5, (0, 0, 255), cv2.FILLED)

                    # Handle 2-second recognition logic
                    if self.first_seen_time is None:
                        self.first_seen_time = time.time()
                        cv2.putText(frame, "Recognizing...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    
                    elif time.time() - self.first_seen_time >= self.required_recognition_time:
                        self.target_locked = True
                        cv2.putText(frame, "TARGET LOCKED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # --- CALCULATE MOVEMENT ---
                        # Error is how far the person is from the center of the frame
                        error_x = obj_cx - center_x
                        error_y = obj_cy - center_y

                        # Calculate Yaw (Rotation to center person horizontally)
                        yaw = int(error_x * self.p_yaw)

                        # Calculate Up/Down (Throttle to center person vertically)
                        # Note: Tello moves down for negative throttle, but image Y goes down as it increases.
                        up_down = int(-error_y * self.p_throttle)

                        # Calculate Forward/Backward (Pitch to maintain distance)
                        if area > self.area_range[1]:      # Too close, move back
                            forward_backward = -20
                        elif area < self.area_range[0]:    # Too far, move forward
                            forward_backward = 20

                else:
                    # --- PERSON LOST ---
                    self.target_locked = False
                    self.first_seen_time = None
                    cv2.putText(frame, "SEARCHING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Search logic: Rotate in the direction they were last seen
                    if self.last_known_x < center_x:
                        yaw = -30 # Rotate Left
                    elif self.last_known_x > center_x:
                        yaw = 30  # Rotate Right
                    else:
                        yaw = 0   # Hover if we have no data

                # 4. Clamp speeds to safe Tello limits (-100 to 100)
                yaw = max(-100, min(100, yaw))
                up_down = max(-100, min(100, up_down))
                forward_backward = max(-100, min(100, forward_backward))

                # 5. Send Commands
                # Format: left_right, forward_backward, up_down, yaw
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
            # Ensure the drone ALWAYS lands safely if the script breaks or stops
            print("Landing and cleaning up...")
            self.tello.send_rc_control(0, 0, 0, 0) # Stop all movement
            self.tello.land()
            self.tello.streamoff()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TelloPersonTracker()
    tracker.connect_and_takeoff()
    tracker.track()