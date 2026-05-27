# MediaPipe Person Tracker for DJI Tello Drone

import cv2
import time
from djitellopy import Tello
import mediapipe as mp

class TelloPersonTracker:
    def __init__(self):
        # 1. Initialize Drone
        self.tello = Tello()
        
        # 2. Initialize MediaPipe Pose Model
        print("Loading MediaPipe Pose model...")
        self.mp_pose = mp.solutions.pose
        # min_detection_confidence limits false positives
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 3. Tracking State Variables
        self.target_locked = False
        self.first_seen_time = None
        self.required_recognition_time = 2.0  # Seconds before moving
        self.last_known_x = 0  # To track which way they went out of frame
        
        # 4. Flight Tuning Parameters (Proportional Control)
        self.p_yaw = 0.3      # Rotation speed multiplier
        self.p_pitch = 0.3    # Forward/Back speed multiplier
        self.p_throttle = 0.4 # Up/Down speed multiplier
        
        # Target sizes for the bounding box (Area = Width * Height)
        # Note: MediaPipe skeleton bounding boxes are tighter than YOLO boxes.
        # You may need to reduce these numbers slightly if the drone flies too close.
        self.target_area = 80000 
        self.area_range = [70000, 90000] # "Goldilocks" zone

    def connect_and_takeoff(self):
        """Connects to the drone, starts the stream, and takes off."""
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")
        
        if self.tello.get_battery() < 15:
            print("Battery too low for safe flight. Exiting.")
            exit()

        self.tello.streamon()
        self.tello.takeoff()
        self.tello.move_up(80) # Move to roughly chest/face height

    def get_person_bounding_box(self, results, w, h):
        """Calculates a pseudo-bounding box from MediaPipe Pose landmarks."""
        if not results.pose_landmarks:
            return None, 0
        
        # Extract X and Y coordinates of all 33 skeleton joints
        x_coords = [lm.x for lm in results.pose_landmarks.landmark]
        y_coords = [lm.y for lm in results.pose_landmarks.landmark]
        
        # Find the outer edges of the person
        x1 = int(min(x_coords) * w)
        y1 = int(min(y_coords) * h)
        x2 = int(max(x_coords) * w)
        y2 = int(max(y_coords) * h)
        
        # Clamp values to frame boundaries to prevent off-screen window errors
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        
        return (x1, y1, bw, bh), area

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

                # 2. Run Inference
                # MediaPipe requires RGB format, OpenCV defaults to BGR
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                # 3. Find our target
                person_box, area = self.get_person_bounding_box(results, w, h)

                # Default speeds (Hover)
                left_right, forward_backward, up_down, yaw = 0, 0, 0, 0

                if person_box is not None:
                    # --- PERSON DETECTED ---
                    x1, y1, bw, bh = person_box
                    obj_cx = x1 + (bw // 2)
                    obj_cy = y1 + (bh // 2)
                    
                    self.last_known_x = obj_cx

                    # Draw the bounding box and the skeleton
                    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
                    cv2.circle(frame, (obj_cx, obj_cy), 5, (0, 0, 255), cv2.FILLED)
                    self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    # Handle 2-second recognition logic
                    if self.first_seen_time is None:
                        self.first_seen_time = time.time()
                        cv2.putText(frame, "Recognizing...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    
                    elif time.time() - self.first_seen_time >= self.required_recognition_time:
                        self.target_locked = True
                        cv2.putText(frame, "TARGET LOCKED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # --- CALCULATE MOVEMENT ---
                        error_x = obj_cx - center_x
                        error_y = obj_cy - center_y

                        yaw = int(error_x * self.p_yaw)
                        up_down = int(-error_y * self.p_throttle)

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
            self.pose.close() # Free MediaPipe resources
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TelloPersonTracker()
    tracker.connect_and_takeoff()
    tracker.track()