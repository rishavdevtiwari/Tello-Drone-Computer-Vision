import cv2
import time
from djitellopy import Tello

def main():
    # 1. Initialize and connect
    tello = Tello()
    tello.connect()
    print(f"Battery Life: {tello.get_battery()}%")
    
    tello.streamon()
    frame_read = tello.get_frame_read()
    
    # 2. Initialize OpenCV Person Detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    print("Taking off soon... Stand back!")
    time.sleep(3)
    tello.takeoff()
    tello.move_up(40)
    
    # --- TUNING PARAMETERS ---
    FRAME_WIDTH = 640   # Reduced from 960 to drastically improve processing speed/FPS
    FRAME_HEIGHT = 480
    CENTER_X = FRAME_WIDTH // 2
    DEADZONE_X = 35     # Tightened deadzone for accuracy
    
    # Proportional Gain (Kp): Controls how aggressively the drone responds.
    # Higher value = faster turns; Lower value = smoother, slower turns.
    KP_YAW = 0.25       
    
    try:
        while True:
            frame = frame_read.frame
            if frame is None:
                continue
                
            # Downscaling the image is the easiest way to solve OpenCV CPU lag
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # Detect people with optimized groupThreshold to reduce false positives
            boxes, weights = hog.detectMultiScale(
                frame, 
                winStride=(8, 8), 
                padding=(4, 4), 
                scale=1.1, 
                groupThreshold=2
            )
            
            yaw_velocity = 0
            
            if len(boxes) > 0:
                # OPTIMIZATION: Always track the LARGEST bounding box (the person closest to the drone)
                # This stops the drone from getting distracted by background noise.
                main_person = max(boxes, key=lambda b: b[2] * b[3])
                (x, y, w, h) = main_person
                
                # Draw targeting overlay
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                person_center_x = x + (w // 2)
                
                # Draw a line from frame center to target center
                cv2.line(frame, (CENTER_X, FRAME_HEIGHT//2), (person_center_x, FRAME_HEIGHT//2), (255, 0, 0), 2)
                
                # --- PROPORTIONAL TRACKING LOGIC ---
                error_x = person_center_x - CENTER_X
                
                if abs(error_x) > DEADZONE_X:
                    # Speed is now dynamically calculated based on distance from center!
                    # If you are far away, it spins fast. As you get closer to center, it slows down perfectly.
                    calculated_speed = int(error_x * KP_YAW)
                    
                    # Constrain speed between safely manageable limits (-40 to 40)
                    yaw_velocity = max(min(calculated_speed, 40), -40)
                    
                    cv2.putText(frame, f"Tracking: Speed {yaw_velocity}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    yaw_velocity = 0
                    cv2.putText(frame, "Target Locked", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Searching...", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Send control (keeping pitch/roll/throttle at 0 for strict rotation)
            tello.send_rc_control(0, 0, 0, yaw_velocity)
            
            cv2.imshow("Tello Enhanced Proportional Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        print("Safely landing...")
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()