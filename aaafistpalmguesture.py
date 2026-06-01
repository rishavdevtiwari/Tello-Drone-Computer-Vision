import cv2
import mediapipe as mp
from djitellopy import Tello
import time
import sys

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
print("Initializing MediaPipe...")
mp_hands = mp.solutions.hands
# Use static_image_mode=False for faster video stream processing
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

print("Connecting to Tello Drone...")
drone = Tello()

try:
    drone.connect()
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to drone. Check Wi-Fi. ({e})")
    sys.exit(1)

battery = drone.get_battery()
print(f"Battery Level: {battery}%")
if battery < 15:
    print("WARNING: Battery too low for a safe flight. Aborting.")
    sys.exit(1)

drone.streamon()
frame_reader = drone.get_frame_read()
time.sleep(2) # Give the camera sensor time to warm up

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def is_open_palm(hand_landmarks):
    """Returns True if all 4 fingers are extended upward."""
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    extended_count = sum(1 for tip, pip in zip(finger_tips, finger_pips) 
                         if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    return extended_count == 4

def is_closed_fist(hand_landmarks):
    """Returns True if all 4 fingers are curled downward (fist)."""
    finger_tips = [8, 12, 16, 20]
    # Check against the lower knuckles (MCP joints) to ensure a tight fist
    finger_mcps = [5, 9, 13, 17] 
    curled_count = sum(1 for tip, mcp in zip(finger_tips, finger_mcps) 
                       if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y)
    return curled_count == 4

# State Machine definitions
STATE_TAKEOFF = 0
STATE_SEARCHING = 1
STATE_TRACKING = 2
current_state = STATE_TAKEOFF

print("Starting main execution loop...")

# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================
try:
    while True:
        frame = frame_reader.frame
        if frame is None:
            continue
            
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape
        frame_center_x, frame_center_y = w // 2, h // 2
        
        # State: Initial Takeoff
        if current_state == STATE_TAKEOFF:
            drone.takeoff()
            drone.move_up(40) # Ascend to chest/face height
            current_state = STATE_SEARCHING
            
        # CV Processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        hand_detected = False
        lr_speed, fb_speed, ud_speed, yaw_speed = 0, 0, 0, 0
        status_text = "Searching..."
        status_color = (0, 0, 255)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # Check for FIST first (Priority override to land)
                if is_closed_fist(hand_lms):
                    print("\n[!] Closed fist detected! Initiating emergency hover and land.")
                    # Setting this flag and breaking the loop hands control to the 'finally' block
                    break 
                
                # Check for PALM
                elif is_open_palm(hand_lms):
                    hand_detected = True
                    current_state = STATE_TRACKING
                    
                    palm_x = int(hand_lms.landmark[0].x * w)
                    palm_y = int(hand_lms.landmark[0].y * h)
                    cv2.circle(frame, (palm_x, palm_y), 10, (0, 255, 0), -1)
                    
                    error_x = palm_x - frame_center_x
                    error_y = frame_center_y - palm_y
                    
                    # Proportional gains
                    k_p_yaw = 0.4
                    k_p_ud = 0.4
                    
                    if abs(error_x) > 40:
                        yaw_speed = max(-50, min(50, int(error_x * k_p_yaw)))
                    if abs(error_y) > 40:
                        ud_speed = max(-40, min(40, int(error_y * k_p_ud)))
                        
                    status_text = "Tracking Palm"
                    status_color = (0, 255, 0)
        else:
            # No hands visible, revert to searching
            current_state = STATE_SEARCHING

        # Send movement commands
        if current_state == STATE_TRACKING:
            drone.send_rc_control(lr_speed, fb_speed, ud_speed, yaw_speed)
        elif current_state == STATE_SEARCHING:
            drone.send_rc_control(0, 0, 0, 0)

        # UI Overlay
        cv2.putText(frame, f"STATE: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.imshow("Tello Camera", frame)
        
        # Manual override: Press 'q' to quit and land
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[!] Manual quit triggered.")
            break

except KeyboardInterrupt:
    print("\n[!] Script interrupted by user (Ctrl+C).")
except Exception as e:
    print(f"\n[!] Unexpected Error during flight: {e}")

# ==========================================
# 4. FAIL-SAFE SHUTDOWN & LANDING
# ==========================================
finally:
    print("\n--- INITIATING FAIL-SAFE SHUTDOWN ---")
    try:
        # 1. Immediately kill all velocities to hover straight
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5) # Brief pause to stabilize
        
        # 2. Execute vertical landing exactly below current position
        print("Landing...")
        drone.land()
    except Exception as land_error:
        print(f"Error during landing sequence: {land_error}")
        
    print("Releasing camera and network resources...")
    try:
        drone.streamoff()
    except:
        pass
    cv2.destroyAllWindows()
    print("Shutdown complete. Safe to approach drone.")