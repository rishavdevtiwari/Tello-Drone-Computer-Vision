#Drone takes off, checks battery, ascends to a stable hover, then lands safely.
import time
from djitellopy import Tello

def main():
    # Initialization and connection
    tello = Tello()
    tello.connect()
    
    # Battery check
    battery = tello.get_battery()
    print(f"Current Battery: {battery}%")
    if battery < 20:
        print("Battery too low for safe flight! Please charge the drone.")
        return

    print("Pre-flight check complete. Prepare for takeoff...")
    time.sleep(2)

    # Takeoff
    print("Taking off...")
    tello.takeoff()
    time.sleep(2) # Drone stabilize

    # Ascend to stable altitude
    print("Ascending to stable hovering height...")
    tello.move_up(50)
    time.sleep(2)

    # Land
    print("Landing...")
    tello.land()

if __name__ == "__main__":
    main()