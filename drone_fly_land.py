#Drone takes off, ascends to a stable hover, then lands safely.

import time
from djitellopy import Tello

def main():
    # Initialization and connection
    tello = Tello()
    tello.connect()
    
    print("Prepare for takeoff...")
    time.sleep(2)

    # Takeoff
    print("Taking off...")
    tello.takeoff()
    time.sleep(2) # Allow drone to stabilize

    # Ascend to stable altitude
    print("Ascending to stable hovering height...")
    tello.move_up(50)
    time.sleep(2)

    # Land
    print("Landing...")
    tello.land()

if __name__ == "__main__":
    main()