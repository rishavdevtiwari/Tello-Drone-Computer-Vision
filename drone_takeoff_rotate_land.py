import time
from djitellopy import Tello

def main():
    # Initialization and connection
    tello = Tello()
    tello.connect()
    
    # battery check just in case
    battery = tello.get_battery()
    print(f"Current Battery: {battery}%")
    if battery < 20:
        print("Battery too low for safe flight! Please charge the drone.")
        return

    print("Pre-flight check complete. Prepare for takeoff...")
    time.sleep(2)

    try:
        #Takeoff
        print("Taking off...")
        tello.takeoff()
        
        #drone stabilize
        time.sleep(2)

        # Move up so no inference
        # 50 centimetres == 1.6 feet
        print("Ascending to stable hovering height...")
        tello.move_up(50)
        time.sleep(2)

        # Rotate 360 degrees
        # rotate_clockwise command rotates 360
        print("Executing 360-degree rotation...")
        tello.rotate_clockwise(360)
        
        # drone stabilized before land
        time.sleep(2)

        # land
        print("Attempting precise landing on the takeoff spot...")
        tello.land()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Emergency safety backup
        print("Executing emergency safety landing...")
        tello.land()

if __name__ == "__main__":
    main()