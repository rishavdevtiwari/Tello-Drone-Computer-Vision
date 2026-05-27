from djitellopy import Tello
import time

def execute_flight_plan():
    # 1. Initialize and connect
    tello = Tello()
    tello.connect()
    
    print(f"Battery Life: {tello.get_battery()}%")
    if tello.get_battery() < 10:
        print("Battery too low. Aborting flight.")
        return

    try:
        # 2. Take off
        print("Taking off...")
        tello.takeoff()
        
        # Give the drone a moment to stabilize its altitude
        time.sleep(2) 

        # 3. Rotate 360 degrees
        print("Executing 360 rotation...")
        tello.rotate_clockwise(360)

        # 4. Move in a perfect square
        # A square consists of moving forward and turning 90 degrees, 4 times.
        print("Executing square pattern...")
        for _ in range(4):
            tello.move_forward(100)    # Move forward 100 cm
            tello.rotate_clockwise(90) # Turn right 90 degrees

        # 5. Rotate 40 degrees both sides
        print("Looking left and right...")
        tello.rotate_clockwise(40)         # Rotate right 40 degrees
        tello.rotate_counter_clockwise(80) # Sweep left 80 degrees (passes center to reach -40)
        tello.rotate_clockwise(40)         # Return to the exact center

        # 6. Slowly land
        print("Initiating slow descent...")
        # Send a manual joystick command to lower altitude slowly (-15 speed)
        tello.send_rc_control(0, 0, -15, 0) 
        time.sleep(3)                       # Hold that slow descent for 3 seconds
        tello.send_rc_control(0, 0, 0, 0)   # Stop descending to prevent ground bounce
        
        print("Touching down...")
        tello.land()

    except Exception as e:
        print(f"Flight error: {e}")
        # Failsafe: Ensure drone lands safely if the script gets interrupted
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()

    print("Done controlling.")

if __name__ == "__main__":
    execute_flight_plan()