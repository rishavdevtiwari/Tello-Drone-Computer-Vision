import time
from djitellopy import Tello

def main():
    # 1. Initialize and connect to the Tello drone
    tello = Tello()
    tello.connect()
    
    # Strict battery check for flips
    battery = tello.get_battery()
    print(f"Current Battery: {battery}%")
    if battery < 50:
        print("⚠️ Warning: Battery is below 50%. Tello firmware will reject flip commands!")
        print("Please charge the battery fully before attempting a flip.")
        return

    print("Pre-flight check passed. Clearing flight path...")
    time.sleep(2)

    try:
        # 2. Takeoff (Default hover height is roughly 90cm to 100cm)
        print("Taking off...")
        tello.takeoff()
        
        # Crucial: Give the drone 4 seconds to become perfectly stable
        print("Stabilizing hover...")
        time.sleep(4)

        # 3. Ascend to a safe flipping altitude
        # We go up an extra 40 cm to give it plenty of room to recover from the flip drop
        print("Ascending to safe flip altitude (approx 140 cm)...")
        tello.move_up(40)
        time.sleep(3)  # Wait for momentum to stop completely

        # 4. Execute the Flip
        # Options: 'f' (forward), 'b' (backward), 'l' (left), 'r' (right)
        print("Look out! Executing forward flip now...")
        tello.flip('f')
        
        # Crucial: Give the drone time to recover its orientation, stabilize its altitude,
        # and re-engage its downward vision positioning system sensors
        print("Flip complete. Re-stabilizing position...")
        time.sleep(4)

        # 5. Land Safely
        print("Landing cleanly...")
        tello.land()

    except Exception as e:
        print(f"An unexpected error occurred during flight: {e}")
        print("Executing emergency safety landing...")
        tello.land()

if __name__ == "__main__":
    main()