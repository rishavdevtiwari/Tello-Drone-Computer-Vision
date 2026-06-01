import time
import threading
import queue
from djitellopy import Tello

# Create a queue to hold commands from the user
command_queue = queue.Queue()

def input_thread():
    """This thread runs in the background and constantly listens for terminal input."""
    print("\n--- Tello Controller Ready ---")
    print("Commands: takeoff, land, f, b, u, d, quit")
    print("---------------------------------")
    
    while True:
        cmd = input("Enter command: ").strip().lower()
        command_queue.put(cmd)
        if cmd == "quit":
            break

def main():
    # Initialize and connect to the drone
    drone = Tello()
    
    print("Connecting to Tello...")
    try:
        drone.connect()
        print(f"Connected! Battery Life: {drone.get_battery()}%")
    except Exception as e:
        print(f"Could not connect to Tello: {e}")
        return

    # Start the background input thread
    threading.Thread(target=input_thread, daemon=True).start()

    distance = 30  # cm
    running = True

    print("Ready for commands. You can type at any time!")
    
    while running:
        # Check if there is a command waiting in the queue
        if not command_queue.empty():
            command = command_queue.get()

            try:
                if command == "takeoff":
                    print("\n[Executing] Taking off...")
                    drone.takeoff()
                    print("[Done] Takeoff complete. Ready for next command.")
                
                elif command == "land":
                    print("\n[Executing] Landing...")
                    drone.land()
                    print("[Done] Landed.")
                
                elif command == "f":
                    print(f"\n[Executing] Moving forward {distance}cm...")
                    drone.move_forward(distance)
                
                elif command == "b":
                    print(f"\n[Executing] Moving backward {distance}cm...")
                    drone.move_backward(distance)
                
                elif command == "u":
                    print(f"\n[Executing] Moving up {distance}cm...")
                    drone.move_up(distance)
                
                elif command == "d":
                    print(f"\n[Executing] Moving down {distance}cm...")
                    drone.move_down(distance)
                
                elif command == "quit":
                    print("\n[Exiting] Landing and closing program...")
                    if drone.is_flying:
                        drone.land()
                    running = False
                
                else:
                    print(f"\nUnknown command '{command}'. Try: takeoff, land, f, b, u, d, quit")

            except Exception as e:
                print(f"\n[Error] Flight error: {e}")
                print("Attempting emergency land...")
                try:
                    drone.land()
                except:
                    pass
                running = False
        
        # Small sleep to prevent the CPU from running at 100%
        time.sleep(0.1)

    print("Program ended.")

if __name__ == "__main__":
    main()