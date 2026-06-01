import time
import threading
import queue
import curses
from djitellopy import Tello

# Safe command queue
command_queue = queue.Queue()
# Global stats dictionary for UI update
drone_stats = {"battery": 0, "status": "Connecting...", "last_action": "None"}

def drone_worker(drone):
    """Handles drone operations sequentially in the background."""
    global drone_stats
    step_dist = 30  # Fixed distance per tap in cm
    step_deg = 45   # Fixed rotation angle per tap in degrees
    
    while True:
        try:
            # Regularly update battery while idling
            if command_queue.empty():
                try:
                    drone_stats["battery"] = drone.get_battery()
                except:
                    pass
                time.sleep(0.4)
                continue

            action = command_queue.get()
            if action == "quit":
                drone_stats["status"] = "Exiting..."
                if drone.is_flying:
                    drone.land()
                break

            # Absolute execution of commands
            if action == "takeoff":
                drone_stats["status"] = "Taking off..."
                drone.takeoff()
            elif action == "land":
                drone_stats["status"] = "Landing..."
                drone.land()
            elif action == "forward":
                drone_stats["status"] = "Moving Forward"
                drone.move_forward(step_dist)
            elif action == "backward":
                drone_stats["status"] = "Moving Backward"
                drone.move_backward(step_dist)
            elif action == "left":
                drone_stats["status"] = "Moving Left"
                drone.move_left(step_dist)
            elif action == "right":
                drone_stats["status"] = "Moving Right"
                drone.move_right(step_dist)
            elif action == "up":
                drone_stats["status"] = "Moving Up"
                drone.move_up(step_dist)
            elif action == "down":
                drone_stats["status"] = "Moving Down"
                drone.move_down(step_dist)
            elif action == "rot_left":
                drone_stats["status"] = "Rotating Left"
                drone.rotate_counter_clockwise(step_deg)
            elif action == "rot_right":
                drone_stats["status"] = "Rotating Right"
                drone.rotate_clockwise(step_deg)

            drone_stats["last_action"] = action
            drone_stats["status"] = "Hovering / Ready"
            
        except Exception as e:
            drone_stats["status"] = f"Error: {str(e)[:25]}"
            time.sleep(1.5)

def safe_addstr(stdscr, y, x, text, attr=curses.A_NORMAL):
    """Safely prints text to the screen, preventing crashes if the terminal is resized too small."""
    max_y, max_x = stdscr.getmaxyx()
    if y < max_y and x < max_x:
        # Truncate text if it overflows the width of the window
        try:
            stdscr.addstr(y, x, text[:max_x - x - 1], attr)
        except curses.error:
            pass

def main_ui(stdscr):
    # Setup curses UI environment
    curses.curs_set(0)   # Hide the blinky typing cursor
    stdscr.nodelay(True) # Make getch() non-blocking
    stdscr.timeout(50)   # Refresh UI every 50ms
    stdscr.keypad(True)  # Enable arrow keys parsing
    
    drone = Tello()
    try:
        drone.connect()
        drone_stats["battery"] = drone.get_battery()
        drone_stats["status"] = "Connected / Idle"
    except Exception as e:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Connection Failed: {e}. Press any key to exit.")
        stdscr.nodelay(False)
        stdscr.getch()
        return

    # Start background thread
    threading.Thread(target=drone_worker, args=(drone,), daemon=True).start()

    while True:
        stdscr.clear()
        
        # --- UI Header & Stats ---
        safe_addstr(stdscr, 1, 2, "=== TELLO FULL TERMINAL REMOTE CONTROL ===", curses.A_BOLD)
        
        bat_color = curses.A_REVERSE if drone_stats["battery"] < 20 else curses.A_NORMAL
        safe_addstr(stdscr, 3, 2, f"Battery Level : [ {drone_stats['battery']}% ]", bat_color)
        safe_addstr(stdscr, 4, 2, f"Current Status: {drone_stats['status']}")
        safe_addstr(stdscr, 5, 2, f"Last Command  : {drone_stats['last_action']}")
        
        # --- Control Pad Map Layout ---
        safe_addstr(stdscr, 7, 2, "CONTROLS (No Enter required - Tap keys directly):", curses.A_UNDERLINE)
        safe_addstr(stdscr, 9, 4, "  [W] Forward        [Q] Rotate Left       [UP ARROW]   Up")
        safe_addstr(stdscr, 10, 4, "[A] Left  [S] Back   [E] Rotate Right      [DOWN ARROW] Down")
        safe_addstr(stdscr, 11, 4, "  [D] Right")
        safe_addstr(stdscr, 13, 4, "[T] Takeoff          [L] Land              [ESC] Emergency Quit")
        safe_addstr(stdscr, 15, 2, "-----------------------------------------------------------------")
        
        stdscr.refresh()

        # --- Capture Keyboard Inputs Instantly ---
        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            command_queue.put("quit")
            break

        if key == -1:
            continue
        
        # Check if the drone is busy processing a command
        is_busy = not command_queue.empty()
        
        # Structural Emergency Overrides
        if key in (ord('l'), ord('L')):
            # Clear queue and land immediately
            while not command_queue.empty():
                command_queue.get()
            command_queue.put("land")
        elif key == 27:  # Escape Key
            while not command_queue.empty():
                command_queue.get()
            command_queue.put("quit")
            break
            
        # Standard Flight Actions (Ignored if busy executing a move)
        elif not is_busy:
            if key in (ord('t'), ord('T')):
                command_queue.put("takeoff")
            elif key in (ord('w'), ord('W')):
                command_queue.put("forward")
            elif key in (ord('s'), ord('S')):
                command_queue.put("backward")
            elif key in (ord('a'), ord('A')):
                command_queue.put("left")
            elif key in (ord('d'), ord('D')):
                command_queue.put("right")
            elif key in (ord('q'), ord('Q')):
                command_queue.put("rot_left")
            elif key in (ord('e'), ord('E')):
                command_queue.put("rot_right")
            elif key == curses.KEY_UP:
                command_queue.put("up")
            elif key == curses.KEY_DOWN:
                command_queue.put("down")

if __name__ == "__main__":
    curses.wrapper(main_ui)