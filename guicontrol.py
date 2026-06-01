import tkinter as tk
import threading
from djitellopy import Tello

class TelloControlPad:
    def __init__(self, root):
        self.root = root
        self.root.title("Tello Flight Controller")
        self.root.geometry("350x450")
        self.root.configure(bg="#2c3e50") # Dark mode background

        # Initialize Drone Object
        self.tello = Tello()

        # --- UI ELEMENTS ---
        
        # Battery / Status Label
        self.lbl_status = tk.Label(root, text="Status: Disconnected", font=("Arial", 14, "bold"), bg="#2c3e50", fg="white")
        self.lbl_status.pack(pady=15)

        # Connect Button
        self.btn_connect = tk.Button(root, text="Connect to Drone", font=("Arial", 12), bg="#3498db", fg="white", command=self.connect_drone)
        self.btn_connect.pack(pady=5)

        # Control Panel Frame (Grid Layout)
        control_frame = tk.Frame(root, bg="#2c3e50")
        control_frame.pack(pady=15)

        # --- BUTTONS ---
        # Format: command=lambda: self.execute_command(function_to_run, argument)
        
        # Takeoff / Land
        tk.Button(control_frame, text="Take Off", bg="#2ecc71", fg="black", width=12, font=("Arial", 11, "bold"),
                  command=lambda: self.execute_command(self.tello.takeoff)).grid(row=0, column=0, padx=10, pady=10)
        
        tk.Button(control_frame, text="Land", bg="#f1c40f", fg="black", width=12, font=("Arial", 11, "bold"),
                  command=lambda: self.execute_command(self.tello.land)).grid(row=0, column=1, padx=10, pady=10)

        # Up / Down (Moves 30 cm per click)
        tk.Button(control_frame, text="Up (30cm)", width=12, font=("Arial", 10),
                  command=lambda: self.execute_command(self.tello.move_up, 30)).grid(row=1, column=0, padx=10, pady=5)
        
        tk.Button(control_frame, text="Down (30cm)", width=12, font=("Arial", 10),
                  command=lambda: self.execute_command(self.tello.move_down, 30)).grid(row=1, column=1, padx=10, pady=5)

        # Rotate Left / Right (Rotates 45 degrees per click)
        tk.Button(control_frame, text="Rotate Left", width=12, font=("Arial", 10),
                  command=lambda: self.execute_command(self.tello.rotate_counter_clockwise, 45)).grid(row=2, column=0, padx=10, pady=5)
        
        tk.Button(control_frame, text="Rotate Right", width=12, font=("Arial", 10),
                  command=lambda: self.execute_command(self.tello.rotate_clockwise, 45)).grid(row=2, column=1, padx=10, pady=5)

        # Tricks
        tk.Button(control_frame, text="Flip Back", width=12, font=("Arial", 10),
                  command=lambda: self.execute_command(self.tello.flip_back)).grid(row=3, column=0, padx=10, pady=10)
        
        tk.Button(control_frame, text="360 Spin", width=12, font=("Arial", 10),
                  command=lambda: self.execute_command(self.tello.rotate_clockwise, 360)).grid(row=3, column=1, padx=10, pady=10)

        # Emergency Stop (Kills motors instantly)
        tk.Button(root, text="EMERGENCY STOP", bg="#e74c3c", fg="white", width=25, font=("Arial", 12, "bold"),
                  command=lambda: self.execute_command(self.tello.emergency)).pack(pady=10)


    def connect_drone(self):
        """Connects to the drone in a background thread."""
        self.lbl_status.config(text="Connecting...", fg="#f1c40f")
        
        def _connect():
            try:
                self.tello.connect()
                battery = self.tello.get_battery()
                # Update GUI safely from thread
                self.root.after(0, self.lbl_status.config, {"text": f"Battery: {battery}%", "fg": "#2ecc71"})
                print("Connection Successful!")
            except Exception as e:
                self.root.after(0, self.lbl_status.config, {"text": "Connection Failed", "fg": "#e74c3c"})
                print(f"Error: {e}")
                
        threading.Thread(target=_connect, daemon=True).start()

    def execute_command(self, command_func, *args):
        """Runs drone movement commands in a separate thread to prevent UI freezing."""
        print(f"Executing: {command_func.__name__}")
        
        def _run():
            try:
                command_func(*args)
            except Exception as e:
                print(f"Command failed: {e}")
                
        threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    # Initialize the Tkinter window
    root = tk.Tk()
    
    # Run our application
    app = TelloControlPad(root)
    
    # Start the GUI event loop
    root.mainloop()