import tkinter as tk
from tkinter import scrolledtext
import subprocess
import os
import threading

def on_consent_check():
    """Enable or disable the 'Start Recording' button based on consent checkbox."""
    if consent_var.get():
        start_button.config(state=tk.NORMAL)
    else:
        start_button.config(state=tk.DISABLED)

def start_script():
    """Function to start the external script."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(current_dir, "test", "main_many.py")
    subprocess.run(["python", script_path])  # Start the script
    window.destroy()

def start_recording():
    """Start recording in a separate thread."""
    start_button.config(text="Starting...", state=tk.DISABLED)
    threading.Thread(target=start_script, daemon=True).start()

def quit_app():
    """Quit the application."""
    window.destroy()

# Create the main window
window = tk.Tk()
window.title("User Consent and Recording")

# Create a scrollable text widget
scroll_txt = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=40, height=10)
scroll_txt.insert(tk.INSERT, "Would you agree to have your video recorded and the resulting data utilized in machine learning models for emotional analysis? If you agree, please check the box below and click 'Start Recording'. If you do not agree, please click 'Quit'")
scroll_txt.grid(column=0, row=0, pady=10, padx=10)

# Checkbox for user consent
consent_var = tk.BooleanVar()
consent_check = tk.Checkbutton(window, text="I consent", var=consent_var, command=on_consent_check)
consent_check.grid(column=0, row=1, pady=5)

# 'Start Recording' button
start_button = tk.Button(window, text="Start Recording", command=start_recording, state=tk.DISABLED)
start_button.grid(column=0, row=2, pady=5)

# 'Quit' button
quit_button = tk.Button(window, text="Quit", command=quit_app)
quit_button.grid(column=0, row=3, pady=5)

# Run the application
window.mainloop()
