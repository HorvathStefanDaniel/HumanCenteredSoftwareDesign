import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def main():
    # Set up the root Tkinter window
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing

    # Show an "Open" dialog box and return the path to the selected file
    file_path = filedialog.askopenfilename(
        initialdir="output\\csv\\", 
        title="Select CSV File", 
        filetypes=(("CSV files", "*.csv"), ("all files", "*.*"))
    )

    # Check if a file was selected
    if file_path:
        # Read data from the selected CSV file
        df = pd.read_csv(file_path)

        # Convert 'video_time_readable' to seconds for easier plotting
        time_parts = df['video_time_readable'].str.split(':', expand=True)
        df['seconds_since_start'] = time_parts[0].astype(int) * 60 + time_parts[1].astype(int)

        # Define the sets for negative, positive, and neutral emotions for simplicity
        negative_emotions = {'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise'}
        positive_emotions = {'Happy'}
        neutral_emotions = {'Neutral'}

        # Filter out rows where no emotion probabilities are provided
        emotion_columns = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        df = df.dropna(subset=emotion_columns)

        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot all emotions
        colors = {'Angry':'red', 'Disgust':'orange', 'Fear':'yellow', 'Happy':'green', 'Sad':'blue', 'Surprise':'purple', 'Neutral':'grey'}
        for emotion in emotion_columns:
            plt.plot(df['seconds_since_start'], df[emotion], label=emotion, color=colors[emotion])

        # Highlight the moments where a negative emotion or surprise is detected uninterrupted
        previous_negative = False
        for index, row in df.iterrows():
            if row['DetectedString'] in negative_emotions:
                if not previous_negative:  # new negative emotion sequence
                    plt.scatter(row['seconds_since_start'], 0, color='black', marker='X')
                    plt.annotate(row['video_time_readable'], 
                                 (row['seconds_since_start'], 0),
                                 textcoords="offset points", 
                                 xytext=(0,10), 
                                 ha='center')
                previous_negative = True
            else:
                previous_negative = False  # reset on non-negative emotion

        plt.xlabel('Seconds Since Start')
        plt.ylabel('Emotion Probability')
        plt.title('Emotion Probabilities Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
