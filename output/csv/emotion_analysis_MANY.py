import pandas as pd
import matplotlib.pyplot as plt

# Read data from the local CSV file
df = pd.read_csv('output\csv\emotions_data_2024-01-05_00-39-06.csv')

# Convert 'video_time_readable' to a datetime format for easier plotting
df['timestamp'] = pd.to_datetime(df['video_time_readable'], format='%M:%S')

# Calculate seconds since the start of the recording
start_time = df['timestamp'].min()
df['seconds_since_start'] = (df['timestamp'] - start_time).dt.total_seconds()

# Define the sets for negative, positive, and neutral emotions for simplicity
negative_emotions = {'Angry', 'Disgust', 'Fear', 'Sad'}
positive_emotions = {'Happy'}
neutral_emotions = {'Surprise', 'Neutral'}

# Filter out rows where no emotion probabilities are provided
df = df.dropna(subset=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])

# Plotting
plt.figure(figsize=(12, 6))

# Plot all emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
colors = {'Angry':'red', 'Disgust':'orange', 'Fear':'yellow', 'Happy':'green', 'Sad':'blue', 'Surprise':'purple', 'Neutral':'grey'}
for emotion in emotions:
    plt.plot(df['seconds_since_start'], df[emotion], label=emotion, color=colors[emotion])

# Highlight the moments where a negative emotion is detected
negative_emotion_detected = df[df['DetectedString'].isin(negative_emotions)]
plt.scatter(negative_emotion_detected['seconds_since_start'], [0] * len(negative_emotion_detected), 
            color='black', label='Negative Emotion Detected', marker='X')

plt.xlabel('Seconds Since Start')
plt.ylabel('Emotion Probability')
plt.title('Emotion Probabilities Over Time')
plt.legend()
plt.tight_layout()
plt.show()
