import matplotlib.pyplot as plt
import pandas as pd

# Read CSV file into a DataFrame
file_path = ''  # Update this with the path to your CSV file
df = pd.read_csv(file_path, delimiter=',')

# Convert the 'timestamp' column to datetime for proper plotting
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plotting
plt.figure(figsize=(10, 6))

# Plot each negative emotion
negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
for emotion in negative_emotions:
    plt.plot(df['timestamp'], df[emotion], label=emotion)

# Customize the plot
plt.xlabel('Timestamp')
plt.ylabel('Emotion Strength')
plt.title('Negative Emotion Changes Over Time')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
