import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
file_path = '.\output\csv\emotions_data_2023-10-10_12-28-58.csv'  # Update this with the actual path to your CSV file
df = pd.read_csv(file_path, delimiter=',')

# Convert the 'timestamp' column to datetime for proper plotting
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plotting
plt.figure(figsize=(10, 6))

# Plot each emotion
emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
for emotion in emotions:
    plt.plot(df['timestamp'], df[emotion], label=emotion)

# Customize the plot
plt.xlabel('Timestamp')
plt.ylabel('Emotion Strength')
plt.title('Emotion Changes Over Time')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
