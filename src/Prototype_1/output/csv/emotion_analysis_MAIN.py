import pandas as pd
import matplotlib.pyplot as plt

# Read data from the local CSV file
df = pd.read_csv('src\\Prototype_1\\output\\csv\\emotions_data_2024-01-03_18-15-02.csv', parse_dates=['timestamp'])

# Define the sets for negative, positive, and neutral emotions for simplicity
negative_emotions = {'contempt', 'anger', 'sadness', 'fear'}
positive_emotions = {'happy'}
neutral_emotions = {'surprise'}

# Identifying the start of negative emotions
df['previous_emotion'] = df['emotion'].shift(1)
df['start_negative'] = df.apply(lambda row: row['emotion'] in negative_emotions and
                                 (row['previous_emotion'] in positive_emotions or 
                                  row['previous_emotion'] in neutral_emotions or 
                                  pd.isna(row['previous_emotion'])), axis=1)

# Plotting
plt.figure(figsize=(12, 6))
colors = {'contempt':'red', 'anger':'red', 'sadness':'red', 'fear':'red', 
          'happy':'green', 'surprise':'blue'}

# Plot all emotions
for emotion, group in df.groupby('emotion'):
    plt.plot(group['timestamp'], group['emotion'], marker='o', linestyle='', 
             color=colors[emotion], label=emotion)

# Highlight the start of negative emotions
start_negatives = df[df['start_negative']]
plt.plot(start_negatives['timestamp'], start_negatives['emotion'], 
         marker='X', linestyle='', color='black', label='Start of Negative', markersize=10)

plt.xlabel('Timestamp')
plt.ylabel('Emotion')
plt.title('Emotion Changes Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
