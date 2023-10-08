import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file into a DataFrame
file_path = ''  # Update this with the actual path to your CSV file
df = pd.read_csv(file_path, delimiter=',')

# Convert the 'timestamp' column to datetime for proper plotting
df['timestamp'] = pd.to_datetime(df['timestamp'])

