
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from Excel file
excel_file = 'vehicle_log.xlsx'
df = pd.read_excel(excel_file)

# Convert timestamps to datetime objects
df['Entry Time'] = pd.to_datetime(df['Entry Time'])
df['Exit Time'] = pd.to_datetime(df['Exit Time'])

# Calculate duration of stay in minutes
df['Duration (minutes)'] = (df['Exit Time'] - df['Entry Time']).dt.total_seconds() / 60

# Extract hour of the day and day of the week from entry and exit times
df['Entry Hour'] = df['Entry Time'].dt.hour
df['Exit Hour'] = df['Exit Time'].dt.hour
df['Entry Day'] = df['Entry Time'].dt.day_name()
df['Exit Day'] = df['Exit Time'].dt.day_name()

# Plotting vehicle entry and exit frequencies by hour of the day
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
sns.histplot(df['Entry Hour'], bins=24, kde=False)
plt.title('Vehicle Entry Frequency by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Entries')

plt.subplot(2, 1, 2)
sns.histplot(df['Exit Hour'], bins=24, kde=False)
plt.title('Vehicle Exit Frequency by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Exits')

plt.tight_layout()
plt.show()

# Peak times analysis
entry_peak_hour = df['Entry Hour'].mode()[0]
exit_peak_hour = df['Exit Hour'].mode()[0]

print(f"Peak Entry Hour: {entry_peak_hour}")
print(f"Peak Exit Hour: {exit_peak_hour}")

# Vehicle movement patterns by day of the week and hour of the day
df['Day of Week'] = df['Entry Time'].dt.day_name()

# Group by day of week and hour to find peak movement times
entry_patterns = df.groupby(['Day of Week', 'Entry Hour']).size().reset_index(name='Entries')
exit_patterns = df.groupby(['Day of Week', 'Exit Hour']).size().reset_index(name='Exits')

# Sort days of the week for proper ordering
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
entry_patterns['Day of Week'] = pd.Categorical(entry_patterns['Day of Week'], categories=days_order, ordered=True)
exit_patterns['Day of Week'] = pd.Categorical(exit_patterns['Day of Week'], categories=days_order, ordered=True)

# Pivot tables for heatmap
entry_heatmap_data = entry_patterns.pivot_table(index='Day of Week', columns='Entry Hour', values='Entries', fill_value=0)
exit_heatmap_data = exit_patterns.pivot_table(index='Day of Week', columns='Exit Hour', values='Exits', fill_value=0)

# Plotting vehicle movement patterns
plt.figure(figsize=(14, 14))

plt.subplot(2, 1, 1)
sns.heatmap(entry_heatmap_data, cmap='Blues', annot=True, fmt='.0f')
plt.title('Vehicle Entry Patterns')
plt.xlabel('Hour of the Day')
plt.ylabel('Day of the Week')

plt.subplot(2, 1, 2)
sns.heatmap(exit_heatmap_data, cmap='Reds', annot=True, fmt='.0f')
plt.title('Vehicle Exit Patterns')
plt.xlabel('Hour of the Day')
plt.ylabel('Day of the Week')

plt.tight_layout()
plt.show()

# Additional insights such as average duration of stay
average_duration = df['Duration (minutes)'].mean()
print(f"Average Duration of Stay: {average_duration:.2f} minutes")
