import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find the most recent CSV log
csv_files = glob.glob('results/emotion_log_*.csv')
if not csv_files:
    print("❌ No emotion log files found!")
    exit()

latest_csv = max(csv_files, key=os.path.getctime)
print(f"📄 Analyzing: {latest_csv}\n")

# Load data
df = pd.DataFrame(pd.read_csv(latest_csv))

# Statistics
print("="*50)
print("EMOTION DETECTION SUMMARY REPORT")
print("="*50)
print(f"\nTotal Detections: {len(df)}")
print(f"Duration: {df['Elapsed Time (s)'].max():.2f} seconds")
print(f"\nEmotion Distribution:")
print(df['Emotion'].value_counts())
print(f"\nAverage Confidence: {df['Confidence (%)'].mean():.2f}%")
print(f"Maximum Confidence: {df['Confidence (%)'].max():.2f}%")
print(f"Minimum Confidence: {df['Confidence (%)'].min():.2f}%")

# Create visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Emotion distribution pie chart
emotion_counts = df['Emotion'].value_counts()
ax1.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
ax1.set_title('Emotion Distribution', fontsize=14, fontweight='bold')

# Confidence over time
ax2.plot(df['Elapsed Time (s)'], df['Confidence (%)'], marker='o', linestyle='-', alpha=0.7)
ax2.set_title('Confidence Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Confidence (%)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/detection_summary.png', dpi=300)
print(f"\n📊 Summary graphs saved to: results/detection_summary.png")
plt.show()

