# log_analyzer.py
# Plot the graph and check the drowsiness in real time. 

import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_sleepiness_log(csv_path):
    """
    Plots Sleepiness Score over time from the given CSV log.
    """
    # Load CSV
    data = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(data)} entries from {csv_path}")

    # Parse timestamps to datetime
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(data['Timestamp'], data['Sleepiness Score'], label="Sleepiness Score", color='blue', linewidth=2)
    plt.scatter(data['Timestamp'], data['Sleepiness Score'], c='blue', s=10)

    plt.title('Driver Sleepiness Score Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Sleepiness Score (%)', fontsize=14)
    plt.ylim(0, 110)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Optional: highlight very sleepy periods
    sleepy_times = data[data['Sleepiness Score'] < 50]
    if not sleepy_times.empty:
        plt.scatter(sleepy_times['Timestamp'], sleepy_times['Sleepiness Score'],
                    c='red', label='Danger Zone (<50%)', s=30)
        plt.legend()

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python log_analyzer.py <path_to_log_csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    plot_sleepiness_log(csv_file)
