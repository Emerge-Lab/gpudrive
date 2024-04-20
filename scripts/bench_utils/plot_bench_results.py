from matplotlib import pyplot as plt
import pandas as pd
import argparse

def plot_afps_vs_num_envs():
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Benchmark Results')
    parser.add_argument('--file_path', type=str, help='Path to the benchmark results file', default='benchmark_results.csv', required=False)
    parser.add_argument('--mode', type=int, help='(1) Normal , (2) Randomized, (3) Binned', default=1, required=False)
    parser.add_argument('--x_axis', type=str, help='X-axis label', default='num_envs', required=False)
    parser.add_argument('--y_axis', type=str, help='Y-axis label', default='afps', required=False)
    args = parser.parse_args()
    df = pd.read_csv("benchmark_results.csv")
    df = df.groupby(['actual_num_agents', 'actual_num_roads', 'useful_num_agents', 'useful_num_roads', 'num_envs']).mean().reset_index()
    df = df.sort_values(by='num_envs')
    plt.plot(df['num_envs'], df['time_to_reset'], label='Time to Reset')
    plt.plot(df['num_envs'], df['time_to_step'], label='Time to Step')
    plt.xlabel('Number of Environments')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()