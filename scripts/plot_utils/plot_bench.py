import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def save_plot(output_path: str):
    output_dir = "plots"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    full_output_path = os.path.join(output_dir, output_path)
    
    plt.tight_layout()
    plt.savefig(full_output_path)

def plot_afps_curves(results_path: str, output_path: str):
    benchmark_data = pd.read_csv(results_path)

    # Using a different color palette and context
    sns.set_theme(style="darkgrid", palette="muted", context="talk")

    # Create a more refined and visually appealing plot
    plt.figure(figsize=(12, 7))

    # Create the enhanced plot
    sns.lineplot(data=benchmark_data, x='num_envs', y='actual_afps', label='Actual AFPS')
    sns.lineplot(data=benchmark_data, x='num_envs', y='useful_afps', label='Useful AFPS')

    plt.title('AFPS vs Number of Environments', fontsize=20)
    plt.xlabel('Number of Environments', fontsize=16)
    plt.ylabel('AFPS (Agent adjusted Frames Per Second)', fontsize=16)

    plt.grid(True)

    plt.legend(title='AFPS Trend', title_fontsize='13', fontsize='12', loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)

def plot_afps_vs_agents(results_path: str, output_path: str):
    benchmark_data = pd.read_csv(results_path)
    #select columns
    benchmark_data = benchmark_data[['useful_num_roads', 'useful_afps']]
    # groupby useful_agents and take the mean of useful_afps
    benchmark_data = benchmark_data.groupby('useful_num_roads').mean().reset_index()

    # Using a different color palette and context
    sns.set_theme(style="darkgrid", palette="muted", context="talk")

    # Create a more refined and visually appealing plot
    plt.figure(figsize=(12, 7))

    # Create the enhanced plot
    sns.lineplot(data=benchmark_data, x='useful_num_roads', y='useful_afps', label='AFPS')

    plt.title('AFPS vs Number of Agents (100 Worlds)', fontsize=20)
    plt.xlabel('Number of Road Segments', fontsize=16)
    plt.ylabel('AFPS (Agent adjusted Frames Per Second)', fontsize=16)

    plt.grid(True)

    plt.legend(title='', title_fontsize='13', fontsize='12', loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)

def plot_randomized(results_path: str, output_path: str):
    benchmark_data = pd.read_csv(results_path)
    #select columns
    benchmark_data = benchmark_data[['num_envs', 'useful_afps', 'actual_afps']]
    benchmark_data = benchmark_data.groupby('num_envs').mean().reset_index()

    # Using a different color palette and context
    sns.set_theme(style="darkgrid", palette="muted", context="talk")

    # Create a more refined and visually appealing plot
    plt.figure(figsize=(12, 7))

    # Create the enhanced plot
    sns.lineplot(data=benchmark_data, x='num_envs', y='actual_afps', label='Actual AFPS')
    sns.lineplot(data=benchmark_data, x='num_envs', y='useful_afps', label='Useful AFPS')

    plt.title('AFPS vs Number of Environments (Randomized Sampling)', fontsize=20)
    plt.xlabel('Number of Environments', fontsize=16)
    plt.ylabel('AFPS (Agent adjusted Frames Per Second)', fontsize=16)

    plt.grid(True)

    plt.legend(title='AFPS Trend', title_fontsize='13', fontsize='12', loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPUDrive Plotting tool for benchmarking results')
    parser.add_argument('--benchmarkResults', type=str, help='Path to the benchmark results', default='benchmark_results.csv', required=False)
    parser.add_argument('--benchmarkType', type=str, help='Type of benchmarking results', default='afps', required=False)
    parser.add_argument('--outputPath', type=str, help='Path to save the plot', default='benchmark_plot.png', required=False)
    args = parser.parse_args()
    # plot_afps_curves(args.benchmarkResults, args.outputPath)
    # plot_afps_vs_agents(args.benchmarkResults, args.outputPath)
    plot_randomized(args.benchmarkResults, args.outputPath)