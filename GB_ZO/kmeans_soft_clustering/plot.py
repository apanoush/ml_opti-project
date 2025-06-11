import os
import json
import matplotlib.pyplot as plt

DIMENSION = 2
RESULTS_PATH = "GB_ZO/kmeans_soft_clustering/results"
OUTPUT_PATH = f"GB_ZO/kmeans_soft_clustering/results/{DIMENSION}D.pdf"
XLIM = {
    6: (0, 800),
    2: (0, 1200)
}[DIMENSION]


def main():
    results = load_results(RESULTS_PATH)

    print(results.keys())

    # verifiying that results have the same parameters
    comparison_columns = ["n_samples", "n_features", "n_clusters"]
    for col in comparison_columns:
        assert results["spsa"][col] == results["multi-point"][col]

    plot_loss_history(results["spsa"], results["multi-point"])

def load_results(results_path):

    paths = {path.split("_")[0]: os.path.join(results_path, path) for path in os.listdir(results_path) if path.endswith(f"{DIMENSION}D.json")}
    results = {}

    print(paths)

    for name, path in paths.items():
        results[name] = json.load(open(path, "r"))
    
    return results

def plot_loss_history(res1, res2):
    """Plots the training loss over iterations."""
    label1 = res1["method"]
    label2 = res2["method"]
    loss_history1 = res1["loss_history"]
    loss_history2 = res2["loss_history"]

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history1, label=label1)
    plt.plot(loss_history2, label=label2)
    plt.xlim(*XLIM)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    title = f'Comparison Training Loss Over Iterations \n {res1["n_samples"]} samples, {res1["n_features"]}D points, {res1["n_clusters"]} clusters'
    if "sparse_dims" in res1:
        title += f', {res1["sparse_dims"]} sparse dimensions'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_PATH)
    plt.show()

if __name__ == "__main__":
    main()