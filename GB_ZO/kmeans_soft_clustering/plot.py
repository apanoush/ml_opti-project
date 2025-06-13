"""script to plot all additional figures from the report along with computing the results tables"""

import os
import json
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from GB_ZO.utils import LABELS
import numpy as np

DIMENSION = [2, 6][0]
RESULTS_PATH = "GB_ZO/kmeans_soft_clustering/results"
OUTPUT_PATH = f"GB_ZO/kmeans_soft_clustering/results/{DIMENSION}D.pdf"
OUTPUT_TABLE = f"GB_ZO/kmeans_soft_clustering/results/table_{DIMENSION}D.tex"
XLIM = {
    6: (-16, 1600),
    2: (-18, 1800)
}[DIMENSION]
FIGSIZE = (9,3)
LOG = True

def main():
    results = load_results(RESULTS_PATH)

    print(results.keys())

    # verifiying that results have the same parameters
    comparison_columns = ["n_samples", "n_features", "n_clusters"]
    for col in comparison_columns:
        assert results["spsa"][col] == results["multi-point"][col] == results["analytical"][col]

    plot_loss_history(results["spsa"], results["multi-point"], results["analytical"])

    table_results = {
        "SPSA": prepare_table_dict(results["spsa"]["loss_history"]),
        "multi-point": prepare_table_dict(results["multi-point"]["loss_history"]),
        "analytical": prepare_table_dict(results["analytical"]["loss_history"])
    }
    generate_latex_table(table_results, OUTPUT_TABLE)

def load_results(results_path):

    paths = {path.split("_")[0]: os.path.join(results_path, path) for path in os.listdir(results_path) if path.endswith(f"{DIMENSION}D.json")}
    results = {}

    print(paths)

    for name, path in paths.items():
        results[name] = json.load(open(path, "r"))
    
    return results

def plot_loss_history(res1, res2, res3):
    """plots the training loss over iterations"""
    label1 = LABELS.get(res1["method"])
    label2 = LABELS.get(res2["method"])
    label3 = LABELS.get(res3["method"])
    loss_history1 = res1["loss_history"]
    loss_history2 = res2["loss_history"]
    loss_history3 = res3["loss_history"]

    plt.figure(figsize=FIGSIZE)
    plt.plot(loss_history1, label=label1)
    plt.plot(loss_history2, label=label2)
    plt.plot(loss_history3, label=label3)
    plt.xlim(*XLIM)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    title = f'Comparison Training Loss Over Iterations \n {res1["n_samples"]} samples, {res1["n_features"]}D points, {res1["n_clusters"]} clusters'
    if "sparse_dims" in res1:
        title += f', {res1["sparse_dims"]} sparse dimensions'
    #plt.title(title)
    plt.legend()
    plt.grid(True)
    if LOG:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.show()

def prepare_table_dict(results, last_n=10):
    """to be used before using generate_latex_table"""
    results_mean = np.mean(results[:XLIM[-1]][-last_n:])
    
    return {
        'final_result': results_mean,
    }


def generate_latex_table(results, filename, round_digits=4):
    latex_header = r"""\begin{table}[H]
    \centering
    \begin{tabular}{lc}
    \toprule
    \textbf{Algorithm} & \textbf{Loss} \\
    \midrule
    """

    latex_footer = r"""\bottomrule
    \end{tabular}
    \caption{Algorithm performance: average of the last 10 iterations}
    \end{table}"""

    rows = []
    for algo, values in results.items():
        final_val = round(values['final_result'], round_digits)
        algo_escaped = LABELS[algo].replace('_', '\\_')
        rows.append(f"{algo_escaped} & ${final_val}$ \\\\")

    latex_content = latex_header + "\n".join(rows) + "\n" + latex_footer
    
    with open(filename, 'w') as f:
        f.write(latex_content)


if __name__ == "__main__":
    main()