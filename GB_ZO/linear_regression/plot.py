"""script to plot all additional figures from the report along with computing the results tables"""

import os
import json
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from GB_ZO.utils import LABELS
import numpy as np

DIMENSION = [1, 10][0]
RESULTS_PATH = "GB_ZO/linear_regression/results"
OUTPUT_PATH = f"GB_ZO/linear_regression/results/{DIMENSION}D.pdf"
OUTPUT_TABLE = f"GB_ZO/linear_regression/results/table_{DIMENSION}D.tex"
LOG = True
XLIM1 = {
    10: (-3, 300),
    1: (-0.4, 40)
}[DIMENSION]

XLIM2 = {
    10: (-80, 8000),
    1: (-8, 800)
}[DIMENSION]
FIGSIZE = (11, 3)

def main():
    results = load_results(RESULTS_PATH)

    print(results.keys())

    # verifiying that results have the same parameters
    comparison_columns = ["n_samples", "n_features", "sparse_dims"]
    for col in comparison_columns:
        assert results["spsa"][col] == results["multi-point"][col] == results["analytical"][col]

    plot_loss_history(results["spsa"], results["multi-point"], results["analytical"], log=LOG)

    table_results = {
        "SPSA": prepare_table_dict(results["spsa"]["loss_history"], results["spsa"]["param_error"]),
        "multi-point": prepare_table_dict(results["multi-point"]["loss_history"], results["multi-point"]["param_error"]),
        "analytical": prepare_table_dict(results["analytical"]["loss_history"], results["analytical"]["param_error"]),
    }
    generate_latex_table(table_results, OUTPUT_TABLE)

def load_results(results_path):

    paths = {path.split("_")[0]: os.path.join(results_path, path) for path in os.listdir(results_path) if path.endswith(f"{DIMENSION}D.json")}
    results = {}

    print(paths)

    for name, path in paths.items():
        results[name] = json.load(open(path, "r"))
    
    return results

def plot_loss_history(res1, res2, res3=None, log=False):
    """plots the training loss and parameter error over iterations"""
    label1 = LABELS.get(res1["method"])
    label2 = LABELS.get(res2["method"])
    loss_history1 = res1["loss_history"]
    loss_history2 = res2["loss_history"]
    param_error1 = res1.get("param_error", [])
    param_error2 = res2.get("param_error", [])
    
    if res3:
        label3 = LABELS.get(res3["method"])
        loss_history3 = res3["loss_history"]
        param_error3 = res3.get("param_error", [])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    ax1.plot(loss_history1, label=label1)
    ax1.plot(loss_history2, label=label2)
    if res3:
        ax1.plot(loss_history3, label=label3)
    if log: ax1.set_yscale('log')
    ax1.set_xlim(*XLIM1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Loss Convergence')
    ax1.grid(True)

    if param_error1:
        ax2.plot(param_error1)#, label=label1)
    if param_error2:
        ax2.plot(param_error2)#, label=label2)
    if res3 and param_error3:
        ax2.plot(param_error3)#, label=label3)
    if log: ax2.set_yscale('log')
    ax2.set_xlim(*XLIM2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Error (L2 Norm)')
    ax2.set_title('Distance to True Parameters')
    ax2.grid(True)
    
    fig.legend(loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(OUTPUT_PATH)
    plt.show()


def prepare_table_dict(results, parameter_error, last_n=10):
    """to be used before using generate_latex_table"""

    results_mean = np.mean(results[:XLIM2[-1]][-last_n:])
    pe_mean = np.mean(parameter_error[:XLIM2[-1]][-last_n:])
    
    return {
        'final_result': results_mean,
        'parameter_error': pe_mean
    }


def generate_latex_table(results, filename, round_digits=4):
    latex_header = r"""\begin{table}[H]
    \centering
    \begin{tabular}{lcc}
    \toprule
    \textbf{Algorithm} & \textbf{Loss} & \textbf{Parameter error} \\
    \midrule
    """

    latex_footer = r"""\bottomrule
    \end{tabular}
    \caption{Algorithm performance: average of the last 10 iterations}
    \end{table}"""

    rows = []
    for algo, values in results.items():
        final_val = round(values['final_result'], round_digits)
        
        cov_val = values['parameter_error']
        cov_str = f"{cov_val:.{round_digits}f}"        
        algo_escaped = LABELS[algo].replace('_', '\\_')
        
        rows.append(f"{algo_escaped} & ${final_val}$ & ${cov_str}$ \\\\")

    latex_content = latex_header + "\n".join(rows) + "\n" + latex_footer
    
    with open(filename, 'w') as f:
        f.write(latex_content)


if __name__ == "__main__":
    main()