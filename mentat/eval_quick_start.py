#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mentat.analysis import analysis_helper_functions

def main():
    results = pd.read_pickle("eval_results_o1-2024-12-17_mentat_data_base_2025_01_26_2031.pkl")
    print(results.keys())
    
    # Accuracy across categories (CE not available for o1)
    analysis_res = analysis_helper_functions.get_acc_crossentropy(results)
    print("Accuracy (95% CL): ", analysis_res[0])

    # For individual categories
    cats = pd.unique(results["category"])
    for c in cats:
        mask = results["category"] == c
        analysis_res = analysis_helper_functions.get_acc_crossentropy(results[mask])
        print(c)
        print("Accuracy (95% CL): ", analysis_res[0])


if __name__ == "__main__":
    main()
