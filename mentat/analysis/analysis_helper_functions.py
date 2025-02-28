
import numpy as np
import pandas as pd
from mentat.eval_models import eval_openai
from mentat.pipeline import bootstrap_tools

def check_last_token(results):
    candidate_tokens = eval_openai.candidate_tokens_default
    def modify_row(row):
        # Use last token instead of first token (for models that got the old prompt)
        if all(x == -np.inf for x in row["candidate_logprobs"]):   
            # print(np.array(row["candidate_logprobs"]))
            try:
                alt_answer = row["model_response"].choices[0].message.content[-1]
            except AttributeError:
                alt_answer = row["model_response"].content[0].text[-1]
            candidate_logprobs = eval_openai.get_candidate_logprobs({alt_answer: 1.}, candidate_tokens)

            true_probs = row["labels"]
            model_probs = eval_openai.calculate_model_probs(candidate_logprobs)
            cross_entropy = eval_openai.calculate_cross_entropy(true_probs, model_probs)
            is_correct = eval_openai.check_is_correct(true_probs, model_probs)

            row["candidate_logprobs"] = candidate_logprobs
            row["cross_entropy"] = cross_entropy
            row["is_correct"] = is_correct
        
        return row
    
    def return_frac_not_answered(data):
        logs = data.to_numpy()
        n_data = logs.shape[0]
        count = 0.
        for sample in logs:
            if np.array(sample).max() < 1:
                count += 1
                
        # print(m, cat, count, count / n_data)
        return count / n_data

    for m in results.keys():
        for cat in ["base", "gender", "nat", "age", "cat_gender_nat_age"]:
            data = results[m][cat]    
            f_0 = return_frac_not_answered(data["candidate_logprobs"])
            
            results[m][cat] = data.apply(modify_row, axis=1)
            f_1 = return_frac_not_answered(results[m][cat]["candidate_logprobs"])
            print(m, f"{f_0:0.4f}", f"{f_1:0.4f}")

    return results

def acc_helper(data):
    return data.sum() / data.shape[0]

def get_acc_crossentropy(df: pd.DataFrame, do_boot: bool = True):
    """For a given DF, calculate average accuracy and CE for all rows"""

    res_acc = {}
    res_ce = {}
    acc_vals = df["is_correct"].to_numpy()
    acc = acc_vals.sum() / acc_vals.shape[0]
    ce_vals = df["cross_entropy"].to_numpy()
    ce = ce_vals.sum() / ce_vals.shape[0]

    acc_data = df["is_correct"].to_numpy().reshape(-1, 1)
    if do_boot:
        boot_res_acc = bootstrap_tools.bootstrap_wrap(acc_data, acc_helper, 1000, out_dim=0)
    else:
        acc = acc_helper(acc_data)
        boot_res_acc = {
            "result": acc,
            "ci_lower": acc,
            "ci_upper": acc,
        }

    ce_data = df["cross_entropy"].to_numpy().reshape(-1, 1)
    if do_boot:
        boot_res_ce = bootstrap_tools.bootstrap_wrap(ce_data, acc_helper, 1000, out_dim=0)
    else:
        ce = acc_helper(ce_data)
        boot_res_ce = {
            "result": ce,
            "ci_lower": ce,
            "ci_upper": ce,
        }

    return boot_res_acc, boot_res_ce

def eval_model_by_column(df: pd.DataFrame, model_name: str, column_name: str):
    """"""
    
    res = {}
    res["model_name"] = model_name
    column = pd.unique(df[column_name])
    res["all"] = get_acc_crossentropy(df)
    for c in column:
        mask = df[column_name] == c
        res[c] = get_acc_crossentropy(df[mask])

    return res


def eval_model_by_column_binned(df: pd.DataFrame, model_name: str, column_name: str = "age"):
    """
    Evaluates a model by calculating metrics (accuracy and cross-entropy) for the entire dataset
    and for two specified bins of a given column.
    """
    res = {}
    res["model_name"] = model_name

    # Overall metrics
    res["all"] = get_acc_crossentropy(df)

    # Define bins and labels
    # bins = [18, 41, 65]  # Bin edges
    # labels = ["18-41", "42-65"]
    bins = [18, 33, 49, 65]  # Bin edges
    labels = ["18-33 years", "33-49 years", "49-65 years"]

    # Create a new column with bin labels
    df["age_bin"] = pd.cut(df[column_name], bins=bins, labels=labels, right=True, include_lowest=True)

    # Calculate metrics for each bin
    for bin_label in labels:
        mask = df["age_bin"] == bin_label
        res[bin_label] = get_acc_crossentropy(df[mask])

    return res