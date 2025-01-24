
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import krippendorff

from mentat.config import config_params
from mentat.pipeline import preference_tools, preferece_HBT, bootstrap_tools

# Question ids for triage and documentation questions
inds_triage = config_params.inds_triage
inds_documentation = config_params.inds_documentation
# Color rubric for question categories for consistent plots
cols = config_params.cols


def import_raw_annotations(directory: str):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Get annotator information
            rater_id = df.loc[1, 'rater_id'].lower()
            participant_name = df.loc[1, 'participant_name'].lower()
            
            # Remove first four lines and last one
            df = df.iloc[4:-1].reset_index(drop=True)
            # print(rater_id, participant_name)
            df['rater_id'] = rater_id
            df['participant_name'] = participant_name
            dataframes.append(df)

    raw_data = pd.concat(dataframes)

    return raw_data


def annotation_data_check(input_data: pd.DataFrame):
    """"""

    assert np.unique(input_data["q_no"]).shape[-1] == 61, "Some questions not annotated."
    hist_plot = plt.hist(input_data["q_no"], bins=440)
    count_min = np.unique(hist_plot[0])[1]
    count_mean = np.mean(hist_plot[0][hist_plot[0] > 0])
    print(f"Minimum number of answers: {count_min}; Mean: {count_mean}")


def process_raw_data_annotations(input_data: pd.DataFrame, q_key_filter: str = None):
    """Processing data"""

    annotator_individual_data = {}
    # Key = Quesiton ID
    # response_data = {key: [] for key in range(1, 220)}
    response_data = []
    for resp_i, resp in enumerate(input_data["response"]):
        loaded_dict = json.loads(resp)
        cmt = loaded_dict["comment"]
        # print(resp_i, loaded_dict)
        
        # Reverse randomization of answer options
        q_order = json.loads(input_data["question_order"].to_numpy()[resp_i])
        # print("order ", q_order, type(q_order))
        a = np.array(
            [
                loaded_dict["Q0"],
                loaded_dict["Q1"],
                loaded_dict["Q2"],
                loaded_dict["Q3"],
                loaded_dict["Q4"],
            ]
        )
        # print("a (unordered) ", a, type(a))
        a = a[q_order]
        # print("a (  ordered) ", a, type(a))
        
        q_no = input_data["q_no"].to_numpy()[resp_i]
        rater_id = input_data["rater_id"].to_numpy()[resp_i]
        # Type of question phrasing, e.g., male, female, they
        q_key = input_data["q_key"].to_numpy()[resp_i]
        q = input_data["q"].to_numpy()[resp_i]
        
        new_entry = {
            "q_no": int(q_no),
            "rater_id": rater_id,
            "response": a,
            "q_key": q_key,
        }

        try:
            annotator_individual_data[rater_id].append(a)
        except KeyError:
            annotator_individual_data[rater_id] = []
            annotator_individual_data[rater_id].append(a)
        
        # Allow for question filtering
        if q_key_filter is None:
            # response_data[q_no].append(a)
            response_data.append(new_entry)
        elif q_key_filter == q_key:
            # response_data[q_no].append(a)
            response_data.append(new_entry)

        # Going through comments to check for flaws in questions or other issues
        if loaded_dict["comment"] != "":
            # print(raw_data["q_no"][resp_i])
            # print(rater_id, q_no, q_key, cmt)
            pass
        
        # # ToDo: Add this as input parameter  
        # # Optional rescaling or binning of annotation values (Don't use!)
        # if 0:
        #     # Bin response entries
        #     # bin_edges = np.array([0, 20, 40, 60, 80, 101])
        #     bin_edges = np.array([0, 33.3333, 66.6666, 100.0001])
        #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        #     # bin_centers = np.array([0, 100])

        #     # response_data = {key: [bin_centers[(np.digitize(vals, bin_edges, right=False)) - 1] for vals in value] for key, value in response_data.items()}
        #     response_data = {
        #         key: [100. / (1 + np.exp(-(vals - 50) * 0.99)) for vals in value] 
        #         for key, value in response_data.items()
        #     }

    return pd.DataFrame(response_data), annotator_individual_data


# ----------
# Helper functions to be used as input operators for bootstrapping wrapper
# ----------
def calc_mean(data):
    """"""
    return np.mean(data, axis=0)

def calc_alpha(data):
    """"""
    return krippendorff.alpha(
        reliability_data=data, level_of_measurement='interval'
    )

def calc_bt_scores(data):
    """"""

    pairwise_data = preference_tools.pairwise_wins(data)
    bt7274 = preference_tools.BradleyTerry(data=pairwise_data, k=data.shape[-1])

    return bt7274.return_probs()

def calc_hbt_scores(data):
    """"""

    res_HBT = preferece_HBT.main(data)
    hbt_probs = {}
    for i in np.unique(data["q_no"]):
        hbt_probs[i] = res_HBT.get_answer_probabilities(i, method="raw")

    return hbt_probs, res_HBT.get_rater_parameters()


# ----------
# High-level analysis functions to called in scripts
# ----------
def calc_mean_and_alphas(input_data: pd.DataFrame, do_boot: bool = False):
    """"""

    # response_data[response_data["q_no"] == 12]["response"].to_numpy().shape[0]
    res_dict = {}
    n_unique_q_ids = np.unique(input_data["q_no"])
    for k in n_unique_q_ids:
        # res = input_data[k]
        if k in inds_documentation or k in inds_triage:
            # res = np.array(res)
            res = input_data[input_data["q_no"] == k]["response"].to_numpy()
            res = np.vstack(res)
            
            res_mean = bootstrap_tools.bootstrap_wrap(res, calc_mean, n_boot=1000)
            res_alpha = bootstrap_tools.bootstrap_wrap(
                res, calc_alpha, n_boot=1000, out_dim=0
            )

            if do_boot:
                boot_res_mean = bootstrap_tools.bootstrap_wrap(res, calc_mean, n_boot=1000)
                boot_res_alpha = bootstrap_tools.bootstrap_wrap(
                    res, calc_alpha, n_boot=1000, out_dim=0
                )
            else:
                res_mean = calc_mean(res)
                boot_res_mean = {
                    "result": res_mean,
                    "ci_lower": res_mean,
                    "ci_upper": res_mean,
                }
                res_alpha = calc_alpha(res)
                boot_res_alpha = {
                    "result": res_alpha,
                    "ci_lower": res_alpha,
                    "ci_upper": res_alpha,
                }

            res_dict[k] = {
                "res": boot_res_mean["result"],
                "ci_lower": boot_res_mean["ci_lower"],
                "ci_upper": boot_res_mean["ci_upper"],
                "alpha": boot_res_alpha["result"],
                "ci_alpha_lower": boot_res_alpha["ci_lower"],
                "ci_alpha_upper": boot_res_alpha["ci_upper"],
            }

    return res_dict

def calc_preference_probs(input_data: pd.DataFrame, do_boot: bool = False):
    """"""

    res_dict = {}
    n_unique_q_ids = np.unique(input_data["q_no"])
    for k in n_unique_q_ids:
        if k in inds_documentation or k in inds_triage:
            get_q = input_data[input_data["q_no"] == k]
            res = np.vstack(get_q["response"].to_numpy())

            if do_boot:
                boot_res = bootstrap_tools.bootstrap_wrap(res, calc_bt_scores, 100)
            else:
                bt_scores = calc_bt_scores(res)
                boot_res = {
                    "result": bt_scores,
                    "ci_lower": bt_scores,
                    "ci_upper": bt_scores,
                }

            res_dict[k] = {
                "bt_scores": boot_res["result"],
                "ci_lower": boot_res["ci_lower"],
                "ci_upper": boot_res["ci_upper"],
            }
    return res_dict


def calc_hbt_preference_probs(input_data: pd.DataFrame, do_boot: bool = False):
    """"""

    hbt_probs = calc_hbt_scores(input_data)
    boot_hbt_probs = []
    boot_hbt_params = []
    for n in range(100):
        df_boot = input_data.sample(
            n=len(input_data), 
            replace=True, 
        )
        boot_res = calc_hbt_scores(df_boot)
        boot_hbt_probs.append(boot_res[0])
        boot_hbt_params.append(boot_res[1])

    res_dict = {}
    for q_id in hbt_probs[0].keys():

        bootstrap_vals = [b[q_id] for b in boot_hbt_probs]
        ci_lower = np.percentile(bootstrap_vals, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_vals, 97.5, axis=0)  

        print(q_id)
        print(hbt_probs[0][q_id])
        print(ci_lower)
        print(ci_upper)

        res_dict[q_id] = {
            "bt_scores": hbt_probs[0][q_id],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    res_dict_params = {}
    for rater_id in hbt_probs[1].keys():

        bootstrap_vals = [[b[rater_id]["slope"], b[rater_id]["offset"]] for b in boot_hbt_params]
        ci_lower = np.percentile(bootstrap_vals, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_vals, 97.5, axis=0)  

        print(rater_id)
        print(hbt_probs[1][rater_id])
        print(ci_lower)
        print(ci_upper)

        res_dict_params[rater_id] = {
            "bt_scores": hbt_probs[1][rater_id],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return res_dict, res_dict_params


def kl_divergence(p, q):
    """
    Compute Kullback-Leibler divergence KL(p || q) = sum(p_i * log(p_i / q_i))
    Assumes p, q >= 0 and sum(p) = sum(q) = 1.
    We ignore terms where p_i=0 to avoid NaN in 0*log(0/q).
    """
    mask = p != 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def jensen_shannon_divergence(p, q):
    """
    Jensen-Shannon divergence:
      JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
      where m = (p + q) / 2.
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def jensen_shannon_distance(p, q):
    """
    Jensen-Shannon distance is the sqrt of the JSD.
    """
    return np.sqrt(jensen_shannon_divergence(p, q))


def calc_preference_probs_differences(
    input_data_0: pd.DataFrame, input_data_1: pd.DataFrame, do_boot: bool = False
):
    """"""

    # Get answered questions for both data sets
    q_id_overlap = list(set(input_data_0.keys()) & set(input_data_1.keys()))
    diffs = []
    for q_id in q_id_overlap:
        # if q_id in inds_triage:
        #     continue
 
        bt_0 = np.array(input_data_0[q_id]['bt_scores'])
        bt_1 = np.array(input_data_1[q_id]['bt_scores'])
        # print(q_id)
        # print(bt_0)
        # print(bt_1)

        # Change in max probability as in first dataset
        # diff = np.max(bt_0) - bt_1[np.argmax(bt_0)]
        # print(diff)
        # print(np.max(bt_0), bt_1[np.argmax(bt_0)])

        # L1 or Manhatten distance
        # diff = np.sum(np.abs(bt_0 - bt_1))

        # Jensen-Shannon distance
        diff = jensen_shannon_distance(bt_0, bt_1)

        diffs.append([diff])

    res_mean = bootstrap_tools.bootstrap_wrap(
        np.array(diffs), np.mean, n_boot=1000, out_dim=0
    )
    print(f"{res_mean['result']:0.3f} +{res_mean['result']-res_mean['ci_lower']:0.3f} -{res_mean['ci_upper']-res_mean['result']:0.3f}")
    
    return diffs
    

    