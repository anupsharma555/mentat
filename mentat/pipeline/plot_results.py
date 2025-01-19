
import numpy as np
import matplotlib.pyplot as plt

from mentat.config import config_params

# Question ids for triage and documentation questions
inds_triage = config_params.inds_triage
inds_documentation = config_params.inds_documentation
# Color rubric for question categories for consistent plots
cols = config_params.cols

def plot_bt_scores(bt_data, title:str = None, do_save: bool = False, file_name: str = None):
    """"""

    fig, axs = plt.subplots(16, 4, figsize=(12, 30))
    for i in range(16): 
        for j in range(4):
            ax = axs[i, j]
            ax.set_ylim([-0.05, 1.05])
            ax.plot([0., 4.], [0., 0.], c="k", alpha=0.5)
            ax.plot([0., 4.], [1., 1.], c="k", alpha=0.5)
            ax.plot([0., 4.], [0.2, 0.2], c="k", ls="dashed", alpha=0.5)
            if j == 0:
                ax.set_ylabel("BT Probability [ ]")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

    pos_counter = 0
    c_counter = 0
    r_counter = 0
    for q_id in bt_data:
        
        res_mean = bt_data[q_id]["bt_scores"]
        ci_lower = bt_data[q_id]["ci_lower"]
        ci_upper = bt_data[q_id]["ci_upper"]
        if q_id in inds_triage:
            c = cols["triage"]
        elif q_id in inds_documentation:
            c = cols["documentation"]         
        else:
            c = cols["other"]

        y_lower = np.max([np.zeros(res_mean.shape[-1]), np.round(res_mean - ci_lower, decimals=2)], axis=0)
        y_upper = np.max([np.zeros(res_mean.shape[-1]), np.round(ci_upper - res_mean, decimals=2)], axis=0)

        axs[c_counter, r_counter].errorbar(
            ["A0", "A1", "A2", "A3", "A4"],
            res_mean,
            yerr=(y_lower, y_upper),
            fmt=".",
            label=f"Q{int(q_id)}",
            ms=9,
            capsize=3,
            c=c,
        )
        axs[c_counter, r_counter].plot(
            np.argmax(res_mean), 1.0, "k*"
        )
        axs[c_counter, r_counter].legend()
        
        pos_counter += 1
        r_counter += 1
        if pos_counter % 4 == 0 and pos_counter > 0:
            c_counter += 1
            r_counter = 0

    if title is not None:
        fig.suptitle(title)
        # plt.tight_layout()
    
    if do_save:
        if file_name is None:
            file_name = "annotation_results_bt_scores"
        plt.savefig(f"{file_name}.pdf")
        plt.savefig(f"{file_name}.png", dpi=300) 