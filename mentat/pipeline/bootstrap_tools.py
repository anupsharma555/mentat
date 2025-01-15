
import numpy as np


# Todo: allow for different CLs
def bootstrap_wrap(data, operator, n_boot: int, out_dim: int = None):
    """Bootstrap wrapper for functions"""

    res = operator(data)
    n_samples = data.shape[0]
    n_dimensions = data.shape[1]
    if out_dim is None:
        bootstrap_means = np.zeros((n_boot, n_dimensions))
    elif out_dim == 0:
        bootstrap_means = np.zeros(n_boot)
    else:
        bootstrap_means = np.zeros((n_boot, out_dim))

    for n in range(n_boot):
        sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = data[sample_indices, :]
        if out_dim == 0:
            bootstrap_means[n] = operator(bootstrap_sample)
        else:
            bootstrap_means[n, :] = operator(bootstrap_sample)
    
    ci_lower = np.percentile(bootstrap_means, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_means, 97.5, axis=0)  

    return {
        "result": res,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
