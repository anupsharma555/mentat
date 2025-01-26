
import numpy as np


inds_triage = np.concatenate([range(32, 40), range(73, 83), range(124, 134)])
inds_documentation = np.concatenate([range(84, 94), range(164, 187)])
inds_bad_post_annotate = np.array([85, 87, 91, 92])
train_vs_test_split = 0.1
random_seed_train_test = 11537

cols = {
    "triage": "#CC79A7",
    "documentation": "#0072B2",
    "else": "#69995D",
}
# cols = Dict(-1 => "#35393C", 0 => "#E69F00", 1 => "#69995D", 2 => "#CC79A7", 3 => "#0072B2")

variable_demo_params = {
    "age_range": [18, 65],
    "nat_short": [
        "White", "Asian", "African American", "Black", "Hispanic", "Native American"
    ],
    "nat_long": [
        "caucasian", "european", "canadian", 
        "american indian", "native hawaiian", "alaska native",
        "chinese", "indian", "filipino", "vietnamese", "taiwanese",
        "latino/a", "mexican", "columbian", "spanish", "puerto rican", "cuban", "haitian", "brazilian"
    ],
}

