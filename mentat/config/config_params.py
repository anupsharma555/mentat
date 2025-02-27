
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


dict_wrong_answer_filter = {
    32: [0],
    33: [0],
    34: [2],
    35: [],
    36: [1, 3],
    37: [],
    38: [0, 1],
    39: [2],
    73: [3],
    74: [0, 4],
    75: [3, 4],
    76: [],
    77: [],
    78: [],
    79: [],
    80: [],
    81: [],
    82: [0, 4],
    84: [0],
    85: [0],
    86: [0],
    87: [0],
    88: [0],
    89: [0],
    90: [0],
    91: [1, 2, 3, 4],
    92: [1, 2, 3, 4],
    93: [1, 2, 3, 4],
    124: [2, 3],
    125: [2],
    126: [],
    127: [3],
    128: [2],
    129: [3],
    130: [4],
    131: [],
    132: [],
    133: [0, 1],
    164: [0, 2],
    165: [0, 2, 3],
    166: [0, 2, 3],
    167: [0, 2],
    168: [0, 2, 3],
    169: [0, 3],
    170: [0, 2, 3],
    171: [0, 2, 3],
    172: [0, 2, 3, 4],
    173: [0, 2, 3, 4],
    174: [0, 2, 3, 4],
    175: [0, 2, 3, 4],
    176: [0, 2, 3, 4],
    177: [],
    178: [],
    179: [0, 2],
    180: [4],
    181: [],
    182: [0, 4],
    183: [],
    184: [],
    185: [],
    186: [4],
}