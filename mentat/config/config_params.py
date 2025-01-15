
import numpy as np

inds_triage = np.concatenate([range(32, 40), range(73, 83), range(124, 134)])
inds_documentation = np.concatenate([range(84, 94), range(164, 187)])
cols = {
    "triage": "#CC79A7",
    "documentation": "#0072B2",
    "else": "#69995D",
}
# cols = Dict(-1 => "#35393C", 0 => "#E69F00", 1 => "#69995D", 2 => "#CC79A7", 3 => "#0072B2")