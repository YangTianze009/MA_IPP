import random
import numpy as np
from parameter import *
import os

cov_trace_final_10 = np.array([1,2,3])
if not os.path.exists("ma_ipp_results/10 agents/destination_intent(8, 3)_0.25"):
    os.makedirs(f"ma_ipp_results/10 agents/destination_intent(8, 3)_0.25")
np.savez(f"ma_ipp_results/10 agents/destination_intent(8, 3)_0.25/diff_graph_budget_2",
         cov_trace_final_10)