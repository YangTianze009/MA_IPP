import numpy as np


def analysis_cov_trace():
    cov_trace = np.load("3 agents/rig_tree/cov_budget_3_0.1.npz")
    # cov_trace = cov_trace["arr_0"]
    cov_trace = cov_trace["arr_0"]
    mean = np.mean(cov_trace)
    max_cov_trace = np.max(cov_trace)
    min_cov_trace = np.min(cov_trace)
    std = np.std(cov_trace)
    print(f"cov_trace is {cov_trace}")
    print(f"mean is {mean}")
    print(f"std is {std}")
    print(f"max_cov_trace is {max_cov_trace}")
    print(f"min_cov_trace is {min_cov_trace}")


def analysis_intent_difference_KL():
    KL_divergence = np.load("3 agents/intent_end_nodes/intent_difference_KL/end_nodes(8, 9)_budget3.npz")
    KL_divergence = KL_divergence["arr_0"]
    print(f"KL divergence is {KL_divergence}")
    mean = np.mean(KL_divergence, axis=0)
    max_KL_divergence = np.max(KL_divergence, axis=0)
    min_KL_divergence = np.min(KL_divergence, axis=0)
    std = np.std(KL_divergence, axis=0)
    print(f"mean is {mean}")
    print(f"std is {std}")
    print(f"max KL divergence is {max_KL_divergence}")
    print(f"min KL divergence is {min_KL_divergence}")


if __name__ == "__main__":
    # analysis_intent_difference_KL()
    analysis_cov_trace()