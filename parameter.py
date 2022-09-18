USE_INTENT = False
PARTIAL_GP = False
# BATCH_SIZE = 1024 #8
BATCH_SIZE = 256

if USE_INTENT:
    INPUT_DIM = 5

else:
    INPUT_DIM = 4

EMBEDDING_DIM = 128
SAMPLE_SIZE = 200

# training parameters
NUM_META_AGENT = 1  # 3
SAVE_IMAGE = False   # True
GREEDY = True
LOAD_MODEL = True  # default False
GAUSSIAN_NUM = (8, 12)

#
# plot parameters
# NUM_META_AGENT = 10  # 3
# SAVE_IMAGE = False   # True
# GREEDY = True
# LOAD_MODEL = True  # default False
# GAUSSIAN_NUM = (11, 12)


K_SIZE = 20
BUDGET_RANGE = (1.999, 2)
# BUDGET_RANGE = (12, 16)
SAMPLE_LENGTH = 0.2
ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.4
USE_GPU = True
USE_GPU_GLOBAL = True
NUM_GPU = 1
LR = 5e-5
GAMMA = 1
DECAY_STEP = 32
SUMMARY_WINDOW = 6
FOLDER_NAME = 'no_sampling_budget3_diff_graph_0.2_with_agent_input'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
SAVE_IMG_GAP = 1000
NUM_THREADS = 10  # default 3
steps = 256

SAMPLING = True

SAMPLING_TIMES = 1  # 8
SAMPLING_STEPS = 1  # 5
BEST_TRAJ = False

USE_WANDB = True
