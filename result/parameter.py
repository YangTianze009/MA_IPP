# BATCH_SIZE = 1024 #8
BATCH_SIZE = 256
INPUT_DIM = 4
EMBEDDING_DIM = 128
SAMPLE_SIZE = 200

# training parameters
# NUM_META_AGENT = 8  # 3
# SAVE_IMAGE = False # True
# GREEDY = False
# LOAD_MODEL = False # default False

# plot parameters
NUM_META_AGENT = 1  # 3
SAVE_IMAGE = True # True
GREEDY = True
LOAD_MODEL = True # default False


K_SIZE = 20
BUDGET_RANGE = (3.999, 4)
# BUDGET_RANGE = (12, 16)
SAMPLE_LENGTH = 0.2
ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.4
USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 1
LR = 1e-4
GAMMA = 1
DECAY_STEP = 32
SUMMARY_WINDOW = 8
FOLDER_NAME = 'ipp'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
SAVE_IMG_GAP = 1000
NUM_THREADS = 3  # default 3
steps = 256
USE_WANDB = True
