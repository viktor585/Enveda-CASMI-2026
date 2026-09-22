import os

# System Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIN_PARQUET = os.path.join(RAW_DATA_DIR, "train.parquet")
TEST_PARQUET = os.path.join(RAW_DATA_DIR, "test.parquet")
VOCAB_PATH = os.path.join(PROCESSED_DATA_DIR, "selfies_vocab.json")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# Spectral Data Constraints
MAX_PEAKS = 256            # Max peaks retained per spectrum
MAX_SELFIES_LEN = 128      # Max target sequence length

# Transformer Model Hyperparameters
D_MODEL = 512              # Latent vector dimension
N_HEADS = 8                # Multi-head attention heads
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
LABEL_SMOOTHING = 0.1

# Inference / Evaluation Settings
BEAM_SIZE = 25             # Top-25 candidates for competition MRR