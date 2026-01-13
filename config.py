import torch
class Config:
    # Data settings
    DATA_DIR = "/home/dima/data/boltz/rcsb_processed_msa/"
    MAX_LENGTH = 512
    # MAX_FILES = 50_000# #npz files to process 
    MAX_FILES = 151040 # (128 * 1180) actual = 151409
    SEQUENCES_PER_FILE = 4  # Number of sequences to extract per file
    MASK_PROB = 0.15  # masking probability
    
    VOCAB_SIZE = 24  # 22 amino acids + MASK + PAD
    D_MODEL = 256
    N_LAYERS = 8 
    N_HEADS = 8
    D_FF = 1024
    DROPOUT = 0.1
    
    BATCH_SIZE = 128
    LEARNING_RATE = 4e-4
    WARMUP_STEPS = 1000
    MAX_EPOCHS = 750 
    TRAIN_SPLIT = 0.8
    
    WEIGHT_DECAY = 0.01
    GRAD_CLIP_NORM = 1.0
    
    LOG_INTERVAL = 100  # Log every N steps
    SAVE_INTERVAL = 1000
    MODEL_DIR = "checkpoints"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "mps"

# Amino acid vocabulary mapping (for reference)
AA_VOCAB = {
    'PAD': 0,   # Padding token
    'A': 1,     # Alanine
    'R': 2,     # Arginine
    'N': 3,     # Asparagine
    'D': 4,     # Aspartic acid
    'C': 5,     # Cysteine
    'Q': 6,     # Glutamine
    'E': 7,     # Glutamic acid
    'G': 8,     # Glycine
    'H': 9,     # Histidine
    'I': 10,    # Isoleucine
    'L': 11,    # Leucine
    'K': 12,    # Lysine
    'M': 13,    # Methionine
    'F': 14,    # Phenylalanine
    'P': 15,    # Proline
    'S': 16,    # Serine
    'T': 17,    # Threonine
    'W': 18,    # Tryptophan
    'Y': 19,    # Tyrosine
    'V': 20,    # Valine
    'X': 21,    # Unknown/non-standard
    'GAP': 22,  # Gap in alignment
    'MASK': 23  # Mask token for MLM
}

VOCAB_TO_AA = {v: k for k, v in AA_VOCAB.items()}
