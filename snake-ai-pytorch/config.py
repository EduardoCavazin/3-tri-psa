# Configuration constants for Snake RL training

# Epsilon (exploration) parameters
EPS0 = 1.0
DECAY = 0.995
EPS_MIN = 0.01

# Game timeout
TIMEOUT_FACTOR = 100

# Checkpointing
CHECKPOINT_EVERY = 100

# Evaluation
EVAL_EPISODES_DEFAULT = 50

# Seeds for reproducibility
SEED = 42

# Model parameters (keeping existing values)
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Advanced training monitoring flags
ENABLE_GRAD_NORM = True
ENABLE_CLIP = True
GRAD_CLIP_MAX_NORM = 10.0

# Experiment configurations for ablation studies
EXPERIMENT_CONFIGS = [
    {"DECAY": 0.995, "TIMEOUT_FACTOR": 100, "gamma": 0.90},
    {"DECAY": 0.997, "TIMEOUT_FACTOR":  50, "gamma": 0.95},
    {"DECAY": 0.993, "TIMEOUT_FACTOR": 150, "gamma": 0.99},
]

# Default episodes for experiments (can be overridden by CLI)
EXPERIMENT_EPISODES = 500

# Training control
MAX_EPISODES = None  # Will be set by experiment runner or default training

# Reward shaping flags
ENABLE_SHAPING = False
REWARD_SURVIVE = 0.10
REWARD_APPROACH = 1.00
PENALTY_DETOUR = -1.00