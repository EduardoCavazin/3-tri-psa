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