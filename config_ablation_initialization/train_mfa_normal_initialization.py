# Wandb configs
wandb_log = True
wandb_project = 'KV_ablation_initialization'

# Model configs
n_layer = 24
n_head = 24
head_dim = 256
n_embd = 3072
intermediate_size = 8024
share_q_dim = 2048
dropout = 0.0
bias = False

# Training configs
batch_size = 1
block_size = 2048
gradient_accumulation_steps = 60 // batch_size
max_iters = 100000
lr_decay_iters = max_iters
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Optimizer configs
optimizer_name = 'adamw'
learning_rate = 1.6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
min_lr = 0.1 * learning_rate
schedule = 'cosine'

# System configs
compile = True
model_type = 'mfa_normal_initialization'
output_name = f'{model_type}'
#checkpoint_step = 
#resume_dir = f"output/{output_name}/checkpoint-{checkpoint_step}"
#init_from = 'resume'
