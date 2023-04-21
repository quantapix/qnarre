# evaluate the base gpt2
# n_layer=12, n_head=12, n_hidden=768
# 124M parameters
batch_size = 8
eval_iters = 500  # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = "gpt2"

# evaluate the base gpt2
# n_layer=24, n_head=16, n_hidden=1024
# 350M parameters
batch_size = 8
eval_iters = 500  # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = "gpt2-medium"

# evaluate the base gpt2
# n_layer=36, n_head=20, n_hidden=1280
# 774M parameters
batch_size = 8
eval_iters = 500  # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = "gpt2-large"

# evaluate the base gpt2
# n_layer=48, n_head=25, n_hidden=1600
# 1558M parameters
batch_size = 8
eval_iters = 500  # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = "gpt2-xl"
