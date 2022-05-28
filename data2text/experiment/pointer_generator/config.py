"""Initialize the config dictionary for varied experiments.

Default model configurations.
"""


# %% tokens
SENTENCE_STA = '<s>'
SENTENCE_END = '</s>'

UNK = 0
PAD = 1
BOS = 2
EOS = 3

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'


# %% model
emb_dim = 128
hidden_dim = 256
vocab_size = 30000

beam_size = 6
max_enc_steps = 512
max_dec_steps = 40
max_tes_steps = 60
min_dec_steps = 8


# batch_size = 64
# lr = 5e-5
cov_loss_wt = 1.0
pointer_gen = True
is_coverage = True

max_grad_norm = 2.0
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
use_gpu = True
# lr_coverage = 5e-5
max_iterations = 500


# %% transformer
tran = False
# d_k = 64
# d_v = 64
# n_head = 6
# dropout = 0.1
# n_layers = 6
# d_model = 128
# d_inner = 512
# n_warmup_steps = 4000