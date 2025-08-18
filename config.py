class Config:
    vocab_size = 50258  # GPT-2 BPE vocabulary + 1 for [MASK] token
    block_size = 128    # Now in BPE tokens, not characters (~512 chars)
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.1

    diffusion_steps = 300
    mask_token_id = 255

    batch_size = 64
    learning_rate = 3e-4
    max_iters = 300000
    eval_interval = 500
    eval_iters = 200

    device = 'cuda'
    dtype = 'bfloat16'
