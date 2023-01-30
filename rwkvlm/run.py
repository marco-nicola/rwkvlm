"""
Module to run RWKV Language Model.
"""

import gc
import os
import sys
import time

import numpy as np
import torch

from .eval import RWKV
from .utils import TOKENIZER

if len(sys.argv) > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# Step 1: set model & config

# 'cuda' or 'cpu'
DEVICE = 'cuda'

# torch.float16 - torch.half - fp16 (good for GPU, does not work for CPU)
# torch.float32 - torch.float - fp32 (good for CPU)
# torch.bfloat16 - bf16 (less accurate, but works for CPU)
DTYPE = torch.bfloat16

TOKEN_MODE = 'pile'
WORD_NAME = [
    '20B_tokenizer.json',
    '20B_tokenizer.json',
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
VOCAB_SIZE = 50277

# Download Pile models: https://huggingface.co/BlinkDL
# or, set MODEL_NAME to your fine-tuned model

# MODEL_NAME = '.../RWKV-4-Pile-169M-20220807-8023.pth'
# N_LAYER = 12
# N_EMBD = 768

# MODEL_NAME = '.../RWKV-4-Pile-430M-20220808-8066.pth'
# N_LAYER = 24
# N_EMBD = 1024

# MODEL_NAME = '.../rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040.pth'
# N_LAYER = 24
# N_EMBD = 2048

# MODEL_NAME = '.../rwkv-4-pile-3b/RWKV-4-Pile-3B-20221008-8023.pth'
# N_LAYER = 32
# N_EMBD = 2560

# MODEL_NAME = '.../rwkv-4-pile-7b/RWKV-4-Pile-7B-20221115-8047.pth'
# N_LAYER = 32
# N_EMBD = 4096

MODEL_NAME = '/home/m/dev/rwkvlm/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023.pth'

# Step 2: set prompt & sampling stuffs

CONTEXT = '\nIn a shocking finding, scientist discovered a herd of dragons ' \
          'living in a remote, previously unexplored valley, in Tibet. ' \
          'Even more surprising to the researchers was the fact that the ' \
          'dragons spoke perfect Chinese.'

NUM_TRIALS = 999
LENGTH_PER_TRIAL = 333

TEMPERATURE = 1.0
TOP_P = 0.8
TOP_P_NEWLINE = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

###############################################################################

model = torch.jit.script(RWKV(
    model_name=MODEL_NAME,
    dtype=DTYPE,
    device=DEVICE,
))

print('\nOptimizing speed...')
model.forward([187], model.new_state())
gc.collect()
torch.cuda.empty_cache()

# input(0)

print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
if TOKEN_MODE == 'pile':
    assert tokenizer.tokenizer.decode([187]) == '\n'

###############################################################################

if tokenizer.charMode:
    CONTEXT = tokenizer.refine_context(CONTEXT)
    ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in CONTEXT]
else:
    ctx = tokenizer.tokenizer.encode(CONTEXT)
src_len = len(ctx)
src_ctx = ctx.copy()

print('\nYour prompt has ' + str(src_len) + ' tokens.')
print(
    'Note: currently the first run takes a while if your prompt is long, as '
    'we are using RNN to preprocess the prompt. Use GPT to build the hidden '
    'state for better speed.\n'
)

time_slot = {}
time_ref = time.time_ns()


def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


INIT_STATE = model.new_state()
INIT_OUT = None

for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print(('-' * 50) + '\n' + CONTEXT, end='')

    time_ref = time.time_ns()
    ctx = src_ctx.copy()

    if TRIAL == 0:
        for i in range(src_len):
            x = ctx[: i + 1]
            if i == src_len - 1:
                INIT_OUT, INIT_STATE = model.forward(x, INIT_STATE)
            else:
                INIT_STATE = model.forward_preprocess(x, INIT_STATE)
        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    out_last = src_len
    for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
        x = ctx[: i + 1]
        x = x[-model.ctx_len:]

        if i == src_len:
            out = INIT_OUT.clone()
            state = INIT_STATE.clone()
        else:
            out, state = model.forward(x, state)
        if DEBUG_DEBUG:
            print('model', np.array(x), '==>', np.array(out),
                  np.max(out.cpu().numpy()), np.min(out.cpu().numpy()))
        if TOKEN_MODE == 'pile':
            out[0] = -999999999  # disable <|endoftext|>

        ttt = tokenizer.sample_logits(
            out,
            x,
            model.ctx_len,
            temperature=TEMPERATURE,
            top_p_usual=TOP_P,
            top_p_newline=TOP_P_NEWLINE,
        )
        ctx += [ttt]

        if tokenizer.charMode:
            char = tokenizer.itos[ttt]
            print(char, end='', flush=True)
        else:
            char = tokenizer.tokenizer.decode(ctx[out_last:])
            if '\ufffd' not in char:  # is valid utf8 string?
                print(char, end='', flush=True)
                out_last = i + 1

    record_time('total')
    # print(f'\n\n{time_slot}\n\n')
    print(
        f'\n\n--- preprocess {round(time_slot["preprocess"], 2)}s, generation '
        f'{round(time_slot["total"] - time_slot["preprocess"], 2)}s',
        end=' '
    )

print(('-' * 50) + '\n')
