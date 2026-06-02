import os
import torch
from huggingface_hub import snapshot_download
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.engine import Engine

# Download repo to local cache (~760 MB, cached after first run)
local_dir = snapshot_download(repo_id="pbarcelos1/nanochat-pt-latn-d12")
os.environ["NANOCHAT_BASE_DIR"] = local_dir  # needed for tokenizer lookup

device = torch.device("cuda:0")
step = find_last_step(local_dir)
model, tokenizer, meta = build_model(local_dir, step, device, phase="eval")

engine = Engine(model, tokenizer)
tokens = tokenizer("Era uma vez, ", prepend="<|bos|>")
samples, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=200, temperature=0.8)
print(tokenizer.decode(samples[0]))
