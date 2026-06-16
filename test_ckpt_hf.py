import os
import argparse
import torch
from huggingface_hub import snapshot_download
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.engine import Engine

REPOS = {
    "base":     "pbarcelos1/nanochat-pt-latn-d12",
    "midtrain": "pbarcelos1/nanochat-pt-latn-d12-midtrain",
    "sft":      "pbarcelos1/nanochat-pt-latn-d12-sft",
}

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=list(REPOS), default="base")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--prompt", default="Era uma vez, ")
parser.add_argument("--max-tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.8)
args = parser.parse_args()

repo_id = REPOS[args.phase]
print(f"Downloading {repo_id} …")
local_dir = snapshot_download(repo_id=repo_id)
os.environ["NANOCHAT_BASE_DIR"] = local_dir

device = torch.device(args.device)
step = find_last_step(local_dir)
model, tokenizer, meta = build_model(local_dir, step, device, phase="eval")

engine = Engine(model, tokenizer)
tokens = tokenizer(args.prompt, prepend="<|bos|>")
samples, _ = engine.generate_batch(
    tokens, num_samples=1, max_tokens=args.max_tokens, temperature=args.temperature
)
print(tokenizer.decode(samples[0]))
