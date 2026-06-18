"""
Demo Streamlit — NanoChat (Etapa 4).

Carrega o checkpoint final do SFT e expõe uma interface de chat com
parâmetros de geração ajustáveis via sidebar.

Uso:
    streamlit run scripts/chat_streamlit.py
    streamlit run scripts/chat_streamlit.py -- --model-tag d12_sft
    streamlit run scripts/chat_streamlit.py -- --model-tag d12_midtrain --source base
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when streamlit launches the script directly
# (i.e. `streamlit run scripts/chat_streamlit.py` from any working directory)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import streamlit as st

# ---------------------------------------------------------------------------
# Parse args before Streamlit takes over sys.argv

def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source",    type=str, default="base",
                        help="Checkpoint source: base|sft|rl")
    parser.add_argument("--model-tag", type=str, default="d12_sft",
                        help="Checkpoint tag (default: d12_sft)")
    parser.add_argument("--step",      type=int, default=None,
                        help="Specific step to load (-1 / None = latest)")
    # Streamlit passes extra args after '--'; ignore them
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    args, _ = parser.parse_known_args(argv)
    return args


_cli = _parse_args()

# ---------------------------------------------------------------------------
# Model loading (cached across reruns)

@st.cache_resource(show_spinner="Carregando modelo…")
def load_nanochat(source: str, model_tag: str, step):
    from nanochat.common import autodetect_device_type
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine

    device_type = autodetect_device_type()
    device = torch.device("cuda" if device_type == "cuda" else "cpu")

    model, tokenizer, meta = load_model(
        source, device, phase="eval", model_tag=model_tag, step=step
    )
    model.eval()
    engine = Engine(model, tokenizer)

    bos           = tokenizer.get_bos_token_id()
    user_start    = tokenizer.encode_special("<|user_start|>")
    user_end      = tokenizer.encode_special("<|user_end|>")
    asst_start    = tokenizer.encode_special("<|assistant_start|>")
    asst_end      = tokenizer.encode_special("<|assistant_end|>")

    special = dict(
        bos=bos,
        user_start=user_start,
        user_end=user_end,
        asst_start=asst_start,
        asst_end=asst_end,
    )
    return engine, tokenizer, special, meta, device


# ---------------------------------------------------------------------------
# Page config

st.set_page_config(
    page_title="NanoChat",
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar — model info + generation parameters

with st.sidebar:
    st.title("NanoChat")
    st.caption("Small Language Model — PUCRS Deep Learning II")

    st.subheader("Modelo")
    st.write(f"**Tag:** `{_cli.model_tag}`")
    st.write(f"**Source:** `{_cli.source}`")

    st.divider()
    st.subheader("Parâmetros de Geração")

    temperature = st.slider(
        "Temperatura", min_value=0.0, max_value=2.0, value=0.7, step=0.05,
        help="Valores mais altos tornam as respostas mais criativas; 0 = greedy.",
    )
    top_k = st.slider(
        "Top-k", min_value=0, max_value=200, value=50, step=5,
        help="Amostra apenas os k tokens mais prováveis. 0 = sem restrição.",
    )
    max_tokens = st.slider(
        "Máx. tokens", min_value=32, max_value=1024, value=256, step=32,
        help="Número máximo de tokens gerados por resposta.",
    )

    st.divider()
    if st.button("🗑 Limpar conversa", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("conv_tokens", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Load model

try:
    engine, tokenizer, special, meta, device = load_nanochat(
        _cli.source, _cli.model_tag, _cli.step
    )
except Exception as exc:
    st.error(f"Erro ao carregar o modelo: {exc}")
    st.stop()

with st.sidebar:
    cfg = meta.get("model_config", {})
    st.caption(
        f"n_layer={cfg.get('n_layer')}  "
        f"n_embd={cfg.get('n_embd')}  "
        f"step={meta.get('step')}"
    )

# ---------------------------------------------------------------------------
# Session state

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conv_tokens" not in st.session_state:
    st.session_state.conv_tokens = [special["bos"]]

# ---------------------------------------------------------------------------
# Render chat history

st.title("💬 NanoChat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Input + generation

user_input = st.chat_input("Escreva sua mensagem…")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build conversation token sequence
    conv = st.session_state.conv_tokens
    conv.append(special["user_start"])
    conv.extend(tokenizer.encode(user_input))
    conv.append(special["user_end"])
    conv.append(special["asst_start"])

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_tokens = []
        response_text   = ""

        gen_kwargs = dict(
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
        )

        with torch.no_grad():
            for token_col, _ in engine.generate(conv, **gen_kwargs):
                token = token_col[0]
                response_tokens.append(token)
                response_text += tokenizer.decode([token])
                placeholder.markdown(response_text + "▌")

        placeholder.markdown(response_text)

    # Ensure conversation ends with asst_end
    if not response_tokens or response_tokens[-1] != special["asst_end"]:
        response_tokens.append(special["asst_end"])
    conv.extend(response_tokens)

    st.session_state.conv_tokens = conv
    st.session_state.messages.append({"role": "assistant", "content": response_text})
