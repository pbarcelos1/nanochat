# nanochat PT-Latn — Guia Resumido

Fork do [karpathy/nanochat](https://github.com/karpathy/nanochat) adaptado para treinar um modelo de linguagem em português. Pipeline completo: pré-treino → mid-training → SFT → avaliação (BLUEX).

**Hardware de referência:** 1× NVIDIA RTX A6000 (48 GB, Ampere SM 86).

---

## Ambiente

```bash
conda env create -f environment.yml
conda activate DL
```

---

## Usando o modelo já treinado (recomendado)

Para pular o treinamento e usar os checkpoints já disponíveis no HuggingFace, basta rodar:

```bash
python -m scripts.download_hf --phase base midtrain sft
```

Isso baixa os três checkpoints (`d12`, `d12_midtrain`, `d12_sft`) e o tokenizador para os diretórios corretos. Após o download, pule direto para a seção de [Avaliação](#avaliação-bluex) ou [Demo Streamlit](#demo-streamlit).

---

## Pipeline de Treinamento

Os scripts de treino ficam em `runs/`. Cada um executa o pipeline completo da etapa correspondente (download de dados, treino, log):

| Script | Etapa | Duração estimada |
|--------|-------|-----------------|
| `runs/entrega1.sh` | Pré-treino no corpus por_Latn (~2 B tokens) | 12–14 h |
| `runs/entrega2_midtrain.sh` | Mid-training no qa-pt (300 M tokens, sem máscara) | 2–4 h |
| `runs/entrega2_sft.sh` | SFT (2 épocas, máscara `assistant_only`) | ~1–2 h |

```bash
bash runs/entrega1.sh
bash runs/entrega2_midtrain.sh
bash runs/entrega2_sft.sh
```

Checkpoints salvos em `base_checkpoints/{d12,d12_midtrain,d12_sft}/`.

### Conceitos principais por etapa

| Etapa | Script Python | Diferencial |
|-------|--------------|-------------|
| **Pré-treino** | `scripts/base_train.py` | AdamW, lei de Chinchilla, atenção full causal |
| **Mid-training** | `scripts/mid_train.py` | Carrega `d12`, LR 10× menor, loss em todos os tokens |
| **SFT** | `scripts/sft.py` | Carrega `d12_midtrain`, LR 10× menor, loss só nos tokens do assistente |

> Para rodar diretamente via `torchrun` (multi-GPU) ou ajustar hiperparâmetros, consulte os próprios `.sh` — cada um tem o comando completo comentado.

---

## Avaliação (BLUEX)

Avalia os checkpoints no benchmark [BLUEX](https://huggingface.co/datasets/Portuguese-NLP/Bluex) — questões de vestibular brasileiro em múltipla escolha (609 questões elegíveis após filtro de texto/imagem).

**Avaliar um checkpoint:**
```bash
python -m scripts.eval \
    --checkpoint d12_sft \
    --task mc \
    --benchmark bluex
```

**Avaliar os três checkpoints e gerar tabela comparativa:**
```bash
bash runs/entrega4_eval_bluex.sh
# Resultado salvo em results_bluex.md
```

| Argumento | Descrição |
|-----------|-----------|
| `--checkpoint` | Tag do checkpoint (`d12`, `d12_midtrain`, `d12_sft`) |
| `--task` | `mc` (múltipla escolha) ou `ppl` (perplexidade) |
| `--benchmark` | `bluex` |
| `--prompt-format` | `auto` (detecta pelo checkpoint), `plain`, ou `chat` |
| `--limit` | Avaliar apenas N questões (útil para testes rápidos) |

**Métricas retornadas:**
- `acc_sum`: Acurácia por soma dos log-probs
- `acc_mean`: Acurácia por média dos log-probs (normaliza pelo comprimento)
- Baseline aleatório: ~0.226

---

## Demo Streamlit

Interface de chat interativa no navegador:

```bash
streamlit run scripts/chat_streamlit.py -- --model-tag d12_sft
```

**Argumentos opcionais:**
| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--model-tag` | `d12_sft` | Checkpoint a carregar |
| `--source` | `base` | `base`, `sft`, ou `rl` |
| `--step` | latest | Passo específico do checkpoint |

A sidebar expõe controles de geração: temperatura (0–2), top-k (0–200) e comprimento máximo (32–1024 tokens). O modelo é carregado uma vez e cacheado na sessão.

> Para usar via linha de comando em vez de navegador: `python -m scripts.chat_cli`

---

## Estrutura de diretórios relevante

```
nanochat/
├── scripts/
│   ├── base_train.py      # Pré-treino
│   ├── mid_train.py       # Mid-training
│   ├── sft.py             # SFT
│   ├── eval.py            # Avaliação BLUEX
│   ├── base_eval.py       # Avaliação BPB / CORE
│   ├── download_hf.py     # Download checkpoints do HuggingFace
│   └── chat_streamlit.py  # Demo Streamlit
├── runs/
│   ├── entrega1.sh            # Pipeline de pré-treino
│   ├── entrega2_midtrain.sh   # Pipeline de mid-training
│   └── entrega4_eval_bluex.sh # Avaliação BLUEX dos 3 checkpoints
├── base_checkpoints/
│   ├── d12/           # Checkpoint do pré-treino
│   ├── d12_midtrain/  # Checkpoint do mid-training
│   └── d12_sft/       # Checkpoint do SFT
├── logs/
└── environment.yml
```
