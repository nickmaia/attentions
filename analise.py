#!/usr/bin/env python3
"""
Inferência Comparativa — Mecanismos de Atenção
===============================================
Carrega cada modelo treinado (descoberta dinâmica de data/),
avalia no test set e gera tabela, gráficos e amostras.

Uso: python inference_comparison.py
"""

import json, math, time, sys
from pathlib import Path

import torch, torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE = Path("data")
CKPT_FILE = "models/model_best.pt"

# ── Hyperparâmetros (fixos) ───────────────────────────────────────────────────
VOCAB_SIZE = 65_000
MAX_LEN    = 128
D_MODEL    = 512
HEAD_DIM   = 64
NUM_HEADS  = 8
FF_RATIO   = 4
NUM_ENC    = 6
NUM_DEC    = 6
DROPOUT    = 0.1
BATCH_SIZE     = 8
SEQ_LEN        = 128
MAX_NEW_TOKENS = 50
BLEU_MAX_BATCHES = 20

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path("model").resolve()))
from transformer import Transformer
from tokens import tokenizer, DataLoader, TranslationDataset, tokenize
from train import load_test_dataset, compute_bleu_methods, perplexity_fn, generate_batch
from model.attention_dense import AttentionDense
from model.attention_factorized_random_synthesizer import AttentionFactorizedRandomSynthesizer
from model.attention_factorized_dense_synthesizer import AttentionFactorizedDenseSynthesizer
from model.attention_luong_general import AttentionLuongGeneral

# ══════════════════════════════════════════════════════════════════════════════
# 0. DESCOBERTA DINÂMICA DOS EXPERIMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def discover_experiments():
    """
    Varre data/ em busca de {noise}/epoch_{N}/{mecanismo}/models/model_best.pt
    Retorna lista de dicts: {id, display, path, noise, epochs, mechanism}
    """
    exps = []
    for noise_dir in sorted(BASE.iterdir()):
        if not noise_dir.is_dir() or noise_dir.name == "artigos":
            continue
        noise_label = noise_dir.name  # ex: "sem_ruido", "ruido_std_0_10"
        for epoch_dir in sorted(noise_dir.iterdir()):
            if not epoch_dir.is_dir() or not epoch_dir.name.startswith("epoch_"):
                continue
            epochs = epoch_dir.name.replace("epoch_", "")
            for mech_dir in sorted(epoch_dir.iterdir()):
                if not mech_dir.is_dir():
                    continue
                ckpt = mech_dir / CKPT_FILE
                if not ckpt.exists():
                    continue
                mechanism = mech_dir.name  # ex: "dot_product", "dense"
                exp_id = f"{noise_label}_{epochs}_{mechanism}"
                display = _make_display(noise_label, epochs, mechanism)
                exps.append({
                    "id": exp_id,
                    "display": display,
                    "path": mech_dir.relative_to(BASE),
                    "noise": noise_label,
                    "epochs": int(epochs),
                    "mechanism": mechanism,
                    "noise_level": _parse_noise(noise_label),
                })
    return exps

def _parse_noise(label):
    if label == "sem_ruido":
        return 0.0
    return float(label.replace("ruido_std_", "").replace("_", "."))

def _make_display(noise, epochs, mechanism):
    name_map = {
        "dot_product": "Dot-Product",
        "dense": "Dense Synth",
        "factorized_dense": "Fact Dense Synth",
        "factorized_random": "Fact Random Synth",
        "luong_general": "Luong General",
    }
    mech_name = name_map.get(mechanism, mechanism)
    if noise == "sem_ruido":
        return f"{mech_name} (sem ruido, {epochs}ep)"
    nl = noise.replace("ruido_std_", "").replace("_", ".")
    return f"{mech_name} (ruido={nl}, {epochs}ep)"

def build_attn_config(mechanism, epochs):
    """Retorna attn_cls e attn_kwargs para o mecanismo."""
    cfg = {"attn_cls": None, "attn_kwargs": {}}
    if mechanism == "dense":
        cfg["attn_cls"] = AttentionDense
        cfg["attn_kwargs"] = {"max_seq_len": MAX_LEN}
    elif mechanism == "factorized_dense":
        cfg["attn_cls"] = AttentionFactorizedDenseSynthesizer
        cfg["attn_kwargs"] = {"max_seq_len": MAX_LEN}
    elif mechanism == "factorized_random":
        cfg["attn_cls"] = AttentionFactorizedRandomSynthesizer
        cfg["attn_kwargs"] = {"max_seq_len": MAX_LEN, "rank": 8}
    elif mechanism == "luong_general":
        cfg["attn_cls"] = AttentionLuongGeneral
        cfg["attn_kwargs"] = {}
    # dot_product usa None (default no Transformer)
    return cfg

EXPERIMENTS = discover_experiments()
print(f"\nExperimentios encontrados: {len(EXPERIMENTS)}")
for e in EXPERIMENTS:
    print(f"  {e['id']:<45} → {e['display']}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DADOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Carregando datasets de teste...")
print("=" * 60)

test_ds = load_test_dataset()
test_ids = tokenize(test_ds, split_name="test", max_len=MAX_LEN)
test_dataset = TranslationDataset(test_ids, SEQ_LEN)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(device == "cuda"), drop_last=False,
)
print(f"Test samples: {len(test_dataset)}")
print(f"Test batches: {len(test_loader)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. INFERÊNCIA
# ══════════════════════════════════════════════════════════════════════════════
results = []

for exp in EXPERIMENTS:
    print("\n" + "=" * 60)
    print(f"Evaluating: {exp['display']} ({exp['id']})")
    print("=" * 60)

    ckpt_path = BASE / exp["path"] / CKPT_FILE
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint não encontrado: {ckpt_path}")
        results.append({"name": exp["id"], "display": exp["display"], "error": "checkpoint_not_found"})
        continue

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    attn_cfg = build_attn_config(exp["mechanism"], exp["epochs"])

    model = Transformer(
        vocab_size=VOCAB_SIZE, max_len=MAX_LEN, d_model=D_MODEL,
        head_dim=HEAD_DIM, num_heads=NUM_HEADS, ff_ratio=FF_RATIO,
        num_encoder_layers=NUM_ENC, num_decoder_layers=NUM_DEC,
        mask=True, dropout=DROPOUT, tie_embeddings=True,
        attn_cls=attn_cfg["attn_cls"], attn_kwargs=attn_cfg["attn_kwargs"],
    )

    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing: print(f"  Missing keys: {missing}")
        if unexpected: print(f"  Unexpected keys: {unexpected}")
    except Exception as e:
        print(f"  [ERROR] Falha ao carregar state_dict: {e}")
        results.append({"name": exp["id"], "display": exp["display"], "error": str(e)})
        continue

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros: {total_params:,} ({total_params/1e6:.2f}M)")

    # Tempo de treino
    hist_path = BASE / exp["path"] / "history/training_history.pt"
    train_time_hours = 0.0
    if hist_path.exists():
        try:
            hist_data = torch.load(hist_path, map_location="cpu", weights_only=False)
            epoch_times = hist_data.get("timing", {}).get("epoch_time_history_sec", [])
            if epoch_times:
                train_time_hours = sum(epoch_times) / 3600
        except Exception:
            pass
    print(f"  Tempo treino: {train_time_hours:.2f}h")

    # Loss + BLEU
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    test_loss_total = 0.0
    test_steps = 0
    test_all_preds, test_all_targets, test_all_sources = [], [], []

    inference_start = time.perf_counter()
    with torch.inference_mode():
        test_bar = tqdm(test_loader, total=min(BLEU_MAX_BATCHES, len(test_loader)),
                        desc=f"Test {exp['display'][:20]}", unit="batch")
        for step, (X_test, y_test) in enumerate(test_bar):
            if step >= BLEU_MAX_BATCHES: break
            X_test = X_test.to(device).long()
            y_test = y_test.to(device).long()
            tgt_input = y_test[:, :-1]
            tgt_output = y_test[:, 1:]
            logits = model(X_test, tgt_input)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
            test_loss_total += loss.item()
            test_steps += 1
            pred_ids = generate_batch(model, X_test, tokenizer, max_len=MAX_NEW_TOKENS)
            test_all_preds.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
            test_all_targets.extend(tokenizer.batch_decode(y_test, skip_special_tokens=True))
            test_all_sources.extend(tokenizer.batch_decode(X_test, skip_special_tokens=True))
            test_bar.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{perplexity_fn(loss.item()):.2f}")

    inference_time = time.perf_counter() - inference_start
    test_loss = test_loss_total / max(test_steps, 1)
    test_ppl = perplexity_fn(test_loss)
    sacre = compute_bleu_methods(test_all_preds, test_all_targets) if test_all_preds else float("nan")

    samples = []
    for i in range(min(5, len(test_all_preds))):
        samples.append({
            "source": test_all_sources[i] if i < len(test_all_sources) else "",
            "reference": test_all_targets[i], "hypothesis": test_all_preds[i],
        })

    print(f"  Test loss: {test_loss:.5f}")
    print(f"  Test ppl:  {test_ppl:.2f}")
    print(f"  SacreBLEU: {sacre:.2f}")
    print(f"  Tempo:     {inference_time:.2f}s")

    r = {
        "name": exp["id"], "display": exp["display"],
        "test_loss": float(f"{test_loss:.5f}"), "test_ppl": float(f"{test_ppl:.2f}"),
        "bleu": float(f"{sacre:.2f}"), "params_millions": round(total_params / 1e6, 2),
        "inference_time_sec": round(inference_time, 2), "train_time_hours": round(train_time_hours, 2),
        "samples": samples, "error": None,
        "mechanism": exp["mechanism"], "epochs": exp["epochs"],
        "noise": exp["noise"], "noise_level": exp["noise_level"],
    }
    results.append(r)
    if device == "cuda": torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# 3. TABELA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTADOS COMPARATIVOS")
print("=" * 60)

header = f"{'Mecanismo':<40} {'Params (M)':<12} {'Loss':<10} {'PPL':<10} {'SacreBLEU':<12} {'Temp (s)':<10}"
sep = "-" * len(header)
print(sep)
print(header)
print(sep)

for r in results:
    if r.get("error"):
        print(f"{r['display']:<40} {'ERROR':<12} {r['error']:<48}")
    else:
        print(f"{r['display']:<40} {r['params_millions']:<11.2f}M {r['test_loss']:<10.5f} {r['test_ppl']:<10.2f} {r['bleu']:<12.2f} {r['inference_time_sec']:<10.2f}")
print(sep)

OUTPUT_DIR = Path("resultados")
OUTPUT_DIR.mkdir(exist_ok=True)
json_path = OUTPUT_DIR / "results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResultados salvos em: {json_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. GRÁFICOS GLOBAIS
# ══════════════════════════════════════════════════════════════════════════════
valid_results = [r for r in results if not r.get("error")]
inf_map = {r["name"]: r for r in valid_results}
names = [r["display"] for r in valid_results]
x = np.arange(len(names))

# ── BLEU (separado 30ep e 100ep, agrupado por mecanismo) ──
for ep_target in sorted(set(r.get("epochs", 0) for r in valid_results)):
    ep_results = [r for r in valid_results if r.get("epochs") == ep_target]
    if not ep_results:
        continue
    # Agrupa por mecanismo
    mech_groups = {}
    for r in ep_results:
        m = r.get("mechanism", "unknown")
        mech_groups.setdefault(m, []).append(r)

    mechanisms = sorted(mech_groups.keys())
    # Níveis de ruído presentes
    all_noise = sorted(set(r.get("noise", "") for r in ep_results))
    noise_display_map = {nl: ("sem ruido" if nl == "sem_ruido"
                               else f"ruido {nl.replace('ruido_std_', '').replace('_', '.')}") for nl in all_noise}
    paleta_noise_bar = {"sem_ruido": "#4C72B0", "ruido_std_0_01": "#55A868",
                        "ruido_std_0_05": "#DD8452", "ruido_std_0_10": "#C44E52"}

    fig, ax = plt.subplots(figsize=(10, 5))
    xg = np.arange(len(mechanisms))
    n_niveis_b = len(all_noise)
    largura_b = 0.7 / max(n_niveis_b, 1)

    for i, nl in enumerate(all_noise):
        bleus_nl = []
        for m in mechanisms:
            match = [r for r in mech_groups[m] if r.get("noise") == nl]
            bleus_nl.append(match[0]["bleu"] if match else 0)
        offset = (i - (n_niveis_b - 1) / 2) * largura_b
        bars = ax.bar(xg + offset, bleus_nl, largura_b,
                      label=noise_display_map.get(nl, nl),
                      color=paleta_noise_bar.get(nl, "#999"),
                      edgecolor="black", linewidth=0.3)
        for bar, v in zip(bars, bleus_nl):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(xg); ax.set_xticklabels(mechanisms, fontsize=11)
    ax.set_ylabel("SacreBLEU", fontsize=11)
    ax.set_title(f"SacreBLEU por Mecanismo ({ep_target} épocas)", fontsize=13)
    ax.legend(title="Nível de ruído", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"bleu_comparison_{ep_target}ep.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot salvo: bleu_comparison_{ep_target}ep.png")

# ── Loss / PPL ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.bar(names, [r["test_loss"] for r in valid_results], color="#4C72B0", alpha=0.85)
ax1.set_title("Test Loss"); ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax1.grid(axis="y", alpha=0.3)
ax2.bar(names, [r["test_ppl"] for r in valid_results], color="#DD8452", alpha=0.85)
ax2.set_title("Perplexidade (PPL)"); ax2.set_ylabel("PPL")
ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_ppl_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ── Params vs BLEU (separado 30ep e 100ep) ──
for ep_target in sorted(set(r.get("epochs", 0) for r in valid_results)):
    ep_results = [r for r in valid_results if r.get("epochs") == ep_target]
    if not ep_results:
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    cores_exp = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860", "#8172B2", "#E377C2", "#8C564E"]
    for i, r in enumerate(ep_results):
        nl = r.get("noise", "")
        nl_label = "sem ruido" if nl == "sem_ruido" else f"ruido {nl.replace('ruido_std_', '').replace('_', '.')}"
        label = f"{r['display']}"
        ax.scatter(r["params_millions"], r["bleu"], s=120, c=cores_exp[i % len(cores_exp)],
                   label=label, zorder=3, edgecolors="black", linewidth=0.5)
        ax.annotate(nl_label, (r["params_millions"], r["bleu"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)
    ax.set_xlabel("Parâmetros (M)"); ax.set_ylabel("SacreBLEU")
    ax.set_title(f"Parâmetros vs. BLEU ({ep_target} épocas)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"params_vs_bleu_{ep_target}ep.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot salvo: params_vs_bleu_{ep_target}ep.png")

# ── Tempo treino ──
fig, ax = plt.subplots(figsize=(12, 5))
train_times = [r.get("train_time_hours", 0) for r in valid_results]
bars = ax.barh(names, train_times, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.3)
for bar, v in zip(bars, train_times):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f"{v:.1f}h", ha="left", va="center", fontsize=8)
ax.set_xlabel("Tempo Total de Treino (horas)")
ax.set_title("Tempo de Treino por Mecanismo")
ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "train_time_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ── Tempo inferência ──
fig, ax = plt.subplots(figsize=(12, 5))
inf_times = [r["inference_time_sec"] for r in valid_results]
ax.barh(names, inf_times, color="#DD8452", alpha=0.85, edgecolor="black", linewidth=0.3)
for i, v in enumerate(inf_times):
    ax.text(v + 0.5, i, f"{v:.1f}s", va="center", fontsize=8)
ax.set_xlabel("Tempo de Inferência (segundos)")
ax.set_title("Tempo de Inferência por Mecanismo")
ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "inference_time_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ── BLEU vs Tempo Inferência (separado 30ep e 100ep) ──
for ep_target in sorted(set(r.get("epochs", 0) for r in valid_results)):
    ep_results = [r for r in valid_results if r.get("epochs") == ep_target]
    if not ep_results:
        continue
    fig, ax = plt.subplots(figsize=(10, 6))
    cores_exp = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860", "#8172B2", "#E377C2", "#8C564E"]
    for i, r in enumerate(ep_results):
        nl = r.get("noise", "")
        nl_label = "sem ruido" if nl == "sem_ruido" else f"ruido {nl.replace('ruido_std_', '').replace('_', '.')}"
        label = f"{r['display']}"
        ax.scatter(r["inference_time_sec"], r["bleu"], s=120, c=cores_exp[i % len(cores_exp)],
                   label=label, zorder=3, edgecolors="black", linewidth=0.5)
        ax.annotate(nl_label, (r["inference_time_sec"], r["bleu"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)
    ax.set_xlabel("Tempo de Inferência (s)"); ax.set_ylabel("SacreBLEU")
    ax.set_title(f"BLEU vs. Tempo de Inferência ({ep_target} épocas)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"tempo_vs_bleu_{ep_target}ep.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot salvo: tempo_vs_bleu_{ep_target}ep.png")

# ── Radar ──
metricas_radar = ["BLEU", "1/Loss", "1/PPL", "1/Params", "1/Tempo"]
raw = {
    "BLEU": [r["bleu"] for r in valid_results],
    "1/Loss":  [1/r["test_loss"] for r in valid_results],
    "1/PPL":   [1/r["test_ppl"]  for r in valid_results],
    "1/Params":[1/r["params_millions"] for r in valid_results],
    "1/Tempo": [1/r["inference_time_sec"] for r in valid_results],
}
def norm(vals):
    mn, mx = min(vals), max(vals)
    if mx == mn: return [0.5]*len(vals)
    return [(v - mn)/(mx - mn + 1e-9) for v in vals]
normalized = {k: norm(v) for k, v in raw.items()}
N = len(metricas_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, r in enumerate(valid_results):
    vals = [normalized[m][i] for m in metricas_radar]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, color=cores_exp[i % len(cores_exp)], label=r["display"])
    ax.fill(angles, vals, alpha=0.05, color=cores_exp[i % len(cores_exp)])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metricas_radar, size=10)
ax.set_ylim(0, 1); ax.set_title("Radar Comparativo (normalizado)")
ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "radar_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ── Heatmap ──
nomes_curtos = [r["display"][:20] for r in valid_results]
metricas_heat = {
    "BLEU": [r["bleu"] for r in valid_results],
    "Test Loss": [r["test_loss"] for r in valid_results],
    "Test PPL": [r["test_ppl"] for r in valid_results],
    "Params (M)": [r["params_millions"] for r in valid_results],
    "Inf. (s)": [r["inference_time_sec"] for r in valid_results],
}
df = pd.DataFrame(metricas_heat, index=nomes_curtos)
for col in ["Test Loss", "Test PPL", "Params (M)", "Inf. (s)"]:
    mn, mx = df[col].min(), df[col].max()
    df[col] = 1 - (df[col] - mn) / (mx - mn + 1e-9) if mx != mn else 0.5
for col in ["BLEU"]:
    mn, mx = df[col].min(), df[col].max()
    df[col] = (df[col] - mn) / (mx - mn + 1e-9) if mx != mn else 0.5
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1, linewidths=0.5, cbar_kws={"label": "Score normalizado"})
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Heatmap de Métricas (normalizado 0-1)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 5. ANÁLISE DE RUÍDO (apenas experimentos que têm training_history.pt)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GRÁFICOS EXTRAS — ANÁLISE DE RUÍDO NO TREINO")
print("=" * 60)

NOISE_STEPS = 10_000

ruido_exps = [e for e in EXPERIMENTS if e["noise"] != "sem_ruido"]
historicos_ruido = {}
for exp in ruido_exps:
    hp = BASE / exp["path"] / "history/training_history.pt"
    if hp.exists():
        historicos_ruido[exp["id"]] = torch.load(hp, map_location="cpu", weights_only=False)

# Constrói pares ruido→baseline dinamicamente
PARES_RUIDO = {}
NOME_RUIDO = {}
NOME_BASE = {}
CORES_RUIDO = {}
CORES_BASE = ["#AAAAAA", "#888888", "#999999", "#BBBBBB", "#CCCCCC"]
ci_base = 0

for exp in ruido_exps:
    # Busca baseline: mesmo mechanism + epoch, noise=sem_ruido
    base_list = [e for e in EXPERIMENTS if e["mechanism"] == exp["mechanism"]
                 and e["epochs"] == exp["epochs"] and e["noise"] == "sem_ruido"]
    if not base_list:
        continue
    base = base_list[0]
    PARES_RUIDO[exp["id"]] = base["id"]
    NOME_RUIDO[exp["id"]] = exp["display"]
    if base["id"] not in NOME_BASE:
        NOME_BASE[base["id"]] = base["display"]
        # Tom de cinza pra baseline
        CORES_RUIDO[base["id"]] = CORES_BASE[ci_base % len(CORES_BASE)]
        ci_base += 1
    # Cor do experimento ruído
    paleta = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860", "#8172B2", "#E377C2", "#C4B454"]
    CORES_RUIDO[exp["id"]] = paleta[len(CORES_RUIDO) % len(paleta)]

if historicos_ruido:
    for ruido_key, base_key in PARES_RUIDO.items():
        if ruido_key not in historicos_ruido:
            continue
        h_ruido = historicos_ruido[ruido_key]
        tr_r = h_ruido["training_results"]
        ep = list(range(1, len(tr_r["train_epoch_loss_history"]) + 1))
        n_epochs = len(ep)

        grad_norm_len = len(tr_r.get("grad_norm_history", []))
        steps_per_epoch = grad_norm_len / max(n_epochs, 1)
        noise_epochs = [s / steps_per_epoch for s in range(NOISE_STEPS, grad_norm_len + 1, NOISE_STEPS)]

        cor_ruido = CORES_RUIDO.get(ruido_key, "#333")
        cor_base  = CORES_RUIDO.get(base_key, "#666")
        nome_curto = NOME_RUIDO.get(ruido_key, ruido_key)

        # Spikes
        spike_json = BASE / [e for e in EXPERIMENTS if e["id"] == ruido_key][0]["path"] / "history/spike_steps.json"
        spikes = []
        if spike_json.exists():
            with open(spike_json) as f:
                spikes = json.load(f)
        spike_epochs_list = [s["epoch"] for s in spikes if "epoch" in s]
        n_spikes = len(spikes)

        # ── Plot perda c/ ruido vs baseline ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        ax1.plot(ep, tr_r["train_epoch_loss_history"], label="Treino", color=cor_ruido, linewidth=2)
        ax1.plot(ep, tr_r["valid_epoch_loss_history"], label="Validação", color=cor_ruido, linewidth=2, linestyle="--", alpha=0.65)
        test_loss_val = tr_r.get("test_loss")
        if test_loss_val is not None and not math.isnan(test_loss_val):
            ax1.axhline(y=test_loss_val, color=cor_ruido, linestyle=":", alpha=0.5, linewidth=1.5, label=f"Teste = {test_loss_val:.3f}")
        for ne in noise_epochs:
            ax1.axvspan(ne - 0.1, ne + 0.1, facecolor="red", alpha=0.25, edgecolor="none", zorder=2)
        if spike_epochs_list:
            ax1.scatter(spike_epochs_list, [0.96]*len(spike_epochs_list), marker="v",
                        color="#C08A2E", s=80, alpha=0.8, zorder=5, transform=ax1.get_xaxis_transform(), clip_on=False)
        ax1.set_title(f"{nome_curto} — c/ Ruído")
        ax1.set_xlabel("Época"); ax1.set_xlim(1 - 0.5, n_epochs + 0.5)
        h1 = ax1.get_legend_handles_labels()[0]
        h1.append(Patch(facecolor="red", alpha=0.25, label="Ruído (10k steps)"))
        if n_spikes > 0:
            h1.append(Line2D([0], [0], marker="v", color="#C08A2E", linestyle="",
                             markerfacecolor="#C08A2E", markersize=8, alpha=0.8, label=f"Spikes: {n_spikes}"))
        ax1.legend(handles=h1, fontsize=8)

        # Baseline
        base_list_b = [e for e in EXPERIMENTS if e["id"] == base_key]
        if base_list_b:
            hp_base = BASE / base_list_b[0]["path"] / "history/training_history.pt"
            if hp_base.exists():
                h_base = torch.load(hp_base, map_location="cpu", weights_only=False)
                tr_base = h_base["training_results"]
                ep_base = list(range(1, len(tr_base["train_epoch_loss_history"]) + 1))
                ax2.plot(ep_base, tr_base["train_epoch_loss_history"], label="Treino", color=cor_base, linewidth=2)
                ax2.plot(ep_base, tr_base["valid_epoch_loss_history"], label="Validação", color=cor_base, linewidth=2, linestyle="--", alpha=0.65)
                tl_base = tr_base.get("test_loss")
                if tl_base is not None and not math.isnan(tl_base):
                    ax2.axhline(y=tl_base, color=cor_base, linestyle=":", alpha=0.5, linewidth=1.5, label=f"Teste = {tl_base:.3f}")
                ax2.set_xlim(1 - 0.5, len(ep_base) + 0.5)
        ax2.set_title(f"{NOME_BASE.get(base_key, base_key)} — sem Ruído")
        ax2.set_xlabel("Época")
        ax2.legend(fontsize=8)

        fig.suptitle(f"Perda — {nome_curto}", fontsize=13, y=1.01)
        fig.supylabel("Perda (cross-entropy)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"analise_ruido_{ruido_key}_loss.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Plot salvo: analise_ruido_{ruido_key}_loss.png")

        # ── Plot grad norm ──
        grad_norm = tr_r.get("grad_norm_history", [])
        threshold_hist = tr_r.get("threshold_history", [])
        if grad_norm:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(grad_norm, color="#5F6B73", alpha=0.18, linewidth=0.75, zorder=1)
            if threshold_hist and len(threshold_hist) >= len(grad_norm):
                thr_plot = [v if not math.isnan(v) else None for v in threshold_hist[:len(grad_norm)]]
                ax.plot(thr_plot, color="#5B7C5A", linewidth=1.15, linestyle="--", alpha=0.95, label="Threshold (rolling)", zorder=2)
            spike_grad = [s.get("grad_norm", 0) for s in spikes]
            spike_steps = [s.get("global_step", 0) for s in spikes]
            if spike_steps:
                ax.scatter(spike_steps, spike_grad, color="#C08A2E", s=24, alpha=0.95, edgecolors="white", linewidths=0.35, zorder=3)
            for s_step in range(NOISE_STEPS, len(grad_norm) + 1, NOISE_STEPS):
                ax.axvspan(s_step - 50, s_step + 50, facecolor="red", alpha=0.12, edgecolor="none", zorder=0)
            ax.set_xlabel("Global step"); ax.set_ylabel("Norma do Gradiente")
            ax.set_title(f"Norma do Gradiente — {nome_curto} (c/ Ruído)")
            ax.grid(True, alpha=0.22, linewidth=0.8)
            handles = [Line2D([0], [0], color="#5F6B73", lw=1.5, alpha=0.5, label="Grad norm"),
                       Patch(facecolor="red", alpha=0.12, label="Ruído (10k steps)")]
            if threshold_hist and len(threshold_hist) >= len(grad_norm):
                handles.append(Line2D([0], [0], color="#5B7C5A", lw=1.5, linestyle="--", label="Threshold"))
            if spike_steps:
                handles.append(Line2D([0], [0], marker="o", color="#C08A2E", linestyle="",
                                      markerfacecolor="#C08A2E", markersize=6, alpha=0.8, label=f"Spikes: {len(spikes)}"))
            ax.legend(handles=handles, fontsize=8, frameon=True, framealpha=0.9)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"analise_ruido_{ruido_key}_grad.png", dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Plot salvo: analise_ruido_{ruido_key}_grad.png")

        # ── Plot loss por step ──
        train_loss_step = tr_r.get("train_loss_history", [])
        if train_loss_step:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(train_loss_step, color="#6B7280", alpha=0.18, linewidth=0.8, zorder=1)
            spike_loss = [s.get("loss", 0) for s in spikes]
            spike_steps = [s.get("global_step", 0) for s in spikes]
            if spike_steps:
                ax.scatter(spike_steps, spike_loss, color="#C08A2E", s=26, alpha=0.95, edgecolors="white", linewidths=0.35, zorder=3)
            for s_step in range(NOISE_STEPS, len(train_loss_step) + 1, NOISE_STEPS):
                ax.axvspan(s_step - 50, s_step + 50, facecolor="red", alpha=0.12, edgecolor="none", zorder=0)
            ax.set_xlabel("Global step"); ax.set_ylabel("Loss")
            ax.set_title(f"Perda por step — {nome_curto} (c/ Ruído)")
            ax.grid(True, alpha=0.22, linewidth=0.8)
            handles = [Line2D([0], [0], color="#6B7280", lw=1.5, alpha=0.5, label="Loss por step"),
                       Patch(facecolor="red", alpha=0.12, label="Ruído (10k steps)")]
            if spike_steps:
                handles.append(Line2D([0], [0], marker="o", color="#C08A2E", linestyle="",
                                      markerfacecolor="#C08A2E", markersize=6, alpha=0.8, label=f"Spikes: {len(spikes)}"))
            ax.legend(handles=handles, fontsize=8, frameon=True, framealpha=0.9)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"analise_ruido_{ruido_key}_loss_step.png", dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Plot salvo: analise_ruido_{ruido_key}_loss_step.png")

    # ── Plot combinado val loss (separado por época) ──
    for ep_alvo in sorted(set(e["epochs"] for e in EXPERIMENTS)):
        fig, ax = plt.subplots(figsize=(12, 5))
        noise_drawn = False
        pares_ep = {k: v for k, v in PARES_RUIDO.items()
                    if k.startswith(f"ruido") and any(
                        e["id"] == k and e["epochs"] == ep_alvo for e in EXPERIMENTS)}
        for ruido_key, base_key in pares_ep.items():
            if ruido_key not in historicos_ruido:
                continue
            h_r = historicos_ruido[ruido_key]
            tr_r = h_r["training_results"]
            ep_r = list(range(1, len(tr_r["train_epoch_loss_history"]) + 1))
            ax.plot(ep_r, tr_r["valid_epoch_loss_history"],
                    label=f"{NOME_RUIDO.get(ruido_key, ruido_key)}", color=CORES_RUIDO.get(ruido_key, "#333"), linewidth=2)
            base_e = [e for e in EXPERIMENTS if e["id"] == base_key]
            if base_e:
                hp_b = BASE / base_e[0]["path"] / "history/training_history.pt"
                if hp_b.exists():
                    try:
                        h_b = torch.load(hp_b, map_location="cpu", weights_only=False)
                        tr_b = h_b["training_results"]
                        ep_b = list(range(1, len(tr_b["valid_epoch_loss_history"]) + 1))
                        ax.plot(ep_b, tr_b["valid_epoch_loss_history"],
                                label=NOME_BASE.get(base_key, base_key),
                                color=CORES_RUIDO.get(base_key, "#666"), linewidth=2, linestyle="--", alpha=0.5)
                    except Exception:
                        pass
            if not noise_drawn:
                gn_len = len(tr_r.get("grad_norm_history", []))
                steps_per_ep = gn_len / max(len(ep_r), 1)
                for s in range(NOISE_STEPS, gn_len + 1, NOISE_STEPS):
                    ax.axvline(x=s / steps_per_ep, color="red", linestyle=":", alpha=0.30, linewidth=0.7)
                noise_drawn = True
        handles_comp, labels_comp = ax.get_legend_handles_labels()
        handles_comp.append(Line2D([0], [0], color="red", linestyle=":", alpha=0.7, linewidth=1, label="Ruído (10k steps)"))
        ax.legend(handles=handles_comp, fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_title(f"Validação Loss — Experimentos c/ Ruído ({ep_alvo} épocas)")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        plt.tight_layout()
        fname = f"analise_ruido_comparativo_valid_loss_{ep_alvo}ep.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Plot salvo: {fname}")

    # ── Plot impacto BLEU ──
    all_comp = sorted(set(PARES_RUIDO.keys()) | set(PARES_RUIDO.values()))
    nomes_comp = [NOME_BASE.get(e) or NOME_RUIDO.get(e, e) for e in all_comp]
    cores_comp_bar = [CORES_RUIDO.get(e, "#999") for e in all_comp]
    bleus_comp = [inf_map.get(e, {}).get("bleu", 0) for e in all_comp]
    # Separa em 2 subplots lado a lado
    n = len(all_comp)
    mid = (n + 1) // 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, idxs, tit in [(ax1, range(mid), "Grupo 1"), (ax2, range(mid, n), "Grupo 2")]:
        sub_comp = [all_comp[i] for i in idxs]
        sub_nomes = [nomes_comp[i] for i in idxs]
        sub_cores = [cores_comp_bar[i] for i in idxs]
        sub_bleus = [bleus_comp[i] for i in idxs]
        xc = np.arange(len(sub_comp))
        ax.bar(xc, sub_bleus, color=sub_cores, width=0.5, edgecolor="black", linewidth=0.5)
        for i2, (v, e) in enumerate(zip(sub_bleus, sub_comp)):
            if v > 0:
                ax.text(i2, v + 0.3, f"{v:.1f}", ha="center", fontsize=10,
                        fontweight="bold" if "ruido" in e else "normal")
        ax.set_xticks(xc); ax.set_xticklabels(sub_nomes, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("SacreBLEU"); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Impacto do Ruído — SacreBLEU", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analise_ruido_impacto_bleu.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot salvo: analise_ruido_impacto_bleu.png")

    # ── Plot agrupado: BLEU por mecanismo x nível de ruído ──────────────────────
    # Monta grupos dinamicamente por época
    for ep_group in sorted(set(e["epochs"] for e in EXPERIMENTS)):
        ep_slug = f"{ep_group}_epocas"
        exps_ep = [e for e in EXPERIMENTS if e["epochs"] == ep_group]

        # Descobre mecanismos e níveis de ruído disponíveis
        mechanisms = sorted(set(e["mechanism"] for e in exps_ep))
        noise_levels = sorted(set(e["noise"] for e in exps_ep))
        # Mapeia noise_label → display
        noise_display = {}
        for nl in noise_levels:
            if nl == "sem_ruido":
                noise_display[nl] = "sem ruido"
            else:
                noise_display[nl] = f"ruido {nl.replace('ruido_std_', '').replace('_', '.')}"

        # Paleta: sem_ruido=azul, ruidos=espectro laranja→vermelho
        CORES_NIVEL = {}
        paleta_noise = ["#4C72B0", "#DD8452", "#C44E52", "#937860", "#8172B2", "#E377C2"]
        for i, nl in enumerate([n for n in noise_levels if n != "sem_ruido"] +
                                (["sem_ruido"] if "sem_ruido" in noise_levels else [])):
            CORES_NIVEL[nl] = paleta_noise[i % len(paleta_noise)]
        ROTULOS_NIVEL = [n for n in noise_levels]  # ordenados

        # BLEU
        fig, ax = plt.subplots(figsize=(10, 5))
        xg = np.arange(len(mechanisms))
        n_niveis = len(noise_levels)
        largura = 0.7 / max(n_niveis, 1)
        for i, nl in enumerate(noise_levels):
            bleus_nl = []
            for m in mechanisms:
                match = [e for e in exps_ep if e["mechanism"] == m and e["noise"] == nl]
                bleus_nl.append(inf_map.get(match[0]["id"], {}).get("bleu", 0) if match else 0)
            offset = (i - (n_niveis - 1) / 2) * largura
            bars = ax.bar(xg + offset, bleus_nl, largura,
                          label=noise_display.get(nl, nl), color=CORES_NIVEL[nl],
                          edgecolor="black", linewidth=0.3)
            for bar, v in zip(bars, bleus_nl):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                            f"{v:.1f}", ha="center", fontsize=8, va="bottom")
        ax.set_xticks(xg); ax.set_xticklabels(mechanisms, fontsize=10)
        ax.set_ylabel("SacreBLEU")
        ax.set_title(f"BLEU por Mecanismo e Nível de Ruído ({ep_group} épocas)")
        ax.legend(title="Nível de ruído", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"analise_ruido_bleu_{ep_slug}.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Plot salvo: analise_ruido_bleu_{ep_slug}.png")

        # Filtros específicos para Loss
        loss_mechanisms = list(mechanisms)
        loss_noise_levels = list(noise_levels)
        if ep_group == 30:
            loss_mechanisms = ["dot_product"]
        elif ep_group == 100:
            loss_noise_levels = [nl for nl in noise_levels if nl in ("sem_ruido", "ruido_std_0_10")]

        # Loss
        fig, ax = plt.subplots(figsize=(10, 5))
        xg_loss = np.arange(len(loss_mechanisms))
        n_niveis_loss = len(loss_noise_levels)
        largura_loss = 0.7 / max(n_niveis_loss, 1)
        for i, nl in enumerate(loss_noise_levels):
            losses_nl = []
            for m in loss_mechanisms:
                match = [e for e in exps_ep if e["mechanism"] == m and e["noise"] == nl]
                losses_nl.append(inf_map.get(match[0]["id"], {}).get("test_loss", 0) if match else 0)
            offset = (i - (n_niveis_loss - 1) / 2) * largura_loss
            bars = ax.bar(xg_loss + offset, losses_nl, largura_loss,
                          label=noise_display.get(nl, nl), color=CORES_NIVEL[nl],
                          edgecolor="black", linewidth=0.3)
            for bar, v in zip(bars, losses_nl):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", fontsize=8, va="bottom", rotation=90)
        ax.set_xticks(xg_loss); ax.set_xticklabels(loss_mechanisms, fontsize=10)
        ax.set_ylabel("Test Loss")
        ax.set_title(f"Loss por Mecanismo e Nível de Ruído ({ep_group} épocas)")
        ax.legend(title="Nível de ruído", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"analise_ruido_loss_{ep_slug}.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Plot salvo: analise_ruido_loss_{ep_slug}.png")

else:
    print("  Nenhum histórico de treino com ruído encontrado.")

# ══════════════════════════════════════════════════════════════════════════════
# 6. AMOSTRAS DE TRADUÇÃO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("AMOSTRAS DE TRADUÇÃO (primeiras 3 sentenças)")
print("=" * 60)
for i in range(3):
    print(f"\n--- Frase {i+1} ---")
    for r in valid_results:
        if i < len(r.get("samples", [])):
            s = r["samples"][i]
            print(f"  [{r['display'][:30]:<30}] {s['hypothesis']}")
    ref = next((r["samples"][i]["reference"] for r in valid_results if i < len(r.get("samples", []))), "")
    print(f"  {'Referência:':<32} {ref}")

print(f"\nTodos os resultados em: {OUTPUT_DIR}/")
print("Concluído!")
