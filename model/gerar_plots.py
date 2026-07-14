#!/usr/bin/env python3
"""Gera todos os plots individuais e comparativos para o artigo CONGRESSO."""
import torch, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd

# ── Config de ruído ───────────────────────────────────────────────────────────
NOISE_STEPS = 10_000  # ruído a cada 10k global steps no train.py

def get_noise_epochs(exp_name):
    """Retorna lista de épocas onde ruído foi injetado, baseado no training_history."""
    pt_path = Path(f"data/{exp_name}/history/training_history.pt")
    if not pt_path.exists():
        return []
    try:
        h = torch.load(pt_path, map_location="cpu", weights_only=False)
        tr = h["training_results"]
        grad_norm = tr.get("grad_norm_history", [])
        if not grad_norm:
            return []
        n_epochs = len(tr.get("train_epoch_loss_history", []))
        if n_epochs == 0:
            return []
        steps_per_epoch = len(grad_norm) / n_epochs
        noise_epochs = []
        for s in range(NOISE_STEPS, len(grad_norm) + 1, NOISE_STEPS):
            noise_epochs.append(s / steps_per_epoch)
        return noise_epochs
    except Exception:
        return []


plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTPUT_DIR = Path("editor/artigo/figuras")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTOS = [
    "attention_dot_product_epoch_30",
    "attention_dot_product_epoch_100",
    "attention_dot_product_ruido_epoch_100",
    "attention_dense_synthesizer_epoch_30",
    "attention_dense_synthesizer_epoch_100",
    "attention_dense_synthesizer_ruido_epoch_100",
    "attention_factorized_dense_synthesizer_epoch_30",
    "attention_factorized_dense_synthesizer_ruido_epoch_100",
]

EXPERIMENTOS_RUIDO = [e for e in EXPERIMENTOS if "ruido" in e]

CORES = {
    "attention_dot_product_epoch_30": "#4C72B0",
    "attention_dot_product_epoch_100": "#DD8452",
    "attention_dot_product_ruido_epoch_100": "#7A3B8E",
    "attention_dense_synthesizer_epoch_30": "#55A868",
    "attention_dense_synthesizer_epoch_100": "#C4B454",
    "attention_dense_synthesizer_ruido_epoch_100": "#D62728",
    "attention_factorized_dense_synthesizer_epoch_30": "#937860",
    "attention_factorized_dense_synthesizer_ruido_epoch_100": "#E377C2",
}

NOMES = {
    "attention_dot_product_epoch_30": "Dot-Product 30ep",
    "attention_dot_product_epoch_100": "Dot-Product 100ep",
    "attention_dot_product_ruido_epoch_100": "Dot-Product c/ Ruído 100ep",
    "attention_dense_synthesizer_epoch_30": "Dense Synth 30ep",
    "attention_dense_synthesizer_epoch_100": "Dense Synth 100ep",
    "attention_dense_synthesizer_ruido_epoch_100": "Dense Synth c/ Ruído",
    "attention_factorized_dense_synthesizer_epoch_30": "Fact Dense 30ep",
    "attention_factorized_dense_synthesizer_ruido_epoch_100": "Fact Dense c/ Ruído",
}

# Carregar dados
historicos = {}
for exp in EXPERIMENTOS:
    pt_path = Path(f"data/{exp}/history/training_history.pt")
    if pt_path.exists():
        try:
            historicos[exp] = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  AVISO: {exp}: {e}")

with open("inference_results/results.json") as f:
    inference = json.load(f)
inference_map = {r["name"]: r for r in inference if not r.get("error")}

emis = [e for e in EXPERIMENTOS if e in inference_map]
novos_exp = []  # sem plots individuais na versao compacta

# GRUPO 1 — Plots individuais para os 2 novos
for exp in novos_exp:
    if exp not in historicos:
        print(f"  AVISO: {exp} sem historico, pulando plots individuais")
        continue
    h = historicos[exp]
    tr = h["training_results"]
    tim = h["timing"]
    gpu = h["gpu_memory"]
    epocas = list(range(1, len(tr["train_epoch_loss_history"]) + 1))
    bleu_epocas = tr.get("bleu_epochs", [])

    # Curvas de treino
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epocas, tr["train_epoch_loss_history"], label="Treino", color=CORES[exp], linewidth=2)
    ax.plot(epocas, tr["valid_epoch_loss_history"], label="Validação", color=CORES[exp], linewidth=2, linestyle="--", alpha=0.7)
    ax.set_xlabel("Época"); ax.set_ylabel("Perda (cross-entropy)"); ax.legend()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"{exp}_curvas_treino.png", bbox_inches="tight"); plt.close()

    # Perplexidade
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epocas, tr["train_epoch_ppl_history"], label="Treino", color=CORES[exp], linewidth=2)
    ax.plot(epocas, tr["valid_epoch_ppl_history"], label="Validação", color=CORES[exp], linewidth=2, linestyle="--", alpha=0.7)
    ax.set_xlabel("Época"); ax.set_ylabel("Perplexidade (PPL)"); ax.legend()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"{exp}_perplexidade.png", bbox_inches="tight"); plt.close()

    # BLEU validação
    if bleu_epocas and tr.get("valid_epoch_bleu_history"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(bleu_epocas, tr["valid_epoch_bleu_history"], "o-", label="SacreBLEU", color="#4C72B0", linewidth=2)
        ax.set_xlabel("Época"); ax.set_ylabel("SacreBLEU"); ax.legend()
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"{exp}_bleu_validacao.png", bbox_inches="tight"); plt.close()

    # Tempo por época
    fig, ax = plt.subplots(figsize=(8, 4))
    epoch_times = tim.get("epoch_time_history_sec", [])
    if epoch_times:
        ax.bar(epocas[:len(epoch_times)], [t/60 for t in epoch_times], color=CORES[exp], alpha=0.8)
        ax.set_xlabel("Época"); ax.set_ylabel("Tempo (min)")
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"{exp}_tempo_epocas.png", bbox_inches="tight"); plt.close()

    # Memória GPU
    fig, ax = plt.subplots(figsize=(8, 4))
    gpu_peak = gpu.get("peak_mb_per_epoch", [])
    if gpu_peak:
        ax.plot(range(1, len(gpu_peak)+1), gpu_peak, color=CORES[exp], linewidth=2)
        ax.set_xlabel("Época"); ax.set_ylabel("GPU Peak (MB)")
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"{exp}_memoria_gpu.png", bbox_inches="tight"); plt.close()

    print(f"   Plots individuais: {exp}")

# GRUPO 2 — Plots comparativos
valid = [inference_map[e] for e in emis]
names_short = [NOMES.get(r["name"], r["name"])[:22] for r in valid]

# Usar atalhos super curtos para legibilidade
NAMES_CURTOS = {
    "attention_dot_product_epoch_30": "DP 30ep",
    "attention_dot_product_epoch_100": "DP 100ep",
    "attention_dot_product_ruido_epoch_100": "DP Ruido",
    "attention_dense_synthesizer_epoch_30": "Dense 30ep",
    "attention_dense_synthesizer_epoch_100": "Dense 100ep",
    "attention_dense_synthesizer_ruido_epoch_100": "Dense Ruido",
    "attention_factorized_dense_synthesizer_epoch_30": "FactDen 30",
    "attention_factorized_dense_synthesizer_ruido_epoch_100": "FactDen Ruid",
}
names_short = [NAMES_CURTOS.get(r["name"], r["name"][:12]) for r in valid]
x = np.arange(len(names_short))
w = 0.25

# BLEU comparativo (apenas SacreBLEU)
fig, ax = plt.subplots(figsize=(14, 5))
cores_list = [CORES.get(e, "#333") for e in emis]
bars = ax.bar(x, [r["bleu"] for r in valid], width=0.5, color=cores_list, edgecolor="black", linewidth=0.3)
for bar, v in zip(bars, [r["bleu"] for r in valid]):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(names_short, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("SacreBLEU")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "comparativo_bleu.png", bbox_inches="tight"); plt.close()
print("   comparativo_bleu.png")

# Loss/PPL
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
cores_list = [CORES.get(e, "#333") for e in emis]
ax1.bar(names_short, [r["test_loss"] for r in valid], color=cores_list, alpha=0.85, edgecolor="black", linewidth=0.3)
ax1.set_title("Perda (teste)", fontsize=13); ax1.set_ylabel("Perda (cross-entropy)"); ax1.tick_params(axis="x", rotation=25, labelsize=9)
for i, v in enumerate([r["test_loss"] for r in valid]):
    ax1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
ax2.bar(names_short, [r["test_ppl"] for r in valid], color=cores_list, alpha=0.85, edgecolor="black", linewidth=0.3)
ax2.set_title("Perplexidade (PPL)", fontsize=13); ax2.set_ylabel("PPL"); ax2.tick_params(axis="x", rotation=25, labelsize=9)
for i, v in enumerate([r["test_ppl"] for r in valid]):
    ax2.text(i, v + 1.0, f"{v:.1f}", ha="center", fontsize=8)
plt.subplots_adjust(bottom=0.25)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "comparativo_loss_ppl.png", bbox_inches="tight"); plt.close()
print("   comparativo_loss_ppl.png")

# Radar
metricas = ["BLEU", "1/Loss", "1/PPL", "1/Params", "1/Tempo"]
raw = {m: [r["bleu"] if m == "BLEU" else (1/r["test_loss"] if m == "1/Loss" else (1/r["test_ppl"] if m == "1/PPL" else (1/r["params_millions"] if m == "1/Params" else 1/r["inference_time_sec"]))) for r in valid] for m in metricas}

# Actually build raw manually
raw = {
    "BLEU": [r["bleu"] for r in valid],
    "1/Loss":    [1/r["test_loss"] for r in valid],
    "1/PPL":     [1/r["test_ppl"]  for r in valid],
    "1/Params":  [1/r["params_millions"] for r in valid],
    "1/Tempo":   [1/r["inference_time_sec"] for r in valid],
}

def norm(vals):
    mn, mx = min(vals), max(vals)
    if mx == mn: return [0.5]*len(vals)
    return [(v - mn)/(mx - mn + 1e-9) for v in vals]

normalized = {k: norm(v) for k, v in raw.items()}
N = len(metricas)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, r in enumerate(valid):
    vals = [normalized[m][i] for m in metricas] + [normalized[metricas[0]][i]]
    ax.plot(angles, vals, linewidth=2, color=CORES.get(r["name"], "#333"),
            label=NOMES.get(r["name"], r["name"][:20]))
    ax.fill(angles, vals, alpha=0.05, color=CORES.get(r["name"], "#333"))
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metricas, size=10)
ax.set_ylim(0, 1)
ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "comparativo_radar.png", bbox_inches="tight"); plt.close()
print("   comparativo_radar.png")

# Heatmap
nomes_heat = [NAMES_CURTOS.get(r["name"], r["name"][:12]) for r in valid]
metricas_hm = {
    "BLEU":  [r["bleu"]  for r in valid],
    "Test Loss":  [r["test_loss"]  for r in valid],
    "Test PPL":   [r["test_ppl"]   for r in valid],
    "Params (M)": [r["params_millions"] for r in valid],
    "Inf. (s)":   [r["inference_time_sec"] for r in valid],
}
df = pd.DataFrame(metricas_hm, index=nomes_heat)
for col in ["Test Loss", "Test PPL", "Params (M)", "Inf. (s)"]:
    mn, mx = df[col].min(), df[col].max()
    df[col] = 1 - (df[col] - mn) / (mx - mn + 1e-9) if mx != mn else 0.5
for col in ["BLEU"]:
    mn, mx = df[col].min(), df[col].max()
    df[col] = (df[col] - mn) / (mx - mn + 1e-9) if mx != mn else 0.5
fig, ax = plt.subplots(figsize=(16, 7))
heatmap = sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1, linewidths=0.8, cbar_kws={"label": "Score normalizado", "shrink": 0.8})
ax.set_xlabel(""); ax.set_ylabel("")
plt.yticks(rotation=0)
plt.xticks(rotation=30, ha="right")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "comparativo_heatmap.png", bbox_inches="tight"); plt.close()
print("   comparativo_heatmap.png")

# GRUPO 2.5 — Tempo de treino e inferencia
fig, ax = plt.subplots(figsize=(10, 7))
train_times = [r.get("train_time_hours", 0) for r in valid]
bars = ax.barh(names_short, train_times, color=cores_list, alpha=0.85, edgecolor="black", linewidth=0.3)
for bar, v in zip(bars, train_times):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f"{v:.1f}h",
            ha="left", va="center", fontsize=9)
ax.set_xlabel("Tempo Total de Treino (horas)", fontsize=11)
ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "comparativo_tempo_treino.png", bbox_inches="tight"); plt.close()
print("   comparativo_tempo_treino.png")

fig, ax = plt.subplots(figsize=(10, 7))
inf_times = [r["inference_time_sec"] for r in valid]
ax.barh(names_short, inf_times, color=cores_list, alpha=0.85, edgecolor="black", linewidth=0.3)
for i, v in enumerate(inf_times):
    ax.text(v + 0.5, i, f"{v:.1f}s", va="center", fontsize=9)
ax.set_xlabel("Tempo de Inferencia (segundos)", fontsize=11)
ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "comparativo_tempo_inferencia.png", bbox_inches="tight"); plt.close()
print("   comparativo_tempo_inferencia.png")

# GRUPO 3 — Plots do artigo
fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(names_short, [r["bleu"] for r in valid], width=0.6, color=cores_list, edgecolor="black", linewidth=0.5)
for bar, v in zip(bars, [r["bleu"] for r in valid]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticklabels(names_short, rotation=20, ha="right", fontsize=9); ax.set_ylabel("SacreBLEU")
ax.set_xlabel("")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "artigo_bleu_comparativo.png", bbox_inches="tight"); plt.close()
print("   artigo_bleu_comparativo.png")

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(names_short, [r["test_loss"] for r in valid], color=cores_list, alpha=0.85, edgecolor="black", linewidth=0.5)
for i, v in enumerate([r["test_loss"] for r in valid]):
    ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticklabels(names_short, rotation=20, ha="right"); ax.set_ylabel("Perda (cross-entropy)")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "artigo_loss_comparativo.png", bbox_inches="tight"); plt.close()
print("   artigo_loss_comparativo.png")

# Curvas sobrepostas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
for exp in EXPERIMENTOS:
    if exp not in historicos: continue
    h = historicos[exp]
    label = NOMES.get(exp, exp)
    cor = CORES.get(exp, "#333")
    eps = range(1, len(h["training_results"]["train_epoch_loss_history"]) + 1)
    ax1.plot(eps, h["training_results"]["train_epoch_loss_history"], label=label, color=cor, linewidth=1.8)
    ax2.plot(eps, h["training_results"]["valid_epoch_loss_history"], label=label, color=cor, linewidth=1.8, linestyle="--")

    # Marcadores de ruído e spikes nos experimentos com ruído
    if "ruido" in exp:
        noise_eps = get_noise_epochs(exp)
        for ne in noise_eps:
            ax1.axvline(x=ne, color=cor, linestyle=":", alpha=0.40, linewidth=0.8)
            ax2.axvline(x=ne, color=cor, linestyle=":", alpha=0.40, linewidth=0.8)
        # Spikes
        spikes = h["training_results"].get("spikes", [])
        for s in spikes:
            if "epoch" in s:
                ax2.scatter(s["epoch"], ax2.get_ylim()[1] * 0.95,
                           marker="v", color=cor, s=15, alpha=0.4, zorder=5)

ax1.set_xlabel("Época"); ax1.set_ylabel("Loss (treino)")
ax2.set_xlabel("Época"); ax2.set_ylabel("Loss (validação)"); ax2.legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "artigo_curvas_sobrepostas.png", bbox_inches="tight"); plt.close()
print("   artigo_curvas_sobrepostas.png")

# Artigo Radar
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, r in enumerate(valid):
    vals = [normalized[m][i] for m in metricas] + [normalized[metricas[0]][i]]
    ax.plot(angles, vals, linewidth=2, color=CORES.get(r["name"], "#333"),
            label=NOMES.get(r["name"], r["name"][:20]))
    ax.fill(angles, vals, alpha=0.04, color=CORES.get(r["name"], "#333"))
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metricas, size=10)
ax.set_ylim(0, 1)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "artigo_radar.png", bbox_inches="tight"); plt.close()
print("   artigo_radar.png")

# ── GRUPO 4 — Análise de Ruído no Gradiente ───────────────────────────────────
if EXPERIMENTOS_RUIDO:
    RUIDO_CORES = {e: CORES[e] for e in EXPERIMENTOS_RUIDO}
    # Pares ruído vs. baseline
    PARES = [
        ("attention_dot_product_epoch_100", "attention_dot_product_ruido_epoch_100"),
        ("attention_dense_synthesizer_epoch_100", "attention_dense_synthesizer_ruido_epoch_100"),
        ("attention_factorized_dense_synthesizer_epoch_30", "attention_factorized_dense_synthesizer_ruido_epoch_100"),
    ]

    # Loss de validação: cada ruído vs seu baseline
    nome_curto_v = {
        "attention_dot_product_epoch_100": "Dot-Product",
        "attention_dot_product_ruido_epoch_100": "Dot-Product c/ Ruído",
        "attention_dense_synthesizer_epoch_100": "Dense Synth",
        "attention_dense_synthesizer_ruido_epoch_100": "Dense Synth c/ Ruído",
        "attention_factorized_dense_synthesizer_epoch_30": "Fact. Dense",
        "attention_factorized_dense_synthesizer_ruido_epoch_100": "Fact. Dense c/ Ruído",
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    noise_drawn = False
    for baseline, ruido in PARES:
        if ruido not in historicos:
            continue
        h_r = historicos[ruido]
        tr_r = h_r["training_results"]
        eps = list(range(1, len(tr_r["train_epoch_loss_history"]) + 1))
        ax.plot(eps, tr_r["valid_epoch_loss_history"],
                label=nome_curto_v.get(ruido, ruido), color=CORES.get(ruido), linewidth=2)

        if baseline in historicos:
            h_b = historicos[baseline]
            tr_b = h_b["training_results"]
            eps_b = list(range(1, len(tr_b["valid_epoch_loss_history"]) + 1))
            ax.plot(eps_b, tr_b["valid_epoch_loss_history"],
                    label=nome_curto_v.get(baseline, baseline), color=CORES.get(baseline),
                    linewidth=2, linestyle="--", alpha=0.5)

        # Marcadores de ruído (uma vez — schedule identico para todos)
        if not noise_drawn:
            ruido_eps = get_noise_epochs(ruido)
            for ne in ruido_eps:
                ax.axvline(x=ne, color="red", linestyle=":", alpha=0.30, linewidth=0.7)
            noise_drawn = True

    ax.axvline(x=-1, color="red", linestyle=":", alpha=0.7, linewidth=1, label="Ruído (10k steps)")
    ax.set_xlabel("Época"); ax.set_ylabel("Perda (cross-entropy)")
    ax.set_title("Comparação: experimentos com e sem ruído")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "artigo_ruido_valid_loss.png", bbox_inches="tight"); plt.close()
    print("   artigo_ruido_valid_loss.png")

    # Loss por época (train + val + test) com ruído e spikes
    for baseline, ruido in PARES:
        if ruido not in historicos:
            continue
        h_r = historicos[ruido]
        tr_r = h_r["training_results"]
        eps = list(range(1, len(tr_r["train_epoch_loss_history"]) + 1))
        ruido_eps = get_noise_epochs(ruido)
        spikes = tr_r.get("spikes", [])
        spike_eps = [s["epoch"] for s in spikes if "epoch" in s]
        test_loss = tr_r.get("test_loss")

        cor_exp = CORES.get(ruido, "#333")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eps, tr_r["train_epoch_loss_history"], label="Treino", color=cor_exp, linewidth=2)
        ax.plot(eps, tr_r["valid_epoch_loss_history"], label="Validação", color=cor_exp,
                linewidth=2, linestyle="--", alpha=0.55)

        if test_loss is not None and not (isinstance(test_loss, float) and math.isnan(test_loss)):
            ax.axhline(y=test_loss, color=cor_exp, linestyle=":",
                       alpha=0.5, linewidth=1.5, label=f"Teste = {test_loss:.4f}")

        for ne in ruido_eps:
            ax.axvline(x=ne, color="red", linestyle=":", alpha=0.50, linewidth=1.0)

        y_top = ax.get_ylim()[1]
        for se in spike_eps:
            ax.scatter(se, y_top * 0.96, marker="v", color=cor_exp,
                       s=22, alpha=0.5, zorder=5)

        leg_elements = [
            mpatches.Patch(color="red", alpha=0.5, label="Ruído (10k steps)"),
            plt.Line2D([0],[0], marker="v", color=cor_exp,
                       markerfacecolor=cor_exp, markersize=8,
                       linestyle="", alpha=0.5, label="Spike"),
        ]
        ax.legend(handles=ax.get_legend_handles_labels()[0] + leg_elements,
                  fontsize=8, loc="upper right")
        ax.set_xlabel("Época"); ax.set_ylabel("Perda (cross-entropy)")
        ax.set_title(f"Treino e Validação — {nome_curto_v.get(ruido, ruido)}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"artigo_ruido_loss_spikes_{ruido.replace('attention_', '')}.png"
        plt.savefig(OUTPUT_DIR / fname, bbox_inches="tight"); plt.close()
        print(f"   {fname}")

    # Barras comparativas BLEU: ruído vs. baseline
    fig, ax = plt.subplots(figsize=(10, 5))
    exps_set = sorted(set([p[0] for p in PARES] + [p[1] for p in PARES]))
    nomes_barras = [nome_curto_v.get(e, e) for e in exps_set]
    cores_barras = [CORES.get(e, "#999") for e in exps_set]
    bleus_barras = [inference_map.get(e, {}).get("bleu", 0) for e in exps_set]

    if len(bleus_barras) == len(exps_set):
        ax.bar(range(len(exps_set)), bleus_barras, color=cores_barras, width=0.5,
               edgecolor="black", linewidth=0.5)
        for i, (v, e) in enumerate(zip(bleus_barras, exps_set)):
            if v > 0:
                ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9,
                        fontweight="bold" if "ruido" in e else "normal")
        ax.set_xticks(range(len(exps_set)))
        ax.set_xticklabels(nomes_barras, rotation=15, ha="right")
        ax.set_ylabel("SacreBLEU")
        ax.set_title("Impacto do Ruído — SacreBLEU")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / "artigo_ruido_bleu.png", bbox_inches="tight"); plt.close()
        print("   artigo_ruido_bleu.png")

print("\n Todos os plots gerados com sucesso!")
