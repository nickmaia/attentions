import json
from pathlib import Path
import math
import time

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from transformer import Transformer
from datasets import load_dataset
from tokens import tokenize, tokenizer, make_dataloaders

from datasets import Dataset
import random
from difflib import SequenceMatcher

from sacrebleu.metrics import BLEU as SacreBLEU

from torch.optim.lr_scheduler import OneCycleLR

from attention_factorized_random_synthesizer import AttentionFactorizedRandomSynthesizer as Attention

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


def is_valid_pair(src: str, tgt: str, min_words=3, max_words=120, max_similarity=0.6) -> bool:
    src_words = src.split()
    tgt_words = tgt.split()

    # Filtro de comprimento
    if len(src_words) < min_words or len(tgt_words) < min_words:
        return False
    if len(src_words) > max_words or len(tgt_words) > max_words:
        return False

    # Filtro de similaridade — descarta pares quase idênticos
    similarity = SequenceMatcher(None, src.lower(), tgt.lower()).ratio()
    if similarity > max_similarity:
        return False

    return True

def load_mixed_dataset(seed=42):
    print("entrou em load_mixed_dataset")
    all_pairs = []

    print("carregando opus100...")
    opus100 = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train[:200_000]")
    for i, item in enumerate(opus100):
        pair = item["translation"]
        src, tgt = pair.get("en", "").strip(), pair.get("fr", "").strip()
        if is_valid_pair(src, tgt):
            all_pairs.append({"translation": {"en": src, "fr": tgt}, "source": "opus100"})
        if (i + 1) % 10000 == 0:
            print("opus100", i + 1, "itens processados")

    print("carregando opus_books...")
    opus_books = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train[:85%]")
    for i, item in enumerate(opus_books):
        pair = item["translation"]
        src, tgt = pair.get("en", "").strip(), pair.get("fr", "").strip()
        if is_valid_pair(src, tgt):
            all_pairs.append({"translation": {"en": src, "fr": tgt}, "source": "books"})
        if (i + 1) % 10000 == 0:
            print("opus_books", i + 1, "itens processados")

    print("carregando wmt14...")
    wmt14 = load_dataset("wmt/wmt14", "fr-en", split="train[:200_000]")
    for i, item in enumerate(wmt14):
        pair = item["translation"]
        src, tgt = pair.get("en", "").strip(), pair.get("fr", "").strip()
        if is_valid_pair(src, tgt):
            all_pairs.append({"translation": {"en": src, "fr": tgt}, "source": "wmt14"})
        if (i + 1) % 10000 == 0:
            print("wmt14", i + 1, "itens processados")

    print("carregando europarl...")
    # europarl só tem split "train" — slice para não misturar com val/test
    europarl = load_dataset("Helsinki-NLP/europarl", "en-fr", split="train[:200_000]")
    for i, item in enumerate(europarl):
        pair = item["translation"]
        src, tgt = pair.get("en", "").strip(), pair.get("fr", "").strip()
        if is_valid_pair(src, tgt):
            all_pairs.append({"translation": {"en": src, "fr": tgt}, "source": "europarl"})
        if (i + 1) % 10000 == 0:
            print("europarl", i + 1, "itens processados")
    
    print(f"Total de pares carregados: {len(all_pairs)}")
    print("embaralhando...")
    random.seed(seed)
    random.shuffle(all_pairs)
    print("retornando dataset...")
    return Dataset.from_list(all_pairs)

def load_validation_dataset() -> Dataset:
    pairs = []
    print("carregando validação...")

    # wmt14 e opus100 têm split "validation" nativo
    for source_name, (repo, config) in {
        "wmt14":   ("wmt/wmt14",            "fr-en"),
        "opus100": ("Helsinki-NLP/opus-100", "en-fr"),
    }.items():
        print(f"carregando {source_name}...")
        raw = load_dataset(repo, config, split="validation")
        for item in raw:
            pair = item["translation"]
            src = pair.get("en", "").strip()
            tgt = pair.get("fr", "").strip()
            if is_valid_pair(src, tgt):
                pairs.append({"translation": {"en": src, "fr": tgt}, "source": source_name})

    # europarl e opus_books só têm "train" — usamos slices do final
    print("carregando europarl val...")
    for item in load_dataset("Helsinki-NLP/europarl", "en-fr", split="train[200_000:220_000]"):
        pair = item["translation"]
        src, tgt = pair.get("en", "").strip(), pair.get("fr", "").strip()
        if is_valid_pair(src, tgt):
            pairs.append({"translation": {"en": src, "fr": tgt}, "source": "europarl"})

    print("carregando opus_books val...")
    for item in load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train[85%:95%]"):
        pair = item["translation"]
        src, tgt = pair.get("en", "").strip(), pair.get("fr", "").strip()
        if is_valid_pair(src, tgt):
            pairs.append({"translation": {"en": src, "fr": tgt}, "source": "opus_books"})

    return Dataset.from_list(pairs)

def load_test_dataset() -> Dataset:
    pairs = []
    print("carregando teste...")

    # wmt14 e opus100 têm split "test" nativo
    for source_name, (repo, config) in {
        "wmt14":   ("wmt/wmt14",            "fr-en"),
        "opus100": ("Helsinki-NLP/opus-100", "en-fr"),
    }.items():
        print(f"carregando {source_name}...")
        raw = load_dataset(repo, config, split="test")
        for item in raw:
            pair = item["translation"]
            src = pair.get("en", "").strip()
            tgt = pair.get("fr", "").strip()
            if is_valid_pair(src, tgt):
                pairs.append({"translation": {"en": src, "fr": tgt}, "source": source_name})

    # europarl e opus_books — slices após a janela de validação
    print("carregando europarl test...")
    for item in load_dataset("Helsinki-NLP/europarl", "en-fr", split="train[220_000:240_000]"):
        pair = item["translation"]
        src, tgt = pair.get("en", "").strip(), pair.get("fr", "").strip()
        if is_valid_pair(src, tgt):
            pairs.append({"translation": {"en": src, "fr": tgt}, "source": "europarl"})

    print("carregando opus_books test...")
    for item in load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train[95%:]"):
        pair = item["translation"]
        src, tgt = pair.get("en", "").strip(), pair.get("fr", "").strip()
        if is_valid_pair(src, tgt):
            pairs.append({"translation": {"en": src, "fr": tgt}, "source": "opus_books"})

    return Dataset.from_list(pairs)

@torch.no_grad()
def generate_sample(model, tokenizer, prompt_src: str, max_new_tokens: int = 50, device="cpu") -> str:
    model.eval()

    src = tokenizer.encode(prompt_src, return_tensors="pt").to(device)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    tgt_input = torch.tensor([[bos_id]], device=device)

    for _ in range(max_new_tokens):
        logits = model(src, tgt_input)
        next_token_logits = logits[:, -1, :] / 0.8
        probs = torch.softmax(next_token_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tgt_input = torch.cat([tgt_input, next_tok], dim=1)

        if next_tok.item() == eos_id:
            break

    out_ids = tgt_input[0].tolist()[1:]
    return tokenizer.decode(out_ids, skip_special_tokens=True)

@torch.no_grad()
def get_grad_norm(model, norm_type=2.0):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type
    ).item()

@torch.inference_mode() 
def generate_batch(model, src, tokenizer, max_len=50):
    model.eval()
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    B = src.size(0)
    tgt = torch.full((B, 1), bos, device=src.device)

    for _ in range(max_len):
        logits = model(src, tgt)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)

        if (next_token == eos).all():
            break

    return tgt[:, 1:]

def compute_bleu_methods(preds, targets):
    # preds: list[str]
    # targets: list[str]
    sacre = SacreBLEU()
    sacre_score = sacre.corpus_score(preds, [targets]).score
    return sacre_score

def perplexity_fn(loss: float) -> float:
    """
    Perplexidade = e^loss.
    Uma loss de 5.0 equivale a ppl ≈ 148.
    """
    return math.exp(loss)

def collect_model_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes

    by_layer = []
    for name, p in model.named_parameters():
        by_layer.append({
            "name": name,
            "shape": list(p.shape),
            "numel": p.numel(),
            "dtype": str(p.dtype),
            "trainable": bool(p.requires_grad),
            "memory_bytes": p.numel() * p.element_size(),
            "memory_mb": (p.numel() * p.element_size()) / (1024 ** 2),
        })

    stats = {
        "parameters": {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
            "total_millions": total / 1e6,
        },
        "memory": {
            "param_bytes": param_bytes,
            "buffer_bytes": buffer_bytes,
            "total_bytes": total_bytes,
            "param_mb": param_bytes / (1024 ** 2),
            "buffer_mb": buffer_bytes / (1024 ** 2),
            "total_mb": total_bytes / (1024 ** 2),
            "total_gb": total_bytes / (1024 ** 3),
        },
        "layers": by_layer,
    }
    return stats

def ensure_dirs():
    Path("checkpoints").mkdir(exist_ok=True)
    Path("checkpoints/plots").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/history").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/models").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    config = {
        "data": {
            "train": {"batch_size": 8,  "seq_len": 128},
            "valid": {"batch_size": 8,  "seq_len": 128},
            "test":  {"batch_size": 8,  "seq_len": 128},
        },
        "model": {
            "max_len": 128,
            "d_model": 512, # head_dim * num_heads == d_model
            "head_dim": 64,  # d_k
            "num_heads": 8,
            "ff_ratio": 4,  # d_model * ff_ratio = d_ff (2048)
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "mask": True,
            "dropout": 0.1,
        },
        "training": {
            "epochs": 100,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "grad_clip": 1,
            "early_stopping_patience": 3,
            "max_train_steps": 2_000,
            "max_valid_steps": 200,
            "max_test_steps": 200,
        },
        "spike": {
            "warmup_steps": 500,
            "window_size":  300,
            "std_factor": 3.0,
            "min_absolute": 10.0,
        },
        "bleu": {
            "n_gram": 4,
        },
        "ruido": {
                "enabled": True,
                "noise_std": 0.01,
                "frequency_steps": 10_000,
        },
        "save": {
            "plot_diagnostics":  "checkpoints/plots/training_diagnostics.png",
            "plot_splits":       "checkpoints/plots/train_valid_test_curves.png",
            "plot_perplexity":   "checkpoints/plots/perplexity_curve.png",
            "plot_bleu":         "checkpoints/plots/bleu_curve.png",
            "plot_epoch_time":   "checkpoints/plots/epoch_time_curve.png",    
            "plot_gpu_memory":   "checkpoints/plots/gpu_memory_curve.png",    
            "plot_model_summary":"checkpoints/plots/model_summary.png",       
            "plot_params_layer": "checkpoints/plots/params_per_layer.png",    
            "plot_memory_layer": "checkpoints/plots/memory_per_layer.png",    
            "history_pt":        "checkpoints/history/training_history.pt",
            "history_json":      "checkpoints/history/spike_steps.json",
            "model_path":        "checkpoints/models/model_last.pt",
            "model_best_path":   "checkpoints/models/model_best.pt",
        }
    }

    ensure_dirs()

    device = config["training"]["device"]

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = load_mixed_dataset(seed=42)
    val_ds   = load_validation_dataset()
    test_ds  = load_test_dataset()

    train_ids = tokenize(train_ds, split_name="train", max_len=config["model"]["max_len"])
    val_ids   = tokenize(val_ds,   split_name="valid", max_len=config["model"]["max_len"])
    test_ids  = tokenize(test_ds,  split_name="test",  max_len=config["model"]["max_len"])
    vocab_size = len(tokenizer)
    
    train_loader, val_loader, test_loader = make_dataloaders(
        train_ids, val_ids, test_ids, config
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = Transformer(
        vocab_size=vocab_size,
        max_len=config["model"]["max_len"],
        d_model=config["model"]["d_model"],
        head_dim=config["model"]["head_dim"],
        num_heads=config["model"]["num_heads"],
        ff_ratio=config["model"]["ff_ratio"],
        num_encoder_layers=config["model"]["num_encoder_layers"],
        num_decoder_layers=config["model"]["num_decoder_layers"],
        mask=config["model"]["mask"],
        dropout=config["model"]["dropout"],
        tie_embeddings=True,
        attn_cls=None,
        attn_kwargs=None,
    ).to(device)

    if device == "cuda":
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM alocada:  {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"VRAM reservada: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=config["training"]["label_smoothing"]
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    steps_per_epoch = min(
        config["training"]["max_train_steps"],
        len(train_loader)
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["training"]["learning_rate"],
        epochs=config["training"]["epochs"],
        steps_per_epoch=steps_per_epoch,
    )
    
    stats = collect_model_stats(model)
    print(f"Total parameters: {stats['parameters']['total']} ({stats['parameters']['total_millions']:.2f}M)")
    print(f"Estimated model memory: {stats['memory']['total_mb']:.2f} MB ({stats['memory']['total_gb']:.2f} GB)")
    print(f"Vocab real: {tokenizer.vocab_size:,} tokens")
    tied = " (com weight tying)" if model.tie_embeddings else ""
    print(f"Params estimados: {tokenizer.vocab_size * 512 / 1e6:.1f}M (embedding + output_layer compartilhados{tied})")
    
    # ── Históricos ────────────────────────────────────────────────────────────
    grad_norm_history      = []
    threshold_history      = []
    train_loss_history     = []

    train_epoch_loss_history  = []
    valid_epoch_loss_history  = []

    train_epoch_ppl_history   = []
    valid_epoch_ppl_history   = []
    
    bleu_epochs = []
    valid_epoch_bleu_history = []

    spike_steps  = []
    global_step  = 0

    best_valid_loss  = float("inf")
    patience         = config["training"]["early_stopping_patience"]
    patience_counter = 0

    epoch_time_history = []
    train_epoch_time_history = []
    valid_epoch_time_history = []
    
    gpu_peak_memory_history = []
    gpu_allocated_end_history = []
    gpu_reserved_end_history = []


    SAMPLE_PROMPTS = [
        # livros
        "He opened the window and listened to the rain.",
        "She turned away without saying a word.",
        "I thought you had already left the city.",
        "The child was sleeping peacefully on the sofa.",
        "They walked slowly through the empty streets.",
        "Why are you looking at me like that?",
        "He took the letter from his pocket and read it again.",
        "For a moment, nobody knew what to say.",
        "She smiled, although she was clearly worried.",
        "It was the kind of silence that made everyone uneasy.",

        # notícias
        "The government announced new measures to reduce inflation.",
        "The president said the negotiations would continue next week.",
        "The company reported a significant increase in quarterly revenue.",
        "Oil prices rose sharply after the latest market update.",
        "The report highlights the need for stronger international cooperation.",
        "Several countries expressed concern over the recent developments.",
        "The central bank decided to keep interest rates unchanged.",
        "According to official figures, unemployment fell in the last quarter.",
        "The European Commission published its annual forecast on Thursday.",
        "The minister denied any involvement in the controversy.",

        # geral / institucional / web
        "Please read the instructions carefully before using this product.",
        "This document provides an overview of the proposed changes.",
        "Users must create an account before accessing the service.",
        "The meeting will begin at 9 a.m. in the main conference room.",
        "Further information is available on the official website.",
        "All applications must be submitted before the deadline.",
        "The system automatically saves your progress every five minutes.",
        "You can contact customer support by email or phone.",
        "The new version includes several performance improvements.",
        "Access to this area is restricted to authorized personnel only."
    ]
    

    
    # ══════════════════════════════════════════════════════════════════════════
    # TREINO
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(config["training"]["epochs"]):
        epoch_start_time = time.perf_counter()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        model.train()
        
        train_start_time = time.perf_counter()
        train_loss_total = 0.0
        train_steps      = 0
        
        train_bar = tqdm(
            train_loader,
            total=steps_per_epoch,
            desc=f"Train {epoch+1}/{config['training']['epochs']}",
            unit="batch"
        )

        train_iter = iter(train_bar)
        
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            
            try:
                X_train, y_train = next(train_iter)
            except StopIteration:
                train_iter = iter(train_bar)
                X_train, y_train = next(train_iter)
            
            X_train = X_train.to(device).long()
            y_train = y_train.to(device).long()

            tgt_input  = y_train[:, :-1]
            tgt_output = y_train[:, 1:]

            logits = model(X_train, tgt_input)
            loss = loss_fn(
                    logits.reshape(-1, vocab_size),
                    tgt_output.reshape(-1)
                )

            loss.backward()

            grad_norm = get_grad_norm(model)
            if config["training"]["grad_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["training"]["grad_clip"])

            # Ruído a cada 10 epocas
            if config["ruido"]["enabled"] and global_step > 0 and global_step % config["ruido"]["frequency_steps"] == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * config["ruido"]["noise_std"]
                        p.grad.add_(noise)
                            
            grad_norm_history.append(grad_norm)
            train_loss_history.append(loss.item())

            is_spike  = False
            threshold = None

            if len(grad_norm_history) > config["spike"]["warmup_steps"]:
                window = config["spike"]["window_size"]
                recent = torch.tensor(grad_norm_history[-window:], dtype=torch.float32)
                mean = recent.mean().item()
                std = recent.std(unbiased=False).item()
                threshold = max(
                    mean + config["spike"]["std_factor"] * std,
                    config["spike"]["min_absolute"]
                )

                if grad_norm > threshold:
                    is_spike = True
                    spike_steps.append({
                        "epoch":       epoch + 1,
                        "global_step": global_step,
                        "train_step":  train_steps,
                        "grad_norm":   float(grad_norm),
                        "threshold":   float(threshold),
                        "loss":        float(loss.item()),
                        "ppl":         perplexity_fn(loss.item()),
                    })

            threshold_history.append(threshold if threshold is not None else float("nan"))

            optimizer.step()
            scheduler.step()

            train_loss_total += loss.item()
            train_steps      += 1
            global_step      += 1

            postfix = {
                "loss":  f"{loss.item():.4f}",
                "ppl":   f"{perplexity_fn(loss.item()):.2f}",
                "grad":  f"{grad_norm:.3f}",
                "spike": "yes" if is_spike else "no",
            }
            if threshold is not None:
                postfix["thr"] = f"{threshold:.3f}"
            train_bar.set_postfix(postfix)

        train_time = time.perf_counter() - train_start_time
        train_epoch_time_history.append(train_time)
        
        train_loss = train_loss_total / max(train_steps, 1)
        train_epoch_loss_history.append(train_loss)
        train_epoch_ppl_history.append(perplexity_fn(train_loss))
        
        
        # ── Validação ─────────────────────────────────────────────────────────
        model.eval()
        
        valid_start_time = time.perf_counter()
        
        valid_loss_total = 0.0
        valid_steps      = 0
        
        valid_steps_target = min(config["training"]["max_valid_steps"], len(val_loader))

        BLEU_EVERY_N_EPOCHS  = 10   # calcula a cada 10 épocas
        BLEU_MAX_BATCHES     = 50  # máximo de batches para BLEU (~800 frases)
        compute_bleu = ((epoch + 1) % BLEU_EVERY_N_EPOCHS == 0)
        
        all_preds = []
        all_targets = []
        
        with torch.inference_mode():
                valid_bar = tqdm(
                    val_loader,
                    total=valid_steps_target,
                    desc=f"Valid {epoch+1}/{config['training']['epochs']}",
                    unit="batch"
                )
                
                bleu_batches = 0 
                
                for step, (X_valid, y_valid) in enumerate(valid_bar):
                    if step >= valid_steps_target:
                        break

                    X_valid = X_valid.to(device).long()
                    y_valid = y_valid.to(device).long()

                    tgt_input_valid  = y_valid[:, :-1]
                    tgt_output_valid = y_valid[:, 1:]

                    valid_logits = model(X_valid, tgt_input_valid)
                    valid_loss = loss_fn(
                            valid_logits.reshape(-1, vocab_size),
                            tgt_output_valid.reshape(-1)
                        )

                    valid_loss_total += valid_loss.item()
                    valid_steps += 1

                    if compute_bleu and bleu_batches < BLEU_MAX_BATCHES:
                        pred_ids = generate_batch(model, X_valid, tokenizer, max_len=128)
                        preds   = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                        targets = tokenizer.batch_decode(y_valid,  skip_special_tokens=True)
                        all_preds.extend(preds)
                        all_targets.extend(targets)
                        bleu_batches += 1
                        
                    valid_bar.set_postfix(
                        loss=f"{valid_loss.item():.4f}",
                        ppl=f"{perplexity_fn(valid_loss.item()):.2f}",
                    )
                    
        valid_loss = valid_loss_total / max(valid_steps, 1)
        
        if compute_bleu and len(all_preds) > 0:
            valid_bleu = compute_bleu_methods(
                all_preds, all_targets
            )
            bleu_epochs.append(epoch + 1)
            valid_epoch_bleu_history.append(valid_bleu)
        else:
            valid_bleu = float("nan")

        print(
        f"[epoch {epoch+1}] "
        f"sacre={valid_bleu}"
        )

        valid_epoch_loss_history.append(valid_loss)
        valid_epoch_ppl_history.append(perplexity_fn(valid_loss))
        
        valid_time = time.perf_counter() - valid_start_time
        valid_epoch_time_history.append(valid_time)
        epoch_total_time = time.perf_counter() - epoch_start_time
        epoch_time_history.append(epoch_total_time)
        
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        else:
            peak_mb = 0.0
            allocated_mb = 0.0
            reserved_mb = 0.0
            
        gpu_peak_memory_history.append(peak_mb)
        gpu_allocated_end_history.append(allocated_mb)
        gpu_reserved_end_history.append(reserved_mb)
        print(
            f"Epoch: {epoch+1} | "
            f"Train loss: {train_loss:.5f}, Train ppl: {perplexity_fn(train_loss):.2f} | "
            f"Valid loss: {valid_loss:.5f}, Valid ppl: {perplexity_fn(valid_loss):.2f} | "
            f"Spikes acumulados: {len(spike_steps)} | "
            f"SacreBLEU: {valid_bleu:.4f} | "
            f"Tempo treino: {train_time:.2f}s | "
            f"Tempo valid: {valid_time:.2f}s | "
            f"Tempo total: {epoch_total_time:.2f}s | "
            f"GPU peak: {peak_mb:.2f}MB"
        )
        # ── Early stopping ────────────────────────────────────────────────────
        
        if valid_loss < best_valid_loss:
            best_valid_loss  = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), config["save"]["model_best_path"])
            print(f"Novo melhor modelo salvo com loss {best_valid_loss:.4f}")
        else:
            patience_counter += 1

        # ── Amostras geradas ──────────────────────────────────────────────────
        print(f"\n--- Amostras geradas (época {epoch+1}) ---")
        for prompt in SAMPLE_PROMPTS:
            out = generate_sample(model, tokenizer, prompt, max_new_tokens=40, device=device)
            print(f"Prompt : {prompt}\nGerado : {out}\n")
        model.train()

    # ══════════════════════════════════════════════════════════════════════════
    # TESTE
    # ══════════════════════════════════════════════════════════════════════════
    model.eval()
    test_loss_total = 0.0
    test_steps      = 0

    test_steps_target = min(config["training"]["max_test_steps"], len(test_loader))

    test_all_preds = []
    test_all_targets = []

    with torch.inference_mode():
        test_bar = tqdm(
            test_loader,
            total=test_steps_target,
            desc="Test",
            unit="batch"
        )

        for step, (X_test, y_test) in enumerate(test_bar):
            if step >= test_steps_target:
                break

            X_test = X_test.to(device).long()
            y_test = y_test.to(device).long()

            tgt_input_test  = y_test[:, :-1]
            tgt_output_test = y_test[:, 1:]


            test_logits = model(X_test, tgt_input_test)
            test_loss = loss_fn(
                    test_logits.reshape(-1, vocab_size),
                    tgt_output_test.reshape(-1)
                )

            test_loss_total += test_loss.item()
            test_steps += 1

            pred_ids = generate_batch(model, X_test, tokenizer, max_len=config["data"]["test"]["seq_len"])
            preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            targets = tokenizer.batch_decode(y_test, skip_special_tokens=True)
            
            test_all_preds.extend(preds)
            test_all_targets.extend(targets)

            test_bar.set_postfix(
                loss=f"{test_loss.item():.4f}",
                ppl=f"{perplexity_fn(test_loss.item()):.2f}",
            )
        
    if len(test_all_preds) > 0:
        test_bleu = compute_bleu_methods(
            test_all_preds, test_all_targets
        )
    else:
        test_bleu = float("nan")

    test_loss = test_loss_total / max(test_steps, 1)

    print(
        f"Test loss: {test_loss:.5f}, "
        f"Test ppl: {perplexity_fn(test_loss):.2f}, "
        f"Test BLEU (SacreBLEU): {test_bleu:.4f}"
    )
    print(f"Total de spikes detectados: {len(spike_steps)}")

    if spike_steps:
        print("Primeiros spikes:")
        for s in spike_steps[:10]:
            print(
                f"epoch={s['epoch']} | global_step={s['global_step']} | "
                f"grad_norm={s['grad_norm']:.4f} | threshold={s['threshold']:.4f} | "
                f"loss={s['loss']:.4f} | ppl={s['ppl']:.2f}"
            )
    print("SacreBLEU history:", valid_epoch_bleu_history)
    # ── Salva modelo ──────────────────────────────────────────────────────────
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config":               config,
            "vocab_size":           vocab_size,
            "test_loss":            test_loss,
            "test_ppl":             perplexity_fn(test_loss),
            "test_bleu": test_bleu,
        },
        config["save"]["model_path"]
    )

    # ── Salva histórico ───────────────────────────────────────────────────────
    model_stats = collect_model_stats(model)
    torch.save(
        {
        "model_stats": model_stats,
        "training_results": {
                "train_loss_history":      train_loss_history,
                "grad_norm_history":       grad_norm_history,
                "threshold_history":       threshold_history,
                "train_epoch_loss_history": train_epoch_loss_history,
                "valid_epoch_loss_history": valid_epoch_loss_history,
                "train_epoch_ppl_history":  train_epoch_ppl_history,
                "valid_epoch_ppl_history":  valid_epoch_ppl_history,
                "test_loss":               test_loss,
                "test_ppl":                perplexity_fn(test_loss),
                "test_bleu":                    test_bleu,
                "valid_epoch_bleu_history":     valid_epoch_bleu_history,
                "bleu_epochs": bleu_epochs,
                "spikes": spike_steps,
            },
        "timing": {
            "epoch_time_history_sec": epoch_time_history,
            "train_epoch_time_history_sec": train_epoch_time_history,
            "valid_epoch_time_history_sec": valid_epoch_time_history,
        },
        "gpu_memory": {
            "peak_mb_per_epoch": gpu_peak_memory_history,
            "allocated_mb_end_per_epoch": gpu_allocated_end_history,
            "reserved_mb_end_per_epoch": gpu_reserved_end_history,
        },
        },
        config["save"]["history_pt"]
    )

    with open(config["save"]["history_json"], "w", encoding="utf-8") as f:
        json.dump(spike_steps, f, ensure_ascii=False, indent=2)
        

    # ══════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════════════
    epochs_axis = list(range(1, len(train_epoch_loss_history) + 1))

    # ── Diagnóstico: loss + grad norm por step ────────────────────────────────
    if grad_norm_history:
        fig, axes = plt.subplots(2, 1, figsize=(22, 12), sharex=True)

        # Paleta elegante
        loss_raw_color = "#6B7280"      # cinza frio suave
        loss_main_color = "#111827"     # quase preto, mais refinado
        grad_raw_color = "#5F6B73"      # ardósia suave
        grad_main_color = "#1F2933"     # grafite elegante
        threshold_color = "#5B7C5A"     # verde dessaturado
        spike_color = "#C08A2E"         # dourado queimado

        # Painel 1: Loss
        axes[0].plot(
            train_loss_history,
            color=loss_raw_color,
            alpha=0.18,
            linewidth=0.8,
            label="Train loss",
            zorder=1
        )

        if spike_steps:
            sx = [s["global_step"] for s in spike_steps]
            axes[0].scatter(
                sx,
                [s["loss"] for s in spike_steps],
                color=spike_color,
                s=26,
                alpha=0.95,
                edgecolors="white",
                linewidths=0.35,
                label="Spikes",
                zorder=3
            )
            # Legendas com contagem de spikes
            spike_count = len(spike_steps)
            axes[0].legend(frameon=True, framealpha=0.9, title=f"Spikes: {spike_count}")

        axes[0].set_title("Loss por step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.22, linewidth=0.8)
        axes[0].legend(frameon=True, framealpha=0.9)

        # Painel 2: Grad norm
        axes[1].plot(
            grad_norm_history,
            color=grad_raw_color,
            alpha=0.18,
            linewidth=0.75,
            label="Grad norm",
            zorder=1
        )

        thr_plot = [v if not math.isnan(v) else None for v in threshold_history]
        axes[1].plot(
            thr_plot,
            color=threshold_color,
            linewidth=1.15,
            linestyle="--",
            alpha=0.95,
            label="Threshold (rolling)",
            zorder=2
        )

        if spike_steps:
            axes[1].scatter(
                sx,
                [s["grad_norm"] for s in spike_steps],
                color=spike_color,
                s=24,
                alpha=0.95,
                edgecolors="white",
                linewidths=0.35,
                label="Spikes",
                zorder=3
            )
            # Legendas com contagem de spikes
            spike_count = len(spike_steps)
            axes[1].legend(frameon=True, framealpha=0.9, title=f"Spikes: {spike_count}")

        axes[1].set_title("Grad norm por step")
        axes[1].set_xlabel("Global step")
        axes[1].set_ylabel("Grad norm")
        axes[1].grid(True, alpha=0.22, linewidth=0.8)
        axes[1].legend(frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(config["save"]["plot_diagnostics"], dpi=300, bbox_inches="tight")
        plt.close()

    # ── Loss por época ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 12))
    ax.plot(epochs_axis, train_epoch_loss_history, marker="o", linewidth=2, label="Train loss", color="tab:green")
    ax.plot(epochs_axis, valid_epoch_loss_history, marker="s", linewidth=2, label="Valid loss", color="tab:purple")
    ax.axhline(y=test_loss, color="tab:pink", linestyle="--", linewidth=2,
            label=f"Test loss = {test_loss:.4f}")
    ax.set_title("Loss por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["save"]["plot_splits"], dpi=300, bbox_inches="tight")
    plt.close()

    # ── Perplexidade por época ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 12))
    ax.plot(epochs_axis, train_epoch_ppl_history, marker="o", linewidth=2, label="Train ppl", color="tab:green")
    ax.plot(epochs_axis, valid_epoch_ppl_history, marker="s", linewidth=2, label="Valid ppl", color="tab:purple")
    ax.axhline(y=perplexity_fn(test_loss), color="tab:pink", linestyle="--", linewidth=2,
            label=f"Test ppl = {perplexity_fn(test_loss):.2f}")
    ax.set_title("Perplexidade por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perplexidade")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["save"]["plot_perplexity"], dpi=300, bbox_inches="tight")
    plt.close()

    # ── BLEU por época ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 12))
    if bleu_epochs:
        ax.plot(bleu_epochs, valid_epoch_bleu_history, marker="o", linewidth=2, color="tab:blue", label="SacreBLEU")
        ax.set_title("BLEU (SacreBLEU) por época")
        ax.set_xlabel("Época")
        ax.set_ylabel("SacreBLEU")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(config["save"]["plot_bleu"], dpi=300, bbox_inches="tight")
        plt.close()
    
    # ── Tempos por época
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_axis, train_epoch_time_history, marker="o", linewidth=2, label="Train time", color="tab:blue")
    ax.plot(epochs_axis, valid_epoch_time_history, marker="s", linewidth=2, label="Valid time", color="tab:orange")
    ax.plot(epochs_axis, epoch_time_history, marker="^", linewidth=2, label="Total epoch time", color="tab:green")
    ax.set_title("Tempo por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Tempo (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["save"]["plot_epoch_time"], dpi=300, bbox_inches="tight")
    plt.close()
    
    # ── Memória GPU por época ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs_axis, gpu_peak_memory_history, marker="o", linewidth=2, label="Peak allocated", color="tab:red")
    ax.plot(epochs_axis, gpu_allocated_end_history, marker="s", linewidth=2, label="Allocated end", color="tab:blue")
    ax.plot(epochs_axis, gpu_reserved_end_history, marker="^", linewidth=2, label="Reserved end", color="tab:green")
    ax.set_title("Memória GPU por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Memória (MB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["save"]["plot_gpu_memory"], dpi=300, bbox_inches="tight")
    plt.close()

    # ── Resumo geral do modelo ───────────────────────────────────────────────────
    summary_labels = ["Parâmetros totais (M)", "Memória params (MB)", "Memória buffers (MB)", "Memória total (MB)"]
    summary_values = [
        model_stats["parameters"]["total_millions"],
        model_stats["memory"]["param_mb"],
        model_stats["memory"]["buffer_mb"],
        model_stats["memory"]["total_mb"],
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(summary_labels, summary_values, color=["tab:blue", "tab:orange", "tab:green", "tab:red"], alpha=0.85)
    ax.set_title("Resumo de parâmetros e memória do modelo")
    ax.set_ylabel("Valor")
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(config["save"]["plot_model_summary"], dpi=300, bbox_inches="tight")
    plt.close()
    
    # ── Parâmetros e memória por layer ───────────────────────────────────────────
    layers = model_stats["layers"]

    layer_names = [x["name"] for x in layers]
    layer_numel = [x["numel"] for x in layers]
    layer_memory_mb = [x["memory_mb"] for x in layers]

    # opcional: mostrar apenas top-N para ficar legível
    top_n = 30
    sorted_idx = sorted(range(len(layer_numel)), key=lambda i: layer_numel[i], reverse=True)[:top_n]

    top_names = [layer_names[i] for i in sorted_idx]
    top_numel = [layer_numel[i] for i in sorted_idx]
    top_mem = [layer_memory_mb[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.barh(top_names, top_numel, color="tab:blue", alpha=0.85)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} parâmetros por layer")
    ax.set_xlabel("Número de parâmetros")
    ax.set_ylabel("Layer")
    ax.grid(True, axis="x", alpha=0.3)

    for bar in bars:
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height()/2, f" {w:,}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(config["save"]["plot_params_layer"], dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.barh(top_names, top_mem, color="tab:red", alpha=0.85)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} memória por layer")
    ax.set_xlabel("Memória (MB)")
    ax.set_ylabel("Layer")
    ax.grid(True, axis="x", alpha=0.3)

    for bar in bars:
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height()/2, f" {w:.2f} MB", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(config["save"]["plot_memory_layer"], dpi=300, bbox_inches="tight")
    plt.close()