import os
from pathlib import Path
import functools
import torch
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm.auto import tqdm

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tokenizers.normalizers import NFKC, Sequence as NormSequence
from transformers import PreTrainedTokenizerFast


# ── Config ─────────────────────────────────────────────────────────────
CACHE_DIR = Path("checkpoints/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_PATH = CACHE_DIR / "tokenizer_fr_en.json"

SRC_LANG = "en"   
TGT_LANG = "fr"   

VOCAB_SIZE = 65_000
MIN_FREQ = 2
DEFAULT_MAX_LEN = 512

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

NUM_WORKERS = 0 if os.name == "nt" else 2  # 0 no Windows, 2 no Linux/Mac


# ── Treino do tokenizer ────────────────────────────────────────────────
def _text_iterator(dataset, batch_size=10_000):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]["translation"]
        for pair in batch:
            yield pair[SRC_LANG]
            yield pair[TGT_LANG]


def train_tokenizer(dataset, vocab_size=VOCAB_SIZE) -> Tokenizer:
    print("[tokenizer] Treinando BPE compartilhado...")

    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tok.normalizer = NormSequence([NFKC()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tok.train_from_iterator(_text_iterator(dataset), trainer=trainer)

    bos_id = tok.token_to_id(BOS_TOKEN)
    eos_id = tok.token_to_id(EOS_TOKEN)

    tok.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, bos_id),
            (EOS_TOKEN, eos_id),
        ],
    )

    tok.save(str(TOKENIZER_PATH))
    print(f"[tokenizer] Salvo em {TOKENIZER_PATH}")
    return tok


def get_tokenizer() -> PreTrainedTokenizerFast:
    if TOKENIZER_PATH.exists():
        print(f"[tokenizer] Carregando cache: {TOKENIZER_PATH}")
        tok = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
    else:
        opus100   = load_dataset("Helsinki-NLP/opus-100",       "en-fr",  split="train[:20_000]")
        books     = load_dataset("Helsinki-NLP/opus_books",     "en-fr",  split="train[:20_000]")
        wmt14     = load_dataset("wmt/wmt14",                   "fr-en",  split="train[:20_000]")
        europarl  = load_dataset("Helsinki-NLP/europarl",       "en-fr",  split="train[:20_000]")

        mixed = []
        for ds in [opus100, books, wmt14, europarl]:
            for item in ds:
                pair = item["translation"]
                en = pair.get("en", "").strip()
                fr = pair.get("fr", "").strip()
                if en and fr:
                    mixed.append({"translation": {"en": en, "fr": fr}})

        mixed = HFDataset.from_list(mixed)
        train_tokenizer(mixed)
        tok = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))

    tok.add_special_tokens({
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
    })
    tok.padding_side = "right"
    return tok



tokenizer = get_tokenizer()


# ── Tokenização + cache ────────────────────────────────────────────────
def tokenize(dataset, split_name: str, max_len: int = DEFAULT_MAX_LEN):
    cache_path = CACHE_DIR / f"{split_name}_{SRC_LANG}_{TGT_LANG}_{max_len}.pt"

    if cache_path.exists():
        print(f"[tokenizer] Carregando cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"[tokenizer] Tokenizando split '{split_name}'...")

    src_ids = []
    tgt_ids = []

    for pair in tqdm(dataset["translation"], desc=f"Tokenizando {split_name}"):
        src_text = pair[SRC_LANG]
        tgt_text = pair[TGT_LANG]

        src_enc = tokenizer(
            src_text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        tgt_enc = tokenizer(
            tgt_text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        src_ids.append(src_enc["input_ids"].squeeze(0).to(torch.int32))
        tgt_ids.append(tgt_enc["input_ids"].squeeze(0).to(torch.int32))

    data = {
        "src": torch.stack(src_ids),   # (N, max_len)
        "tgt": torch.stack(tgt_ids),   # (N, max_len)
    }

    torch.save(data, cache_path)
    print(
        f"[tokenizer] {split_name}: "
        f"src={tuple(data['src'].shape)} | tgt={tuple(data['tgt'].shape)} "
        f"-> salvo em {cache_path}"
    )
    return data


# ── Batch sampler ──────────────────────────────────────────────────────
def get_batch(
    data,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
):
    n = data["src"].size(0)
    idx = torch.randint(0, n, (batch_size,))

    src = data["src"][idx][:, :block_size]
    tgt = data["tgt"][idx][:, :block_size]

    return src.to(device), tgt.to(device)




class TranslationDataset(TorchDataset):
    def __init__(self, data: dict, block_size: int):
        self.src = data["src"]   # (N, max_len)
        self.tgt = data["tgt"]   # (N, max_len)
        self.block_size = block_size

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, idx):
        return (
            self.src[idx][:self.block_size],
            self.tgt[idx][:self.block_size],
        )

def make_dataloaders(train_ids, val_ids, test_ids, config):
    train_ds = TranslationDataset(train_ids, config["data"]["train"]["seq_len"])
    val_ds   = TranslationDataset(val_ids,   config["data"]["valid"]["seq_len"])
    test_ds  = TranslationDataset(test_ids,  config["data"]["test"]["seq_len"])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["train"]["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,        # paralelo — ajuste conforme sua CPU
        pin_memory=True,      # mais rápido para GPU
        drop_last=True,       # evita batch incompleto no final
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["valid"]["batch_size"],
        shuffle=False,        # validação não precisa shuffle
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["data"]["test"]["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader