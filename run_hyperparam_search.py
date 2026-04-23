"""
run_hyperparam_search.py
========================
Chạy grid/random search siêu tham số cho LlamaEmbeddingClassifier (finetune)
và xuất bảng so sánh ra CSV.

Cách dùng:
    # SST
    python run_hyperparam_search.py \
        --dataset sst \
        --train data/sst-train.txt \
        --dev   data/sst-dev.txt   \
        --test  data/sst-test.txt  \
        --pretrained_model_path stories42M.pt [--use_gpu]

    # CFIMDB
    python run_hyperparam_search.py \
        --dataset cfimdb \
        --train data/cfimdb-train.txt \
        --dev   data/cfimdb-dev.txt   \
        --test  data/cfimdb-test.txt  \
        --pretrained_model_path stories42M.pt [--use_gpu]

Kết quả:
    hyperparam_results_<dataset>.csv  – bảng CSV đầy đủ, lưu sau mỗi thử nghiệm
"""

import argparse
import csv
import itertools
import os
import random
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── import từ project ──────────────────────────────────────────────────────────
from classifier import LlamaEmbeddingClassifier
from optimizer import AdamW
from tokenizer import Tokenizer
from run_llama import LlamaDataset, create_data, model_eval, seed_everything

HYPERPARAM_GRID = {
    "lr":                  [1e-5, 2e-5, 5e-5],  # learning rate
    "epochs":              [5, 10],               # số epoch
    "hidden_dropout_prob": [0.1, 0.3],           # dropout
}

BATCH_SIZE = {"sst": 64, "cfimdb": 8}

FIXED = {
    "option":           "finetune",
    "max_sentence_len": None,
    "seed":             1337,
}

TQDM_DISABLE = False   


# ══════════════════════════════════════════════════════════════════════════════
#  HÀM TRAIN + EVAL MỘT BỘ SIÊU THAM SỐ
# ══════════════════════════════════════════════════════════════════════════════
def run_single(combo: dict, args) -> dict:
    """
    Train và eval một bộ siêu tham số.
    Trả về dict kết quả gồm dev_acc, dev_f1, test_acc, test_f1, train_time.
    """
    seed_everything(FIXED["seed"])
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    tokenizer = Tokenizer(FIXED["max_sentence_len"])
    bs = BATCH_SIZE[args.dataset]

    # ── load data ──
    train_data, num_labels = create_data(args.train, tokenizer, "train")
    dev_data   = create_data(args.dev,  tokenizer, "valid")
    test_data  = create_data(args.test, tokenizer, "test")

    ns = SimpleNamespace(max_sentence_len=FIXED["max_sentence_len"])
    train_ds = LlamaDataset(train_data, ns)
    dev_ds   = LlamaDataset(dev_data,   ns)
    test_ds  = LlamaDataset(test_data,  ns)

    train_loader = DataLoader(train_ds, shuffle=True,  batch_size=bs, collate_fn=train_ds.collate_fn)
    dev_loader   = DataLoader(dev_ds,   shuffle=False, batch_size=bs, collate_fn=dev_ds.collate_fn)
    test_loader  = DataLoader(test_ds,  shuffle=False, batch_size=bs, collate_fn=test_ds.collate_fn)

    # ── khởi tạo model ──
    config = SimpleNamespace(
        hidden_dropout_prob=combo["hidden_dropout_prob"],
        pretrained_model_path=args.pretrained_model_path,
        num_labels=num_labels,
        data_dir=".",
        option=FIXED["option"],
    )
    model = LlamaEmbeddingClassifier(config).to(device)
    optimizer = AdamW(model.parameters(), lr=combo["lr"])

    best_dev_acc = 0.0
    best_dev_f1  = 0.0

    t0 = time.time()
    for epoch in range(combo["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"  epoch {epoch+1}", disable=TQDM_DISABLE):
            b_ids    = batch["token_ids"].to(device)
            b_labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(b_ids)
            loss   = F.nll_loss(logits, b_labels.view(-1), reduction="sum") / bs
            loss.backward()
            optimizer.step()

        dev_acc, dev_f1, *_ = model_eval(dev_loader, model, device)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_dev_f1  = dev_f1

    train_time = time.time() - t0

    test_acc, test_f1, *_ = model_eval(test_loader, model, device)

    return {
        **combo,
        "batch_size":    bs,
        "best_dev_acc":  round(best_dev_acc, 4),
        "best_dev_f1":   round(best_dev_f1,  4),
        "test_acc":      round(test_acc,      4),
        "test_f1":       round(test_f1,       4),
        "train_time_s":  round(train_time,    1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  XUẤT BẢNG CSV
# ══════════════════════════════════════════════════════════════════════════════
RESULT_COLS = [
    "lr", "epochs", "batch_size", "hidden_dropout_prob",
    "best_dev_acc", "best_dev_f1", "test_acc", "test_f1", "train_time_s",
]

def save_csv(results: list[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()
        writer.writerows(results)
    print(f"[✓] Đã lưu CSV  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def get_args():
    p = argparse.ArgumentParser(description="Hyperparameter sweep (finetune only) cho LlamaEmbeddingClassifier")
    p.add_argument("--dataset", required=True, choices=["sst", "cfimdb"],
                   help="Tên dataset – quyết định batch_size (sst=64, cfimdb=8)")
    p.add_argument("--train",  default=None)
    p.add_argument("--dev",    default=None)
    p.add_argument("--test",   default=None)
    p.add_argument("--pretrained_model_path", default="stories42M.pt")
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--out_csv", default=None,
                   help="Tên file CSV output (mặc định: hyperparam_results_<dataset>.csv)")
    p.add_argument("--max_trials", type=int, default=None,
                   help="Nếu đặt, chỉ chạy N bộ tham số ngẫu nhiên (random search)")
    return p.parse_args()


def main():
    args = get_args()

    # Điền đường dẫn data mặc định theo dataset
    if args.train is None:
        args.train = f"data/{args.dataset}-train.txt"
    if args.dev is None:
        args.dev   = f"data/{args.dataset}-dev.txt"
    if args.test is None:
        args.test  = f"data/{args.dataset}-test.txt"
    if args.out_csv is None:
        args.out_csv = f"hyperparam_results_{args.dataset}.csv"

    bs = BATCH_SIZE[args.dataset]
    print(f"Dataset: {args.dataset.upper()}  |  batch_size={bs}  |  option=finetune")

    # Tạo tất cả tổ hợp siêu tham số
    keys   = list(HYPERPARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*HYPERPARAM_GRID.values())]

    # Random search nếu max_trials được đặt
    if args.max_trials and args.max_trials < len(combos):
        random.seed(42)
        combos = random.sample(combos, args.max_trials)
        print(f"[Random search] Chọn {len(combos)} / {len(list(itertools.product(*HYPERPARAM_GRID.values())))} tổ hợp")
    else:
        print(f"[Grid search] Tổng số thử nghiệm: {len(combos)}")

    results = []
    for i, combo in enumerate(combos, 1):
        desc = ", ".join(f"{k}={v}" for k, v in combo.items())
        print(f"\n{'='*55}")
        print(f"[{i}/{len(combos)}] {desc}")
        print(f"{'='*55}")
        try:
            result = run_single(combo, args)
            results.append(result)
            print(f"  → dev_acc={result['best_dev_acc']:.4f}  test_acc={result['test_acc']:.4f}  "
                  f"time={result['train_time_s']:.0f}s")
            # Lưu ngay sau mỗi thử nghiệm để không mất dữ liệu nếu bị ngắt
            save_csv(results, args.out_csv)
        except Exception as e:
            print(f"  [LỖI] {e}")

    # Sắp xếp theo dev_acc giảm dần rồi lưu + in tổng kết
    results.sort(key=lambda r: r["best_dev_acc"], reverse=True)
    save_csv(results, args.out_csv)


if __name__ == "__main__":
    main()