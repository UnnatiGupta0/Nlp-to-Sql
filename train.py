"""Reproducible experiment: baseline vs. targeted-data T5 NLP-to-SQL model.

1. Build the SQLite database from company_data_dirty.json.
2. Generate question->SQL pairs and hold out a fixed TEST set (never used for training).
3. Baseline: fine-tune t5-small on simple single-table pairs only (what v1 did).
4. Error analysis on the test set -> failures concentrate in aggregations, joins, subqueries.
5. Improved: add training pairs that target those patterns, retrain from scratch.
6. Score both on the SAME held-out test set with execution accuracy.

Usage:  python train.py            (CPU ~25 min, Colab GPU ~3 min)
        python train.py --quick    (tiny smoke test)
"""
import argparse
import json
import time

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          T5ForConditionalGeneration, T5Tokenizer)

from nlp2sql import execution_match, load_db, make_pairs, normalise

PREFIX = "translate English to SQL: "
torch.manual_seed(0)


class PairDataset(Dataset):
    def __init__(self, df, tok):
        self.enc = [tok(PREFIX + q, max_length=64, truncation=True) for q in df["question"]]
        self.lab = [tok(s, max_length=128, truncation=True)["input_ids"] for s in df["sql"]]

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        return {**self.enc[i], "labels": self.lab[i]}


def load_tokenizer():
    tok = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    tok.add_tokens(["<"])  # T5's vocabulary has no '<', which SQL needs (e.g. dob < '1990-01-01')
    return tok


def train(train_df, tok, out_dir, epochs):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.resize_token_embeddings(len(tok))
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=16, num_train_epochs=epochs,
        learning_rate=1e-3, save_strategy="no", logging_steps=50, report_to="none", seed=0)
    Seq2SeqTrainer(model=model, args=args, train_dataset=PairDataset(train_df, tok),
                   data_collator=DataCollatorForSeq2Seq(tok, model=model)).train()
    return model


@torch.no_grad()
def predict(model, tok, questions, bs=32):
    model.eval()
    out = []
    for i in range(0, len(questions), bs):
        batch = tok([PREFIX + q for q in questions[i:i + bs]], return_tensors="pt", padding=True)
        gen = model.generate(**batch, max_new_tokens=128)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def score(conn, test_df, preds):
    df = test_df.copy()
    df["pred"] = preds
    df["exec_ok"] = [execution_match(conn, p, g) for p, g in zip(df["pred"], df["sql"])]
    df["exact_ok"] = [normalise(p) == normalise(g) for p, g in zip(df["pred"], df["sql"])]
    by_cat = {k: round(float(x), 3) for k, x in df.groupby("category")["exec_ok"].mean().items()}
    return df, {"execution_accuracy": round(float(df["exec_ok"].mean()), 3),
                "exact_match": round(float(df["exact_ok"].mean()), 3), "by_category": by_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    n_train, n_test, epochs = (100, 20, 1) if a.quick else (1000, 150, a.epochs)

    conn, tables = load_db()
    pool = make_pairs(conn, tables, n_simple=n_train + n_test, n_complex=n_train + n_test, seed=42)
    simple, complex_ = pool[pool.category == "simple"], pool[pool.category != "simple"]
    test = pd.concat([simple.sample(n_test, random_state=1), complex_.sample(n_test, random_state=1)])
    rest = pool.drop(test.index)
    assert not set(test.question) & set(rest.question), "test leaked into training"

    baseline_train = rest[rest.category == "simple"].head(n_train)
    improved_train = pd.concat([baseline_train, rest[rest.category != "simple"].head(n_train)])
    tok = load_tokenizer()
    results = {"test_set": {"size": len(test), **test.category.value_counts().to_dict()}}

    for name, tr in (("baseline", baseline_train), ("improved", improved_train)):
        t0 = time.time()
        model = train(tr, tok, f"runs/{name}", epochs)
        df, res = score(conn, test, predict(model, tok, test.question.tolist()))
        res.update(train_pairs=len(tr), train_minutes=round((time.time() - t0) / 60, 1))
        results[name] = res
        df.to_csv(f"results/{name}_predictions.csv", index=False)
        print(name, json.dumps(res, indent=1))
        if name == "improved":
            model.save_pretrained("t5_sql_model")
            tok.save_pretrained("t5_sql_model")

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
