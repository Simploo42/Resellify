"""
Train a spaCy NER model on title_engine BIO-labeled data.

Usage:
  python -m src.title_engine.trainer
  python -m src.title_engine.trainer --data data/title_engine/train_v0.jsonl
                                     --output models/title_ner
                                     --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from spacy.tokens import Doc, DocBin, Span
from spacy.util import filter_spans

from .tokenizer import bio_to_spans, tokenize

DATA_PATH = Path("data/title_engine/train_v0.jsonl")
MODEL_DIR = Path("models/title_ner")
SEED = 42

TOK2VEC_WIDTH = 96


def _spacy_config(train_path: str, dev_path: str) -> str:
    return f"""\
[paths]
train = "{train_path}"
dev = "{dev_path}"

[system]
gpu_allocator = null
seed = {SEED}

[nlp]
lang = "xx"
pipeline = ["tok2vec","ner"]
batch_size = 128

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = {TOK2VEC_WIDTH}
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
rows = [5000,2500,2500,2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = {TOK2VEC_WIDTH}
depth = 4
window_size = 1
maxout_pieces = 3

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${{components.tok2vec.model.encode.width}}
upstream = "*"

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${{system.seed}}
gpu_allocator = ${{system.gpu_allocator}}
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_steps = 0
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.0001
learn_rate = 0.001

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[pretraining]

[initialize]
vectors = null
init_tok2vec = null
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""


def _make_docbin(records: list[dict]) -> tuple[DocBin, dict[str, int]]:
    import spacy
    nlp = spacy.blank("xx")
    db = DocBin()
    label_counts: dict[str, int] = {}
    skipped = 0

    for rec in records:
        tokens = rec["tokens"]
        tags = rec["tags"]
        if len(tokens) != len(tags):
            skipped += 1
            continue

        doc = Doc(nlp.vocab, words=tokens)
        spans = []
        for ts in bio_to_spans(tags):
            sp = Span(doc, ts.start, ts.end, label=ts.label)
            spans.append(sp)
            label_counts[ts.label] = label_counts.get(ts.label, 0) + 1

        doc.ents = filter_spans(spans)
        db.add(doc)

    if skipped:
        print(f"[Trainer] Skipped {skipped} records with token/tag length mismatch")
    return db, label_counts


def train(data_path: Path, model_dir: Path, seed: int = SEED):
    import spacy
    from spacy.cli.train import train as spacy_train

    records = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"[Trainer] Loaded {len(records)} records from {data_path}")

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_recs = [records[i] for i in indices[:split]]
    dev_recs = [records[i] for i in indices[split:]]
    print(f"[Trainer] Split: {len(train_recs)} train / {len(dev_recs)} dev")

    # Warn on low-count labels
    _, all_counts = _make_docbin(records)
    for label, count in sorted(all_counts.items()):
        if count < 50:
            print(f"[Trainer] WARNING: label {label!r} has only {count} examples — F1 may be unreliable")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        train_db, _ = _make_docbin(train_recs)
        dev_db, _ = _make_docbin(dev_recs)

        train_path_tmp = tmp / "train.spacy"
        dev_path_tmp = tmp / "dev.spacy"
        train_db.to_disk(train_path_tmp)
        dev_db.to_disk(dev_path_tmp)

        config_text = _spacy_config(str(train_path_tmp), str(dev_path_tmp))
        config_path = tmp / "config.cfg"
        config_path.write_text(config_text)

        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Trainer] Starting spaCy training → {model_dir}")
        spacy_train(config_path, output_path=model_dir, use_gpu=-1, overrides={})

    print(f"\n[Trainer] Training complete. Best model at {model_dir / 'model-best'}")
    _report_f1(model_dir / "model-best", dev_recs)


def _report_f1(model_path: Path, dev_recs: list[dict]):
    import spacy
    from spacy.training import Example

    nlp = spacy.load(str(model_path))
    examples = []
    for rec in dev_recs:
        tokens = rec["tokens"]
        tags = rec["tags"]
        if len(tokens) != len(tags):
            continue

        ref = Doc(nlp.vocab, words=tokens)
        pred = Doc(nlp.vocab, words=tokens)

        spans = [Span(ref, ts.start, ts.end, label=ts.label) for ts in bio_to_spans(tags)]
        ref.ents = filter_spans(spans)

        for _, pipe in nlp.pipeline:
            pred = pipe(pred)

        examples.append(Example(pred, ref))

    scores = nlp.evaluate(examples)
    print("\n── Per-entity F1 (dev set) ─────────────────────")
    for label, m in sorted(scores.get("ents_per_type", {}).items()):
        print(f"  {label:<15} P={m['p']:.3f}  R={m['r']:.3f}  F={m['f']:.3f}")
    print(f"\n  Overall  F={scores.get('ents_f', 0):.3f}  "
          f"P={scores.get('ents_p', 0):.3f}  R={scores.get('ents_r', 0):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spaCy NER on title_engine data")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--output", default=str(MODEL_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(Path(args.data), Path(args.output), seed=args.seed)
