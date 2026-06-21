"""
Inference helper — loads the trained spaCy NER model and tags raw titles
using the same frozen v1 tokenizer used at training time.

Usage:
  from src.title_engine.inference import TitleNER
  ner = TitleNER()
  spans = ner.tag("iPhone 13 128GB negru")

CLI:
  python -m src.title_engine.inference "iPhone 13 128GB negru"
"""
from __future__ import annotations

from pathlib import Path

from spacy.tokens import Doc, Span

from .tokenizer import TokenSpan, tokenize

DEFAULT_MODEL = Path("models/title_ner/model-best")


class TitleNER:
    def __init__(self, model_path: Path = DEFAULT_MODEL):
        import spacy
        self.nlp = spacy.load(str(model_path))

    def tag(self, raw: str) -> list[TokenSpan]:
        """
        Tokenize with the frozen v1 tokenizer, run NER, return TokenSpan list.
        Never passes raw text through spaCy's tokenizer.
        """
        tokens = tokenize(raw)
        doc = Doc(self.nlp.vocab, words=tokens)
        for _, pipe in self.nlp.pipeline:
            doc = pipe(doc)
        return [TokenSpan(e.start, e.end, e.label_) for e in doc.ents]

    def tag_bio(self, raw: str) -> tuple[list[str], list[str]]:
        """Returns (tokens, bio_tags) for the raw title."""
        tokens = tokenize(raw)
        spans = self.tag(raw)
        tags = ["O"] * len(tokens)
        for sp in spans:
            for i in range(sp.start, sp.end):
                tags[i] = f"{'B' if i == sp.start else 'I'}-{sp.label}"
        return tokens, tags


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.title_engine.inference \"<title>\"")
        sys.exit(1)

    raw = " ".join(sys.argv[1:])
    ner = TitleNER()
    tokens, tags = ner.tag_bio(raw)

    print(f"\nInput : {raw}")
    print(f"Tokens: {tokens}\n")
    col = max(len(t) for t in tokens) + 2
    for tok, tag in zip(tokens, tags):
        print(f"  {tok:<{col}} {tag}")
