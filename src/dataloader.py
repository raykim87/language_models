from datasets import load_dataset
from collections import Counter
import re

def load_wikitext2(split="train"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    lines = dataset[split]["text"]
    return lines


def preprocess(lines):
    # remove empty lines and lines starting with "=" (headings)
    lines = [line.strip() for line in lines if line.strip()!='' and not line.strip().startswith("=")]
    # remove special characters and convert to lowercase
    lines = [re.sub(r"[^\w\s]", "", line.lower()).strip() for line in lines]
    return lines


def tokenize(lines):
    return [line.split() for line in lines]


def build_vocab(tokenized_sentences, min_count=5):
    vocab_counter = Counter(word for sentence in tokenized_sentences for word in sentence)
    vocab = {word: idx for idx, (word, count) in enumerate(vocab_counter.items()) if count >= min_count}
    return vocab

if __name__ == "__main__":
    lines = load_wikitext2("train")
    sentences = preprocess(lines)
    tokenized = tokenize(sentences)
    vocab = build_vocab(tokenized, min_count=5)
