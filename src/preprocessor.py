import re
from collections import Counter

# Word2Vec Preprocessor
def generate_skipgram_pairs(tokenized_sentences, vocab, window_size=2):
    pairs = []
    for sentence in tokenized_sentences:
        valid_tokens = [w for w in sentence if w in vocab]
        for i, center in enumerate(valid_tokens):
            context_window = valid_tokens[max(i - window_size, 0): i] + valid_tokens[i + 1: i + 1 + window_size]
            for context in context_window:
                pairs.append((vocab[center], vocab[context]))
    return pairs

def generate_cbow_pairs(tokenized_sentences, vocab, window_size=2):
    pairs = []
    for sentence in tokenized_sentences:
        valid_tokens = [w for w in sentence if w in vocab]
        for i, target in enumerate(valid_tokens):
            context_window = valid_tokens[max(i - window_size, 0): i] + valid_tokens[i + 1: i + 1 + window_size]
            if context_window:
                pairs.append((vocab[target], [vocab[context] for context in context_window]))
    return pairs

def truncate_sentences(tokenized_sentences, max_length=50):
    return [sentence[:max_length] for sentence in tokenized_sentences]

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
    filtered_vocab = [word for word, count in vocab_counter.items() if count >= min_count]
    vocab = {word: idx for idx, word in enumerate(filtered_vocab, start=2)}
    vocab["<pad>"] = 0 # Add padding token
    vocab["<unk>"] = 1
    return vocab
