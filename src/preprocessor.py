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



