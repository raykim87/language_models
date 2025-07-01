from datasets import load_dataset

def load_wikitext2(split="train"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    lines = dataset[split]["text"]
    return lines

