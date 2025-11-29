from tokenizer import Tokenizer
import time

def sample_text(textsource: str):
    with open(textsource) as f:
        text = f.read()
    text = text.split("<|endoftext|>")
    return text[:10]

tokenizer = Tokenizer.from_files("data/out/tinystories_vocab.json", "data/output/tinystories_merges.txt", ["<|endoftext|>"])

sample = sample_text("data/TinyStoriesV2-GPT4-train.txt")

start = time.time()
encoded = [tokenizer.encode(t) for t in sample]
end = time.time()
encoded_length = sum([len(t) for t in encoded])
bytes_length = sum([len(t.encode("utf-8")) for t in sample])

ratio = bytes_length/encoded_length
print(ratio)
throughput = bytes_length / (end - start)
print(throughput)

tokenizer = Tokenizer.from_files("data/out/owt_vocab.json", "data/output/owt_merges.txt", ["<|endoftext|>"])

sample = sample_text("data/owt_train.txt")

start = time.time()
encoded = [tokenizer.encode(t) for t in sample]
end = time.time()
encoded_length = sum([len(t) for t in encoded])
bytes_length = sum([len(t.encode("utf-8") for t in sample)])

ratio = bytes_length/encoded_length
print(ratio)
throughput = bytes_length / (end - start)
print(throughput)

