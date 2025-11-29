
from train_bpe import train_bpe
from utils.io import save_voacb_and_merge

vocab,merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])

save_voacb_and_merge(vocab, merges,"./data/out/tinystories_vocab.json","./data/output/tinystories_merges.txt")

vocab,merges = train_bpe("./data/owt_train.txt", 32000, ["<|endoftext|>"])

save_voacb_and_merge(vocab, merges,"./data/out/owt_vocab.json","./data/output/owt_merges.txt")
