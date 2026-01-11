import numpy as np
from cs336_basics.tokenizer import Tokenizer

DATA_DIR = "data/in"
OUTPUT_DIR = "data/out" 

tokenizer = Tokenizer.from_files(
    vocab_filepath=f"{OUTPUT_DIR}/tinystories_vocab.json",
    merges_filepath=f"{OUTPUT_DIR}/tinystories_merges.txt",
    special_tokens=["<|endoftext|>"]
)

def tokenize_file(input_path, output_path):
    print(f"Currently tokenizing: {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    ids = tokenizer.encode(text)
    arr = np.array(ids, dtype=np.uint16)

    np.save(output_path, arr)
    print(f"Tokenized data is saved to: {output_path} (shape: {arr.shape}, dtype: {arr.dtype})")

# 处理训练集和验证集
tokenize_file(
    f"{DATA_DIR}/TinyStoriesV2-GPT4-train.txt",
    f"{OUTPUT_DIR}/tinystories_train.npy"
)
tokenize_file(
    f"{DATA_DIR}/TinyStoriesV2-GPT4-valid.txt",
    f"{OUTPUT_DIR}/tinystories_val.npy"
)
print("所有文件处理完毕！")