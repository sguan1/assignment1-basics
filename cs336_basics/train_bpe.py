from collections import Counter, defaultdict
import os
from typing import BinaryIO
import regex as re
import os
from multiprocessing import Pool
import multiprocessing
import heapq

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class ReverseLexOrderPair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair > other.pair
    
    def __eq__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair == other.pair

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def build_new_key(old_key: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    i = 0
    new_key = []
    while i < len(old_key):
        if i + 1 < len(old_key) and old_key[i] == pair[0] and old_key[i + 1] == pair[1]:
            new_key.append(pair[0] + pair[1])
            i += 2
        else:
            new_key.append(old_key[i])
            i += 1

    return tuple(new_key)

def merge(result: dict[tuple[bytes], int], pair: tuple[bytes, bytes], new_vocab: bytes, counts: dict[tuple[bytes, bytes]], pair_to_keys:dict[tuple[bytes,bytes], set[tuple[bytes]]]) -> set[tuple[bytes]]:
    changed_paris = set()
    keys_to_modify = pair_to_keys[pair].copy()
    for old_key in keys_to_modify:
        old_freq = result.pop(old_key)
        new_key = build_new_key(old_key, pair)
        
        for i in range(len(old_key) - 1):
            left = old_key[i]
            right = old_key[i + 1]
            counts[(left,right)] -= old_freq
            changed_paris.add((left, right))
            if counts[(left, right)] <= 0:
                del counts[(left, right)]
            pair_to_keys[(left, right)].discard(old_key)

        for i in range(len(new_key) - 1):
            left = new_key[i]
            right = new_key[i + 1]
            counts[(left, right)] += old_freq
            changed_paris.add((left,right))
            pair_to_keys[(left, right)].add(new_key)

        result[new_key] = result.get(new_key, 0) + old_freq
    pair_to_keys[pair] = set()

    return changed_paris

def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    pattern = re.compile("|".join(re.escape(token) for token in special_tokens))
    subtexts = pattern.split(text)
    result: dict[tuple[int], int] = {}
    for subtext in subtexts:
        for match in PAT.finditer(subtext):
            key = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            result[key] = result.get(key, 0) + 1
    return result

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    chunks = []
    num_processes = multiprocessing.cpu_count()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    final_result = Counter()
    results = []
    args = [(chunk, special_tokens) for chunk in chunks]
    with Pool(num_processes) as pool:
        results = pool.starmap(pretokenize, args)

    for result in results:
        final_result.update(result)

    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    index = len(vocab)
    for special_token in special_tokens:
        vocab[index] = special_token.encode("utf-8")
        index += 1
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    counts = defaultdict(int)
    pair_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]] = defaultdict(set)
    for key, value in final_result.items():
        for i in range(len(key) - 1):
            pair = (key[i], key[i + 1])
            counts[pair] += value
            pair_to_keys[pair].add(key)

    heap = []
    for pair, freq in counts.items():
        if freq > 0:
            heapq.heappush(heap, (-freq, ReverseLexOrderPair(pair),pair))
    
    for i in range(len(vocab), vocab_size):
        if not heap:
            break

        while heap:
            negative_freq, _, top_pair = heapq.heappop(heap)
            freq = -negative_freq
            if counts[top_pair] == freq:
                pair = top_pair
                break
            if counts[top_pair] > 0:
                heapq.heappush(heap, (-counts[top_pair], ReverseLexOrderPair(top_pair),top_pair))
        else:
            break

        if counts[pair] <= 0:
            break
        
        new_vocab = pair[0] + pair[1]
        merges.append(pair)
        vocab[i] = new_vocab
        changed_pairs = merge(final_result, pair, new_vocab, counts, pair_to_keys)

        for changed_pair in changed_pairs:
            if changed_pair in counts and counts[changed_pair] > 0:
                heapq.heappush(heap, (-counts[changed_pair], ReverseLexOrderPair(changed_pair), changed_pair))

    return (vocab, merges)