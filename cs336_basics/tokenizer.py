from collections.abc import Iterable, Iterator
import regex as re
import os
import json
from cs336_basics.utils.io import get_tokenizer_from_vocab_merges_path



PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens:list[str] | None = None):
        """Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None"""
        self.vocab = vocab
        self.vocab_inv ={value: key for key,value in vocab.items()}
        self.merges = merges
        self.merges_order = {(self.vocab_inv[merge[0]], self.vocab_inv[merge[1]]): self.vocab_inv[merge[0] + merge[1]] for merge in merges}
        self.special_tokens = None
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key = len, reverse=True)
            next_id = max(self.vocab.keys()) + 1
            for special_token in self.special_tokens:
                encoded_token = special_token.encode("UTF_8")
                if encoded_token not in self.vocab_inv:
                    self.vocab_inv[encoded_token] = next_id
                    self.vocab[next_id] = encoded_token
                    next_id += 1

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath:str, special_tokens:list[str] | None = None):
        """Class
        method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None"""
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_filepath, merges_filepath, special_tokens)
        return cls(vocab, merges, special_tokens)
        

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.
        def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory."""

        if not self.special_tokens:
            return self.merge(text)
        
        result = []
        # Split on special toekns and keep delimiters
        pattern = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
        subtexts = re.split(pattern, text)
        for subtext in subtexts:
            if subtext in self.special_tokens:
                result.append(self.vocab_inv[subtext.encode("UTF-8")])
            else:
                result.extend(self.merge(subtext))
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory."""
        for text in iterable:
            yield from self.encode(text)
                
    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        text = b''.join(self.vocab[id] for id in ids)
        return text.decode("utf-8", errors="replace")

    def merge(self, text:str) -> list[int]:
        result=[]
        pairs=[]
        for match in PAT.finditer(text):
            group = match.group()
            ids = [self.vocab_inv[bytes([b])] for b in group.encode("utf-8")]
            while len(ids) >= 2:
                pairs = self.getPairs(ids)
                pair = min(pairs, key = lambda e: self.merges_order.get(e, float('inf')))
                if pair not in self.merges_order:
                    break
                id = self.merges_order[pair]
                ids = self.update(ids, id, pair)
            result.extend(ids)
        return result

    def getPairs(self, ids: list[int]) -> list[int]:
        pairs = []
        for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pairs.append(pair)
        return pairs

    def update(self, ids: list[int], id: int, pair: tuple[int, int])-> list[int]:
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(self.vocab_inv[self.vocab[pair[0]] + self.vocab[pair[1]]])
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result    
    


                



