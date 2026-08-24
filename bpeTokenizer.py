import json
import re
from collections import Counter


# GPT-4's splitting pattern, translated to the stdlib `re` module:
#   \p{L} (any letter) -> [^\W\d]   a word char that isn't a digit: letters, plus underscore
#   \p{N} (any number) -> \d
# Python's \w and \d are unicode-aware by default, so 'e' and 'a' still count as letters.
# Order matters - at each position the first alternative that matches wins.
SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"   # contractions: 's 't 'd 'm 'll 've 're, either case
    r"| ?[^\s\w]+"             # runs of punctuation, with an optional leading space
    r"| ?[^\W\d]+"             # runs of letters,     with an optional leading space
    r"| ?\d{1,3}"              # numbers, at most 3 digits per token
    r"|\s*[\r\n]"              # a newline, keeping the whitespace in front of it
    r"|\s+(?!\S)"              # trailing whitespace at the end of a line
    r"|\s+"                    # any other whitespace run
)


def get_stats(ids, counts=None, weight=1):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):          # zip(ids, ids[1:]) = every consecutive pair
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def merge(ids, pair, idx):
    out = []
    i = 0
    while i < len(ids):
        # match the pair starting here? (the len-1 guard keeps ids[i+1] in bounds)
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2                          # skip both halves, no overlapping matches
        else:
            out.append(ids[i])
            i += 1
    return out


class BasicTokenizer:
    def __init__(self):
        self.pattern = None
        self.merges = {}                    # (int, int) -> int, in the order they were learned
        self.vocab = {i: bytes([i]) for i in range(256)}   # int -> the bytes it stands for

    def __repr__(self):
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _build_vocab(self):
        """merges -> vocab. Order matters: a merge only ever references older tokens."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def train(self, text, vocab_size, verbose=False):
        """Learn `vocab_size - 256` merges from `text`. Returns self, so you can chain."""
        assert vocab_size >= 256, "the 256 byte values are the floor of any byte-level vocab"
        ids = list(text.encode('utf-8'))    # 0..255, one int per byte
        self.merges = {}
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break                       # ran out of text before we ran out of merges
            pair = max(stats, key=stats.get)        # the most common pair right now
            idx = 256 + i                           # ...gets the next free token id
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            if verbose:
                print(f"merge {i+1}/{vocab_size-256}: {pair} -> {idx} "
                      f"({stats[pair]} occurrences)")
        self._build_vocab()
        return self

    def decode(self, ids):
        """ids -> str. Every token is a byte string, so glue them and decode once."""
        raw = b"".join(self.vocab[i] for i in ids)
        # errors='replace' because an arbitrary token sequence (say, one a half-trained
        # model just sampled) can easily be invalid UTF-8. Crashing there is no fun.
        return raw.decode('utf-8', errors='replace')

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break                       # nothing left that we know how to merge
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        return self._encode_chunk(text.encode('utf-8'))

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'type': type(self).__name__,
                'pattern': self.pattern,
                'merges': [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
            }, f)
        return path


    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        tok = cls(saved['pattern']) if saved.get('pattern') else cls()
        tok.merges = {(p0, p1): idx for p0, p1, idx in saved['merges']}
        tok._build_vocab()
        return tok


class RegexTokenizer(BasicTokenizer):
    def __init__(self, pattern=SPLIT_PATTERN):
        super().__init__()
        self.pattern = pattern
        self._compiled = re.compile(pattern)
        self._cache = {}                    # chunk str -> ids, since real text repeats a lot

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        counts = Counter(self._compiled.findall(text))
        # length-1 chunks have no pairs in them, so they can never affect a merge
        work = [(list(ch.encode('utf-8')), n) for ch, n in counts.items()]
        work = [(ids, n) for ids, n in work if len(ids) >= 2]

        self.merges = {}
        self._cache = {}
        for i in range(vocab_size - 256):
            stats = {}
            for ids, n in work:
                get_stats(ids, stats, n)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # `pair[0] in ids` is a C-level scan and skips the chunks that can't match,
            work = [(merge(ids, pair, idx) if pair[0] in ids else ids, n) for ids, n in work]
            work = [(ids, n) for ids, n in work if len(ids) >= 2]
            self.merges[pair] = idx
            if verbose:
                print(f"merge {i+1}/{vocab_size-256}: {pair} -> {idx} "
                      f"({stats[pair]} occurrences)")
        self._build_vocab()
        return self

    def encode(self, text):
        out = []
        for chunk in self._compiled.findall(text):
            ids = self._cache.get(chunk)
            if ids is None:
                ids = self._cache[chunk] = self._encode_chunk(chunk.encode('utf-8'))
            out.extend(ids)
        return out

    @classmethod
    def load(cls, path):
        tok = super().load(path)
        tok._compiled = re.compile(tok.pattern)
        tok._cache = {}
        return tok
