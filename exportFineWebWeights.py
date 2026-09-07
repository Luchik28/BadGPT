"""Export the byte-pair model trained in GPTwithTokenizerNoPlot.ipynb to JSON.

This is a sibling of exportGPTWeights.py, not a replacement for it. That one
still owns the character-level Shakespeare model and docs/gptWeights.json, and
nothing here writes to either. The transformer half of the format is identical
- same layer walk, same tensor names - so docs/fineweb.js can reuse the same
forward pass that docs/gpt.js runs.

What's different is everything to do with the vocabulary. The Shakespeare model
had 65 tokens that each happened to be one character, so the browser could turn
an id into text with a lookup and be done. A byte-pair token is a *byte string*,
and it's routinely half of a UTF-8 character, so this file writes:

    itosBytes    id -> the raw bytes it stands for, so the page can concatenate
                 bytes and decode UTF-8 once at the end (what tok.decode does)
                 instead of decoding each token on its own and getting U+FFFD
    eosId        the <|endoftext|> id, so generation can stop the way the
                 notebook's sampler does instead of printing the token
    tokenizer    the learned merges, so the page can *encode* a typed prompt
                 with the exact merge order the model was trained on

Usage from the notebook (pass the tokenizer itself - everything vocabulary-side
is derived from it, so the weights and the vocab can't drift apart):

    from exportFineWebWeights import export_fineweb_weights
    export_fineweb_weights(model, tok)          # -> docs/finewebWeights.json
"""

import json
import os

import numpy as np


def _refuse_vocab_mismatch(path, vocab_size, allow):
    """Bail out if `path` already holds weights for a differently-sized vocabulary.

    The mirror of the guard in exportGPTWeights.py, kept here as its own copy so the
    two exporters stay independent. Re-exporting this model after more training keeps
    vocab 1025 and sails through; aiming it at docs/gptWeights.json does not.
    """
    if allow or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (ValueError, OSError):
        return  # unreadable or not ours; nothing to protect
    found = existing.get("vocabSize")
    if found is None or found == vocab_size:
        return
    raise ValueError(
        f"refusing to overwrite {path}: it holds a vocab-{found} model, but this export "
        f"has vocab {vocab_size}. docs/gptWeights.json belongs to the character-level "
        f"shakespear model and is written by exportGPTWeights.py. If you really do mean "
        f"to replace it, pass overwrite_mismatch=True."
    )


def _tensor(t, decimals):
    """Value | ndarray -> nested lists, rounded (rounding ~halves the file size)."""
    data = t if isinstance(t, np.ndarray) else t.data
    return np.round(np.asarray(data, dtype=np.float64), decimals).tolist()


def _linear(layer, decimals):
    return {
        "weight": _tensor(layer.weight, decimals),  # (fan_in, fan_out)
        "bias": None if layer.bias is None else _tensor(layer.bias, decimals),
    }


def _layernorm(layer, decimals):
    return {
        "gamma": np.asarray(_tensor(layer.gamma, decimals)).reshape(-1).tolist(),
        "beta": np.asarray(_tensor(layer.beta, decimals)).reshape(-1).tolist(),
        "eps": float(layer.eps),
    }


def _block(block, decimals):
    ff = block.ff.net.layers  # [Linear, ReLU, Linear]
    return {
        "ln1": _layernorm(block.ln1, decimals),
        # each head keeps its own k/q/v; the browser concatenates them exactly
        # like Value.cat does, then runs proj over the result
        "heads": [
            {
                "key": _linear(h.key, decimals),
                "query": _linear(h.query, decimals),
                "value": _linear(h.value, decimals),
            }
            for h in block.attn.heads
        ],
        "proj": _linear(block.attn.proj, decimals),
        "ln2": _layernorm(block.ln2, decimals),
        "ff": {
            "fc": _linear(ff[0], decimals),
            "act": type(ff[1]).__name__.lower(),  # "relu"
            "proj": _linear(ff[2], decimals),
        },
    }


def export_fineweb_weights(model, tok, path="docs/finewebWeights.json", decimals=6,
                           eos_token="<|endoftext|>", overwrite_mismatch=False):
    """Serialize `model` (a Sequential of transformer parts) plus `tok`'s vocabulary.

    `tok` is a bpeTokenizer.RegexTokenizer - the same object the notebook trained
    against. Its vocab, merges and special tokens all go into the file, so the
    page never needs a second fetch to tokenize a prompt.
    """
    out = {
        "format": "gpt-bpe-v1",
        # kept as text purely so the file is readable/greppable; the page decodes
        # from itosBytes, because this field is lossy for partial-character tokens
        "itos": {str(i): b.decode("utf-8", errors="replace") for i, b in tok.vocab.items()},
        "itosBytes": {str(i): list(b) for i, b in tok.vocab.items()},
        "blocks": [],
    }

    for layer in model.layers:
        kind = type(layer).__name__
        if kind == "Embedding":
            out["tokenEmbedding"] = _tensor(layer.weight, decimals)      # (vocab, n_embd)
        elif kind == "PositionalEmbedding":
            out["positionEmbedding"] = _tensor(layer.weight, decimals)   # (block_size, n_embd)
        elif kind == "Block":
            out["blocks"].append(_block(layer, decimals))
        elif kind == "LayerNorm":
            out["lnFinal"] = _layernorm(layer, decimals)                 # the one before the head
        elif kind == "Linear":
            out["head"] = _linear(layer, decimals)                       # (n_embd, vocab)
        else:
            raise ValueError(f"export_fineweb_weights doesn't know how to serialize a {kind}")

    for required in ("tokenEmbedding", "positionEmbedding", "lnFinal", "head"):
        if required not in out:
            raise ValueError(f"model is missing a {required} layer - is this the GPT model?")

    out["vocabSize"] = len(out["tokenEmbedding"])
    out["nEmbd"] = len(out["tokenEmbedding"][0])
    out["blockSize"] = len(out["positionEmbedding"])
    out["numHeads"] = len(out["blocks"][0]["heads"]) if out["blocks"] else 0
    out["nBlocks"] = len(out["blocks"])
    out["nParams"] = int(sum(p.data.size for p in model.parameters()))

    # -1 rather than null: the page compares every sampled id against this, and a
    # model exported without an EOS token should simply never match
    out["eosId"] = int(tok.special_tokens.get(eos_token, -1))
    out["tokenizer"] = {
        # the Python split pattern, for reference only - docs/fineweb.js carries
        # its own translation of it, since (?i:...) isn't valid JavaScript regex
        "pattern": tok.pattern,
        "merges": [[int(p0), int(p1), int(idx)] for (p0, p1), idx in tok.merges.items()],
        "specialTokens": {name: int(i) for name, i in tok.special_tokens.items()},
    }

    assert len(out["itos"]) == out["vocabSize"], "itos doesn't match the embedding table"
    assert len(out["itosBytes"]) == out["vocabSize"], "itosBytes doesn't match the embedding table"

    _refuse_vocab_mismatch(path, out["vocabSize"], overwrite_mismatch)

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(
        f"Wrote {path} ({os.path.getsize(path)/1024:.1f} KB) - "
        f"{out['nParams']:,} params, vocab {out['vocabSize']}, "
        f"n_embd {out['nEmbd']}, {out['nBlocks']} blocks x {out['numHeads']} heads, "
        f"block size {out['blockSize']}, {len(out['tokenizer']['merges'])} merges, "
        f"eos {out['eosId']}"
    )
    return path
