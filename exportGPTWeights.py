"""Export a trained transformer (the model built in GPT.ipynb) to JSON for docs/.

The old exporter in MakeMore5Wavenet.ipynb only understood a flat
Linear/BatchNorm1D/Tanh stack. This one walks the Sequential that GPT.ipynb
builds - Embedding, PositionalEmbedding, N x Block, LayerNorm, Linear - and
writes out every tensor docs/index.html needs to run the forward pass in JS.

Usage from the notebook (no need to rebuild or retrain anything):

    from exportGPTWeights import export_gpt_weights
    export_gpt_weights(model, itos)                 # -> docs/gptWeights.json

Layers are matched by class *name*, not isinstance, so a reloaded
waveNetArchitecture module in a long-lived kernel can't break the export.
"""

import json
import os

import numpy as np


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
            "act": type(ff[1]).__name__.lower(),  # "relu" (was "tanh" before the switch)
            "proj": _linear(ff[2], decimals),
        },
    }


def export_gpt_weights(model, itos, path="docs/gptWeights.json", decimals=6):
    """Serialize `model` (a Sequential of transformer parts) to `path`."""
    out = {
        "format": "gpt-v1",
        "itos": {str(k): v for k, v in itos.items()},
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
            raise ValueError(f"export_gpt_weights doesn't know how to serialize a {kind}")

    for required in ("tokenEmbedding", "positionEmbedding", "lnFinal", "head"):
        if required not in out:
            raise ValueError(f"model is missing a {required} layer - is this the GPT model?")

    out["vocabSize"] = len(out["tokenEmbedding"])
    out["nEmbd"] = len(out["tokenEmbedding"][0])
    out["blockSize"] = len(out["positionEmbedding"])
    out["numHeads"] = len(out["blocks"][0]["heads"]) if out["blocks"] else 0
    out["nBlocks"] = len(out["blocks"])
    out["nParams"] = int(sum(p.data.size for p in model.parameters()))

    assert len(out["itos"]) == out["vocabSize"], "itos doesn't match the embedding table"

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(
        f"Wrote {path} ({os.path.getsize(path)/1024:.1f} KB) - "
        f"{out['nParams']:,} params, vocab {out['vocabSize']}, "
        f"n_embd {out['nEmbd']}, {out['nBlocks']} blocks x {out['numHeads']} heads, "
        f"block size {out['blockSize']}"
    )
    return path
