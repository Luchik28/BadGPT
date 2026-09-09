// Forward pass + byte-pair tokenizer for the model trained in
// GPTwithTokenizerNoPlot.ipynb (FineWeb-Edu, vocab 1025).
//
// This is a deliberate copy of transformer.js rather than an extension of it. transformer.js
// drives the character-level Shakespeare panel and is left exactly as it was;
// duplicating ~150 lines of matmul is cheaper than risking that model to share
// them. The transformer half below is identical to transformer.js line for line -
// nothing in it ever depended on the vocabulary size.
//
// Everything that *is* new is tokenizer work, and it all comes from one place:
// a character model could turn an id into text with a lookup, because every id
// was one character. A byte-pair id is a byte string, and it's routinely half a
// UTF-8 character, so text has to be assembled as bytes and decoded once at the
// end. The same asymmetry runs the other way for prompts, which is why the BPE
// merge loop is reimplemented here.
var BadGPTBPE = (function () {

  // ---------------------------------------------------------------- weights

  // nested [[...]] -> { data: Float32Array, rows, cols }
  function matrix(rows2d) {
    const rows = rows2d.length;
    const cols = rows2d[0].length;
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const row = rows2d[i];
      for (let j = 0; j < cols; j++) data[i * cols + j] = row[j];
    }
    return { data: data, rows: rows, cols: cols };
  }

  function linearWeights(layer) {
    return {
      w: matrix(layer.weight),
      b: layer.bias ? Float32Array.from(layer.bias) : null,
    };
  }

  function normWeights(ln) {
    return { gamma: Float32Array.from(ln.gamma), beta: Float32Array.from(ln.beta), eps: ln.eps };
  }

  // Turn the exported JSON into typed arrays. Do this once, at load.
  function prepare(json) {
    const net = {
      blockSize: json.blockSize,
      vocabSize: json.vocabSize,
      nEmbd: json.nEmbd,
      nParams: json.nParams,
      // -1 when the export carried no end-of-text token, so nothing ever matches
      eosId: json.eosId == null ? -1 : json.eosId,
      itos: [],
      itosBytes: [],
      merges: new Map(),
      tok: matrix(json.tokenEmbedding),
      pos: matrix(json.positionEmbedding),
      lnFinal: normWeights(json.lnFinal),
      head: linearWeights(json.head),
      blocks: json.blocks.map(function (b) {
        return {
          ln1: normWeights(b.ln1),
          heads: b.heads.map(function (h) {
            return { key: matrix(h.key.weight), query: matrix(h.query.weight), value: matrix(h.value.weight) };
          }),
          proj: linearWeights(b.proj),
          ln2: normWeights(b.ln2),
          ff: { fc: linearWeights(b.ff.fc), act: b.ff.act, proj: linearWeights(b.ff.proj) },
        };
      }),
    };

    for (let i = 0; i < json.vocabSize; i++) {
      net.itos.push(json.itos[String(i)]);
      net.itosBytes.push(Uint8Array.from(json.itosBytes[String(i)] || []));
    }

    // (p0, p1) -> merged id. Packed into one number because a Map keyed on
    // strings costs a concat per pair per merge step, and encoding a prompt
    // walks every pair many times over.
    const merges = (json.tokenizer && json.tokenizer.merges) || [];
    for (let i = 0; i < merges.length; i++) {
      net.merges.set(merges[i][0] * 65536 + merges[i][1], merges[i][2]);
    }

    return net;
  }

  // -------------------------------------------------------------- tokenizer

  // bpeTokenizer.py's SPLIT_PATTERN, translated to JavaScript. Two differences
  // are forced by the regex engine, not chosen:
  //   - Python's (?i:...) scoped-flag group has no JS equivalent, so the
  //     contraction branch spells both cases out.
  //   - Python's \w and \d are unicode-aware; JS's are ASCII-only. Using them
  //     as-is would send 'e' down the punctuation branch instead of the letter
  //     branch and silently retokenize any accented text, so the classes are
  //     written with \p{...} property escapes (hence the `u` flag) - which is
  //     what the Python pattern was approximating in the first place.
  const SPLIT_PATTERN = new RegExp(
    "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])" + // contractions: 's 't 'd 'm 'll 've 're
    "| ?[^\\s\\p{L}\\p{N}_]+" +                    // runs of punctuation, optional leading space
    "| ?[\\p{L}_]+" +                              // runs of letters,     optional leading space
    "| ?\\p{Nd}{1,3}" +                            // numbers, at most 3 digits per token
    "|\\s*[\\r\\n]" +                              // a newline, keeping the whitespace in front
    "|\\s+(?!\\S)" +                               // trailing whitespace at the end of a line
    "|\\s+",                                       // any other whitespace run
    "gu"
  );

  // The training-time merge loop: repeatedly apply the lowest-numbered merge
  // that's present. Merge ids are assigned in the order they were learned, so
  // "lowest id" is "earliest merge", and applying them out of order would
  // produce a different tokenization than the model ever saw.
  function encodeChunk(bytes, merges) {
    let ids = Array.from(bytes);
    while (ids.length >= 2) {
      let bestPair = -1;
      let bestId = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const key = ids[i] * 65536 + ids[i + 1];
        const id = merges.get(key);
        if (id !== undefined && (bestId < 0 || id < bestId)) {
          bestId = id;
          bestPair = key;
        }
      }
      if (bestPair < 0) break; // nothing left that we know how to merge

      const p0 = Math.floor(bestPair / 65536);
      const p1 = bestPair % 65536;
      const out = [];
      let i = 0;
      while (i < ids.length) {
        // match the pair starting here? (the len-1 guard keeps ids[i+1] in bounds)
        if (i < ids.length - 1 && ids[i] === p0 && ids[i + 1] === p1) {
          out.push(bestId);
          i += 2; // skip both halves, no overlapping matches
        } else {
          out.push(ids[i]);
          i += 1;
        }
      }
      ids = out;
    }
    return ids;
  }

  const encoder = new TextEncoder();

  // text -> token ids. Splitting first is what stops a merge from ever spanning
  // a word boundary, so " the" can become one token but "the cat" cannot.
  function encode(net, text) {
    const chunks = String(text).match(SPLIT_PATTERN) || [];
    const out = [];
    for (let i = 0; i < chunks.length; i++) {
      const ids = encodeChunk(encoder.encode(chunks[i]), net.merges);
      for (let j = 0; j < ids.length; j++) out.push(ids[j]);
    }
    return out;
  }

  // token ids -> text. Bytes are concatenated *first* and decoded once, exactly
  // like tok.decode does; decoding token by token would turn every id that
  // holds half a character into U+FFFD.
  function decode(net, ids) {
    let total = 0;
    for (let i = 0; i < ids.length; i++) total += net.itosBytes[ids[i]].length;
    const bytes = new Uint8Array(total);
    let o = 0;
    for (let i = 0; i < ids.length; i++) {
      bytes.set(net.itosBytes[ids[i]], o);
      o += net.itosBytes[ids[i]].length;
    }
    return new TextDecoder().decode(bytes);
  }

  // ------------------------------------------------------------ forward pass

  // out[t] = x[t] @ w (+ b);  x is (T, w.rows), out is (T, w.cols)
  function linear(x, T, w, b) {
    const K = w.rows;
    const N = w.cols;
    const wd = w.data;
    const out = new Float32Array(T * N);
    for (let t = 0; t < T; t++) {
      const xo = t * K;
      const oo = t * N;
      if (b) out.set(b, oo);
      for (let k = 0; k < K; k++) {
        const v = x[xo + k];
        if (v === 0) continue;
        const wo = k * N;
        for (let n = 0; n < N; n++) out[oo + n] += v * wd[wo + n];
      }
    }
    return out;
  }

  // normalize each row to zero mean / unit variance, then scale and shift
  function layerNorm(x, T, C, ln) {
    const out = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
      const o = t * C;
      let mean = 0;
      for (let c = 0; c < C; c++) mean += x[o + c];
      mean /= C;
      let varr = 0;
      for (let c = 0; c < C; c++) {
        const d = x[o + c] - mean;
        varr += d * d;
      }
      varr /= C;
      const stdInv = 1 / Math.sqrt(varr + ln.eps);
      for (let c = 0; c < C; c++) out[o + c] = (x[o + c] - mean) * stdInv * ln.gamma[c] + ln.beta[c];
    }
    return out;
  }

  // one causal head: softmax(q @ k^T / sqrt(head_size)) @ v, masked so position
  // i only ever attends to positions <= i
  function attentionHead(x, T, h) {
    const hs = h.key.cols;
    const k = linear(x, T, h.key, null);
    const q = linear(x, T, h.query, null);
    const v = linear(x, T, h.value, null);
    const scale = 1 / Math.sqrt(hs);
    const out = new Float32Array(T * hs);
    const wei = new Float32Array(T);

    for (let i = 0; i < T; i++) {
      const qo = i * hs;
      let max = -Infinity;
      for (let j = 0; j <= i; j++) {
        const ko = j * hs;
        let s = 0;
        for (let d = 0; d < hs; d++) s += q[qo + d] * k[ko + d];
        s *= scale;
        wei[j] = s;
        if (s > max) max = s;
      }
      let sum = 0;
      for (let j = 0; j <= i; j++) {
        const e = Math.exp(wei[j] - max); // subtract the max, same as the numpy softmax
        wei[j] = e;
        sum += e;
      }
      const oo = i * hs;
      for (let j = 0; j <= i; j++) {
        const a = wei[j] / sum;
        const vo = j * hs;
        for (let d = 0; d < hs; d++) out[oo + d] += a * v[vo + d];
      }
    }
    return out;
  }

  // Full forward pass over `context` (an array of token ids, at most blockSize
  // long). Returns the logits for the *last* position only - that's the only
  // row generation ever looks at, and the head is the widest matmul here.
  function forwardLast(net, context) {
    const T = context.length;
    const C = net.nEmbd;
    let x = new Float32Array(T * C);

    for (let t = 0; t < T; t++) {
      const to = context[t] * C;
      const o = t * C;
      for (let c = 0; c < C; c++) x[o + c] = net.tok.data[to + c] + net.pos.data[o + c];
    }

    for (let bi = 0; bi < net.blocks.length; bi++) {
      const blk = net.blocks[bi];

      // communicate: attention, wrapped in a residual
      const h1 = layerNorm(x, T, C, blk.ln1);
      const cat = new Float32Array(T * C);
      let off = 0;
      for (let hi = 0; hi < blk.heads.length; hi++) {
        const hs = blk.heads[hi].key.cols;
        const ho = attentionHead(h1, T, blk.heads[hi]);
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < hs; d++) cat[t * C + off + d] = ho[t * hs + d];
        }
        off += hs;
      }
      const attn = linear(cat, T, blk.proj.w, blk.proj.b);
      for (let i = 0; i < x.length; i++) x[i] += attn[i];

      // compute: per-token MLP, also residual
      const h2 = layerNorm(x, T, C, blk.ln2);
      const hidden = linear(h2, T, blk.ff.fc.w, blk.ff.fc.b);
      if (blk.ff.act === "tanh") {
        for (let i = 0; i < hidden.length; i++) hidden[i] = Math.tanh(hidden[i]);
      } else {
        for (let i = 0; i < hidden.length; i++) if (hidden[i] < 0) hidden[i] = 0; // relu
      }
      const ff = linear(hidden, T, blk.ff.proj.w, blk.ff.proj.b);
      for (let i = 0; i < x.length; i++) x[i] += ff[i];
    }

    const normed = layerNorm(x, T, C, net.lnFinal);
    const last = normed.subarray((T - 1) * C, T * C);
    return linear(last, 1, net.head.w, net.head.b);
  }

  function sampleFromLogits(logits, temperature, rand) {
    const t = temperature || 1;
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    let sum = 0;
    const probs = new Float64Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp((logits[i] - max) / t);
      sum += probs[i];
    }
    let r = (rand || Math.random)() * sum;
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r < 0) return i;
    }
    return probs.length - 1;
  }

  // ------------------------------------------------------------- generation

  // Generate up to `length` tokens, sliding the context window as it fills.
  // `onChunk` is awaited every `chunkSize` tokens so the browser can paint
  // between batches instead of locking up for the whole sample.
  //
  // Returns { text, tokens, stoppedOnEos } rather than a bare string, because
  // "the model decided it was finished" is worth telling the user about.
  async function generate(net, options) {
    const opts = options || {};
    const length = opts.length || 300;
    const temperature = opts.temperature || 1;
    const chunkSize = opts.chunkSize || 8;
    const onChunk = opts.onChunk;
    const rand = opts.rand;

    // Match the notebook's sampler exactly: left-pad the prompt with token 0 up
    // to blockSize, or keep only the most recent blockSize tokens if it's
    // longer. Every window the model trained on was a full blockSize wide, so
    // handing it a short one is off-distribution in a way padding avoids.
    const prime = opts.prime && opts.prime.length ? opts.prime.slice() : [];
    let context;
    if (prime.length >= net.blockSize) {
      context = prime.slice(-net.blockSize);
    } else {
      context = new Array(net.blockSize - prime.length).fill(0).concat(prime);
    }

    const decoder = new TextDecoder();
    const tokens = [];
    let text = "";
    let pending = [];   // bytes not yet handed to onChunk
    let sinceFlush = 0; // tokens since the last flush
    let stoppedOnEos = false;

    for (let i = 0; i < length; i++) {
      const ix = sampleFromLogits(forwardLast(net, context), temperature, rand);
      if (ix === net.eosId) {
        stoppedOnEos = true; // the model signaled the end of its own generation
        break;
      }

      const bytes = net.itosBytes[ix];
      for (let b = 0; b < bytes.length; b++) pending.push(bytes[b]);
      tokens.push(ix);

      context.push(ix);
      if (context.length > net.blockSize) context.shift(); // only the last blockSize fit

      if (onChunk && ++sinceFlush >= chunkSize) {
        // stream:true holds back a trailing partial character instead of
        // emitting U+FFFD for it, so a multi-byte character split across two
        // flushes still arrives as one character
        const piece = decoder.decode(Uint8Array.from(pending), { stream: true });
        pending = [];
        sinceFlush = 0;
        if (piece) {
          text += piece;
          await onChunk(piece, i + 1, length);
        }
      }
    }

    const tail = decoder.decode(Uint8Array.from(pending)); // final call flushes
    if (tail) {
      text += tail;
      if (onChunk) await onChunk(tail, length, length);
    }

    return { text: text, tokens: tokens, stoppedOnEos: stoppedOnEos };
  }

  return {
    prepare: prepare,
    encode: encode,
    decode: decode,
    forwardLast: forwardLast,
    sampleFromLogits: sampleFromLogits,
    generate: generate,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = BadGPTBPE; // for a node test
