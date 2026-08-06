// Forward pass for the character-level transformer trained in GPT.ipynb.
// Mirrors waveNetArchitecture.py exactly: Embedding + PositionalEmbedding,
// then N x (LayerNorm -> multi-head causal attention -> residual,
// LayerNorm -> feedforward -> residual), a final LayerNorm and a linear head.
//
// Everything is flat Float32Arrays (row-major) rather than arrays-of-arrays:
// a full forward pass over 64 positions is ~4M multiply-adds, and nested JS
// arrays make that take long enough to freeze the page between characters.
var BadGPT = (function () {

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
      itos: [],
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
    for (let i = 0; i < json.vocabSize; i++) net.itos.push(json.itos[String(i)]);
    return net;
  }

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

  // Generate `length` characters, sliding the context window once it fills up.
  // `onChunk` is awaited every `chunkSize` characters so the browser can paint
  // between batches instead of locking up for the whole sample.
  async function generate(net, options) {
    const opts = options || {};
    const length = opts.length || 400;
    const temperature = opts.temperature || 1;
    const chunkSize = opts.chunkSize || 8;
    const onChunk = opts.onChunk;
    const rand = opts.rand;

    let context = opts.prime && opts.prime.length ? opts.prime.slice() : [0]; // token 0 is '\n'
    if (context.length > net.blockSize) context = context.slice(-net.blockSize);

    let text = "";
    let pending = "";
    for (let i = 0; i < length; i++) {
      const ix = sampleFromLogits(forwardLast(net, context), temperature, rand);
      const ch = net.itos[ix];
      text += ch;
      pending += ch;
      context.push(ix);
      if (context.length > net.blockSize) context.shift(); // only the last blockSize tokens fit
      if (onChunk && pending.length >= chunkSize) {
        await onChunk(pending, i + 1, length);
        pending = "";
      }
    }
    if (onChunk && pending) await onChunk(pending, length, length);
    return text;
  }

  return {
    prepare: prepare,
    forwardLast: forwardLast,
    sampleFromLogits: sampleFromLogits,
    generate: generate,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = BadGPT; // for the node test
