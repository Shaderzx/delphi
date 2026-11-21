import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
FLT_MAX = F32(np.finfo(np.float32).max)
FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=F32)
FP4_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=F32)


def e4m3_table():
    values = np.zeros(256, dtype=F32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        exponent, mantissa = (b >> 3) & 15, b & 7
        if exponent == 15 and mantissa == 7:
            values[b] = np.nan
        elif exponent == 0:
            values[b] = sign * mantissa / 8 * 2.0**-6
        else:
            values[b] = sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    return values


E4M3 = e4m3_table()
E4M3_POSITIVE = E4M3[:127].copy()
FP4_PAIRS = np.stack([FP4[np.arange(256) & 7] * np.where(np.arange(256) & 8, -1, 1),
                      FP4[(np.arange(256) >> 4) & 7] * np.where(np.arange(256) & 0x80, -1, 1)], axis=1).astype(np.float16)


def bf16(x):
    u = np.ascontiguousarray(x, dtype=F32).view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(F32)


def e4m3_rne(x):
    x = np.asarray(x, dtype=F32)
    hi = np.clip(np.searchsorted(E4M3_POSITIVE, x, side="left"), 1, 126)
    lo = hi - 1
    dlo, dhi = x - E4M3_POSITIVE[lo], E4M3_POSITIVE[hi] - x
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0))
    out = np.where(pick_hi, E4M3_POSITIVE[hi], E4M3_POSITIVE[lo]).astype(F32)
    return np.where(x <= 0, F32(0), np.where(x >= 448, F32(448), out)).astype(F32)


def e4m3_rne_signed(x):
    return (np.sign(x) * e4m3_rne(np.abs(x))).astype(F32)


def sigmoid(x):
    return (F32(1) / (F32(1) + np.exp(-x.astype(F32)))).astype(F32)


def rsqrt_mean_square(x, eps=1e-6):
    return F32(1) / np.sqrt(F32(np.sum(x * x, dtype=F32)) / F32(x.shape[-1]) + F32(eps))


class Checkpoint:
    def __init__(self, path):
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(size))
        self.header.pop("__metadata__", None)
        self.data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size)

    def raw(self, name):
        info = self.header[name]
        start, end = info["data_offsets"]
        kind = {"BF16": np.uint16, "F32": F32, "U8": np.uint8, "F8_E4M3": np.uint8}[info["dtype"]]
        return self.data[start:end].view(kind).reshape(info["shape"])

    def f32(self, name):
        info = self.header[name]
        raw = self.raw(name)
        if info["dtype"] == "BF16":
            return (raw.astype(np.uint32) << 16).view(F32)
        if info["dtype"] == "F8_E4M3":
            return E4M3[raw]
        return np.asarray(raw, dtype=F32)

    def f32_row(self, name, row):
        info = self.header[name]
        raw = self.raw(name)[row]
        if info["dtype"] == "BF16":
            return (raw.astype(np.uint32) << 16).view(F32)
        if info["dtype"] == "F8_E4M3":
            return E4M3[raw]
        return np.asarray(raw, dtype=F32)

    def has(self, name):
        return name in self.header


def attn_res(sources, addend, queries, normalize_values):
    xs = [s.astype(F32) for s in sources]
    if addend is not None:
        xs[-1] = (xs[-1] + addend.astype(F32)).astype(F32)
    rs = np.array([rsqrt_mean_square(x) for x in xs], dtype=F32)
    stacked = np.stack(xs)
    logits = (queries.astype(F32) @ stacked.T) * rs[None, :]
    weights = np.exp(logits - logits.max(axis=1, keepdims=True)).astype(F32)
    weights = weights / weights.sum(axis=1, keepdims=True, dtype=F32)
    if normalize_values:
        weights = weights * rs[None, :]
    return (weights @ stacked).astype(F32)


def rms_norm(x, weight, eps=1e-6):
    x = x.astype(F32)
    return bf16(x * rsqrt_mean_square(x, eps) * weight)


def static_conv(x, state, weight):
    taps = weight.shape[1]
    history = np.concatenate([state, x[None, :]], axis=0)
    acc = np.zeros_like(x, dtype=F32)
    for tap in range(taps):
        acc = acc + history[tap] * weight[:, tap]
    y = bf16((acc + x) * F32((taps + 1) ** -0.5))
    return y, np.concatenate([state[1:], x[None, :]], axis=0)


def q_nvfp4(row):
    row = row.astype(F32)
    amax = F32(np.abs(row).max())
    if amax == 0:
        return np.zeros_like(row), F32(0)
    enc = F32(min(F32(1536) / amax, FLT_MAX))
    dec = F32(1) / enc
    outer = F32(amax * F32(1 / 1536))
    blocks = row.reshape(-1, 16)
    bmax = np.abs(blocks).max(axis=1)
    candidates = []
    for im in (F32(1 / 6), F32(0.25)):
        scale = e4m3_rne(np.minimum(bmax * F32(enc * im), F32(448)))
        with np.errstate(divide="ignore", over="ignore"):
            inv = np.minimum(F32(1) / (scale * dec), FLT_MAX).astype(F32)
        v = blocks * inv[:, None]
        a = np.abs(v)
        code = np.searchsorted(FP4_BOUNDS, a, side="left")
        code = code + ((a == FP4_BOUNDS[np.minimum(code, 6)]) & (code % 2 == 1))
        value = (FP4[code] * np.where(np.signbit(v), F32(-1), F32(1))).astype(F32)
        err = np.abs((value * scale[:, None]) * outer - blocks).astype(F32)
        for _ in range(4):
            err = err[:, 0::2] + err[:, 1::2]
        candidates.append(((value * scale[:, None]).astype(F32), err[:, 0]))
    six, four = candidates
    choose_four = four[1] < six[1]
    operand = np.where(choose_four[:, None], four[0], six[0]).reshape(row.shape)
    return operand.astype(F32), outer


def q_mxfp8(row):
    blocks = row.astype(F32).reshape(-1, 32)
    bmax = np.abs(blocks).max(axis=1)
    with np.errstate(divide="ignore"):
        exponent = np.where(bmax > 0, np.floor(np.log2(bmax)) - 8, -127)
    scale = np.exp2(np.clip(exponent, -127, 127)).astype(F32)
    return (e4m3_rne_signed(blocks / scale[:, None]) * scale[:, None]).astype(F32).reshape(row.shape)


def capped_swiglu(gu, gamma, gate_cap, up_cap):
    g, u = gu[0::2].astype(F32), gu[1::2].astype(F32)
    sg = sigmoid(g)
    g = F32(gate_cap) * np.tanh(g / F32(gate_cap))
    u = F32(up_cap) * np.tanh(u / F32(up_cap))
    return (g * sg * u * F32(gamma)).astype(F32)


class Inkling:
    def __init__(self, model_dir, expert_cache_bytes=24 << 30):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text())["text_config"]
        files = sorted(model_dir.glob("model*.safetensors"))
        self.ck = Checkpoint(files[0])
        c = self.config
        self.hidden = c["hidden_size"]
        self.layers = c["num_hidden_layers"]
        self.local = set(c["local_layer_ids"])
        self.block = c["n_layers_per_attn_res_block"]
        loop = c["loop_config"]
        self.core = range(loop["core_start"], loop["core_end"])
        self.passes = loop["num_passes"]
        self.heads, self.kv_heads, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.window = c["sliding_window_size"]
        self.rel_extent = c["rel_extent"]
        self.vocab = c["unpadded_vocab_size"]
        self.dense = {}
        self.experts = {}
        self.expert_bytes = 0
        self.expert_cache_bytes = expert_cache_bytes
        p = "model.llm."
        self.p = p
        scale = F32(self.hidden ** -0.5)
        self.attn_queries = {i: self.ck.f32(f"{p}attn_res.attn_queries.{i - 1}") * scale for i in range(1, self.layers)}
        self.mlp_queries = {i: self.ck.f32(f"{p}attn_res.mlp_queries.{i}") * scale for i in range(self.layers)}
        self.output_query = self.ck.f32(f"{p}attn_res.output_query") * scale
        q = self.ck.f32(f"{p}middle_cycle_cross_pass_queries")
        kv = self.ck.f32(f"{p}middle_cycle_cross_pass_kv_queries")
        self.loop_queries = np.stack((q[0::2], kv[0::2], kv[1::2], q[1::2]), axis=1).astype(F32) * scale
        self.unembed = None

    def w(self, name):
        if name not in self.dense:
            self.dense[name] = np.ascontiguousarray(self.ck.f32(self.p + name))
        return self.dense[name]

    def visit_id(self, layer, pass_idx):
        if pass_idx == 0:
            return layer
        return self.layers + (pass_idx - 1) * len(self.core) + (layer - self.core.start)

    def expert(self, layer, e):
        key = (layer, e)
        if key in self.experts:
            value = self.experts.pop(key)
            self.experts[key] = value
            return value
        base = f"{self.p}layers.{layer}.mlp.experts."
        if self.ck.header[base + "w13_weight"]["dtype"] == "U8":
            mats = []
            for name in ("w13", "w2"):
                packed = self.ck.raw(base + name + "_weight")[e]
                values = FP4_PAIRS[packed].reshape(packed.shape[0], -1)
                scales = E4M3[self.ck.raw(base + name + "_weight_scale")[e]].astype(np.float16)
                alpha = F32(self.ck.f32(base + name + "_weight_amax")[e] / self.ck.f32(base + name + "_weight_outer_scale_denominator")[0])
                mats.append(((values * np.repeat(scales, 16, axis=1)).astype(np.float16), alpha))
            value = ("nvfp4", mats)
        else:
            mats = []
            for name in ("w13", "w2"):
                weights = E4M3[self.ck.raw(base + name + "_weight")[e]]
                scales = np.exp2(self.ck.raw(base + name + "_weight_scale")[e].astype(F32) - 127)
                mats.append((weights * np.repeat(scales, 32, axis=1)).astype(F32))
            value = ("mxfp8", mats)
        size = sum(m[0].nbytes if isinstance(m, tuple) else m.nbytes for m in value[1])
        self.experts[key] = value
        self.expert_bytes += size
        while self.expert_bytes > self.expert_cache_bytes:
            old_key = next(iter(self.experts))
            old = self.experts.pop(old_key)
            self.expert_bytes -= sum(m[0].nbytes if isinstance(m, tuple) else m.nbytes for m in old[1])
        return value

    def moe(self, layer, x):
        c = self.config
        logits = (x @ self.w(f"layers.{layer}.mlp.gate.weight").T).astype(F32)
        selection = sigmoid(logits) + self.w(f"layers.{layer}.mlp.gate.bias")
        order = np.lexsort((np.arange(logits.size), -selection))
        ids = order[: c["num_experts_per_tok"]]
        lg = logits[ids]
        logsig = np.minimum(lg, 0) - np.log1p(np.exp(-np.abs(lg)))
        m = logsig.max()
        s = np.exp(logsig - m)
        total = ((s[0] + s[2]) + (s[1] + s[3])) + ((s[4] + s[6]) + (s[5] + s[7]))
        gammas = np.exp(logsig - (m + np.log(total))) * F32(c["route_scale"])
        rows = {}
        for e, gamma in zip(ids, gammas):
            kind, mats = self.expert(layer, int(e))
            if kind == "nvfp4":
                (w13, a13), (w2, a2) = mats
                xq, outer = q_nvfp4(x)
                gu = ((xq @ w13.T.astype(F32)) * outer * a13).astype(F32)
                h = bf16(capped_swiglu(gu, gamma, c["mlp_gate_softcap"], c["mlp_up_softcap"]))
                hq, h_outer = q_nvfp4(h)
                rows[int(e)] = bf16((hq @ w2.T.astype(F32)) * h_outer * a2)
            else:
                w13, w2 = mats
                gu = bf16(q_mxfp8(x) @ w13.T)
                h = capped_swiglu(gu, gamma, c["mlp_gate_softcap"], c["mlp_up_softcap"])
                rows[int(e)] = bf16(q_mxfp8(h) @ w2.T)
        total = np.zeros(self.hidden, dtype=F32)
        for e in sorted(rows):
            total = total + rows[e]
        return bf16(total), [int(i) for i in ids]

    def clip(self, x, scale, threshold, tau):
        xs = x.astype(F32) * scale if scale is not None else x.astype(F32)
        norm = np.sqrt(np.sum(xs * xs, axis=-1, keepdims=True, dtype=F32))
        y = xs * np.minimum(F32(threshold) / (F32(1e-6) + norm), F32(1))
        if tau is not None:
            y = bf16(y) * F32(tau)
        return bf16(y)

    def attention(self, layer, visit, sources, pos, cache):
        c = self.config
        L = f"layers.{layer}.attn."
        src_q, src_k, src_v = sources
        is_local = layer in self.local
        q = bf16(src_q @ self.w(L + "wq_du.weight").T).reshape(self.heads, self.head_dim)
        r = bf16(src_q @ self.w(L + "wr_du.weight").T).reshape(self.heads, c["d_rel"])
        gate = bf16(src_q @ self.w(L + "gate_attn.weight").T)
        sinks = bf16(src_q @ self.w(L + "attn_sink_proj.weight").T)
        k_raw = bf16(src_k @ self.w(L + "wk_dv.weight").T)
        v_raw = bf16(src_v @ self.w(L + "wv_dv.weight").T)
        convs = []
        for name, raw, src in (("k", k_raw, src_k), ("v", v_raw, src_v)):
            state = cache.setdefault((visit, name), np.zeros((3, raw.size), dtype=F32))
            if is_local:
                weight = bf16(self.w(L + f"{name}_sconv.weight")[:, 0, :])
                y, state = static_conv(raw, state, weight)
            else:
                dyn = bf16(src @ self.w(L + f"{name}_sconv.weight_proj").T).reshape(self.kv_heads, 4)
                static = self.w(L + f"{name}_sconv.static_w")
                history = np.concatenate([state, raw[None, :]], axis=0)
                acc = np.zeros_like(raw)
                for lag in range(4):
                    weight = static[lag] + np.repeat(dyn[:, lag], self.head_dim)
                    acc = acc + history[3 - lag] * weight
                y = bf16((acc + raw) * F32(1 / 3))
                state = np.concatenate([state[1:], raw[None, :]], axis=0)
            cache[(visit, name)] = state
            convs.append(y)
        k = convs[0].reshape(self.kv_heads, self.head_dim)
        v = convs[1].reshape(self.kv_heads, self.head_dim)
        tau = None
        if not is_local:
            tau = 1 + c["log_scaling_alpha"] * math.log(max((pos + 1) / c["log_scaling_n_floor"], 1.0))
        q = self.clip(q, self.w(L + "q_l2_clip.scale"), c["qk_l2_clip_threshold"], tau)
        k = self.clip(k, None, c["qk_l2_clip_threshold"], None)
        v = bf16(v * (F32(1) / np.sqrt(np.sum(v * v, axis=-1, keepdims=True, dtype=F32) / F32(self.head_dim) + F32(1e-6))))
        r = self.clip(r, self.w(L + "rel_l2_clip.scale"), c["rel_l2_clip_threshold"], tau)
        rel = bf16(r @ self.w(L + "rel_logits_proj.proj"))
        keys = cache.setdefault((visit, "K"), [])
        values = cache.setdefault((visit, "V"), [])
        positions = cache.setdefault((visit, "P"), [])
        keys.append(k)
        values.append(v)
        positions.append(pos)
        kp = np.array(positions)
        dist = pos - kp
        allowed = dist >= 0
        if is_local:
            allowed &= dist <= self.window - 1
        K = np.stack(keys)[allowed]
        V = np.stack(values)[allowed]
        d = dist[allowed]
        out = np.zeros((self.heads, self.head_dim), dtype=F32)
        lse = np.zeros(self.heads, dtype=F32)
        group = self.heads // self.kv_heads
        for h in range(self.heads):
            s = (K[:, h // group] @ q[h]) * F32(self.head_dim ** -0.5)
            bias = np.where(d < self.rel_extent, rel[h, np.minimum(d, self.rel_extent - 1)], F32(0))
            s = (s + bias).astype(F32)
            mx = s.max()
            e = np.exp(s - mx)
            lse[h] = mx + np.log(e.sum(dtype=F32))
            out[h] = (e / e.sum(dtype=F32)) @ V[:, h // group]
        y = out.reshape(-1) * sigmoid(gate) * np.repeat(sigmoid(lse - sinks), self.head_dim)
        return bf16(bf16(y) @ self.w(L + "wo_ud.weight").T)

    def controllers(self, layer, side, destinations, messages):
        names = ("attn", "k", "v") if side == "attn" else ("mlp",)
        out = []
        for name, d, x in zip(names, destinations, messages):
            base = f"layers.{layer}.controllers.{name}."
            d = d.astype(F32)
            normed = np.concatenate([bf16(d * rsqrt_mean_square(d)), bf16(x * rsqrt_mean_square(x))])
            h = bf16(normed @ self.w(base + "down_weight").T)
            gain = h @ self.w(base + "gain_weight").T
            forget = h @ self.w(base + "forget_weight").T
            out.append(((F32(1) - np.tanh(forget)) * d + np.tanh(gain) * x).astype(F32))
        return out

    def layer(self, state, layer, pass_idx, bank, pos, cache, trace):
        visit = self.visit_id(layer, pass_idx)
        blocks, partial = state
        is_local = layer in self.local
        if not blocks:
            streams = [partial] * (1 if is_local else 3)
        else:
            queries = self.attn_queries[layer].reshape(-1, self.hidden)
            streams = list(attn_res(blocks + [partial], None, queries, True))
        L = f"layers.{layer}."
        if bank is not None:
            messages = attn_res(bank, None, self.loop_queries[layer - self.core.start, 0:3], True)
            destinations = streams * 3 if is_local else streams
            mixed = self.controllers(layer, "attn", destinations, messages)
            if is_local:
                sources = [rms_norm(m, self.w(L + "attn_norm.weight")) for m in mixed]
            else:
                sources = [rms_norm(m, self.w(L + f"{n}_input_norm.weight")) for m, n in zip(mixed, "qkv")]
        elif is_local:
            sources = [rms_norm(streams[0], self.w(L + "attn_norm.weight"))] * 3
        else:
            sources = [rms_norm(s, self.w(L + f"{n}_input_norm.weight")) for s, n in zip(streams, "qkv")]
        attention = self.attention(layer, visit, sources, pos, cache)
        state_a = cache.setdefault((visit, "attn_sconv"), np.zeros((3, self.hidden), dtype=F32))
        attention, cache[(visit, "attn_sconv")] = static_conv(attention, state_a, bf16(self.w(L + "attn_sconv.weight")[:, 0, :]))
        boundary = layer % self.block == 0
        if boundary:
            mlp_in = attn_res(blocks + [partial, attention], None, self.mlp_queries[layer][None, :], True)[0]
        else:
            mlp_in = attn_res(blocks + [partial], attention, self.mlp_queries[layer][None, :], True)[0]
        if bank is not None:
            messages = attn_res(bank, None, self.loop_queries[layer - self.core.start, 3:4], True)
            mlp_in = self.controllers(layer, "mlp", [mlp_in], messages)[0]
        mlp, ids = self.moe(layer, rms_norm(mlp_in, self.w(L + "mlp_norm.weight")))
        state_m = cache.setdefault((visit, "mlp_sconv"), np.zeros((3, self.hidden), dtype=F32))
        conv, cache[(visit, "mlp_sconv")] = static_conv(mlp, state_m, bf16(self.w(L + "mlp_sconv.weight")[:, 0, :]))
        if trace is not None:
            trace.append({"visit": visit, "layer": layer, "experts": ids})
        if boundary:
            return blocks + [partial], bf16(attention + conv)
        return blocks, bf16(partial + attention + conv)

    def step(self, token, pos, cache, trace=None):
        embed = self.ck.f32_row(self.p + "embed.weight", token)
        x = rms_norm(embed, self.w("embed_norm.weight"))
        state = ([], x)
        for layer in range(self.core.start):
            state = self.layer(state, layer, 0, None, pos, cache, trace)
        prelude = (list(state[0]), state[1])
        bank = None
        for pass_idx in range(self.passes):
            if pass_idx:
                blocks, partial = state
                bank = blocks[len(prelude[0]) + 1:] + [partial]
                state = (list(prelude[0]), prelude[1])
            for layer in self.core:
                state = self.layer(state, layer, pass_idx, bank, pos, cache, trace)
        for layer in range(self.core.stop, self.layers):
            state = self.layer(state, layer, 0, None, pos, cache, trace)
        blocks, partial = state
        final = attn_res(blocks + [partial], None, self.output_query[None, :], True)[0]
        hidden = rms_norm(final, self.w("norm.weight"))
        if self.unembed is None:
            self.unembed = np.ascontiguousarray(self.ck.f32(self.p + "unembed.weight"))
        logits = bf16(hidden @ self.unembed.T)
        cap = F32(self.config["final_logit_softcapping"])
        logits = bf16(cap * np.tanh(logits / cap))
        logits[self.vocab:] = -np.inf
        return logits

    def generate(self, input_ids, steps, trace=None):
        cache = {}
        logits = None
        for pos, token in enumerate(input_ids):
            logits = self.step(token, pos, cache, trace)
        out = []
        for i in range(steps):
            lse = logits.max() + np.log(np.exp(logits - logits.max()).sum(dtype=np.float64))
            top = np.argsort(-logits, kind="stable")[:5]
            token = int(top[0])
            out.append({"id": token, "logprob": float(logits[token] - lse), "top": [[float(logits[t] - lse), int(t)] for t in top]})
            if i + 1 < steps:
                logits = self.step(token, len(input_ids) + i, cache, trace)
        return out


def main():
    parser = argparse.ArgumentParser(description="Exact numpy reference for Inkling-2-Mini, one token at a time.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-ids", required=True, help="JSON list of prompt token ids")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    model = Inkling(args.model)
    result = model.generate(json.loads(args.input_ids), args.steps)
    json.dump({"output_ids": [r["id"] for r in result], "steps": result}, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()