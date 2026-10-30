#!/usr/bin/env python3
"""forge_brain: a small neural network that learns what your engine prompts mean.

What it learns (one shared hidden layer, several output heads):
  intent   new_engine | edit
  layout   inline | v | flat | radial | radial2 (twin-row) | none
  firing   even | pairs | groups | order | none
  taper    yes | no            invert  yes | no
  exhaust  single | per_bank | per_cylinder | none
  target   what an edit is about: redline idle cam ports compression intake exhaust gearing clutch power torque other
  dir      up | down | set | none

Numbers (cylinder count, litres, rpm, mph, idle value, ...) are read by a unit-aware tagger, because a
network trained on a few hundred prompts is better at *meaning* than at arithmetic.

It trains on (1) synthetic paraphrases generated here, (2) the labelled prompts in seed_prompts.json,
and (3) your own corrections in user_data.jsonl, which are weighted heavily. Every time you correct it
(`engine_forge.py --teach`) it retrains in a few seconds, so it gets better at YOUR wording.
Pure numpy; no other dependencies.
"""
import json
import math
import os
import random
import re
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "brain_model.npz")
USER_DATA = os.path.join(HERE, "user_data.jsonl")
SEED_DATA = os.path.join(HERE, "seed_prompts.json")

HEADS = {
    "intent": ["new_engine", "edit"],
    "layout": ["none", "inline", "v", "flat", "radial", "radial2"],
    "firing": ["none", "even", "pairs", "groups", "order"],
    "taper": ["no", "yes"],
    "invert": ["no", "yes"],
    "exhaust": ["none", "single", "per_bank", "per_cylinder"],
    "target": ["none", "redline", "idle", "cam", "ports", "compression", "intake", "exhaust", "gearing", "clutch",
               "power", "torque", "other"],
    "dir": ["none", "up", "down", "set"],
}
D = 3072
H = 128


# ------------------------------------------------------------------ features
def norm_tokens(text):
    t = text.lower()
    t = re.sub(r"\d", "#", t)
    return re.findall(r"[a-z#][a-z#\.\-:]*|[0-9#]+", t)


def featurize(text):
    toks = norm_tokens(text)
    x = np.zeros(D, dtype=np.float32)

    def add(s, w=1.0):
        x[zlib.crc32(s.encode()) % D] += w
    for i, w in enumerate(toks):
        add("u:" + w)
        if i + 1 < len(toks):
            add("b:" + w + "_" + toks[i + 1])
        if len(w) >= 4:
            for j in range(len(w) - 2):
                add("c:" + w[j:j + 3], 0.5)
    n = np.linalg.norm(x)
    return x / n if n else x


# ------------------------------------------------------------------ numeric tagger
def tag_numbers(text):
    t = text.lower()
    out = {}
    m = re.search(r"twin[- ]?(?:row|bank)\s+radial\s*-?(\d+)|(\d+)[- ]?cyl(?:inder)?s?\s+(?:twin|double|two)[- ]?(?:row|bank)?\s*radial", t)
    if m:
        out["cylinders"] = int(m.group(1) or m.group(2))
    if "cylinders" not in out:
        m = re.search(r"\b(?:v|flat|boxer|radial|inline|straight|i|w)\s*-?(\d{1,2})\b", t)
        if m:
            out["cylinders"] = int(m.group(1))
    if "cylinders" not in out:
        m = re.search(r"\b(\d{1,2})[- ]?cyl(?:inder)?s?\b", t)
        if m:
            out["cylinders"] = int(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|°|degree)s?\s*v\b|v[- ]?angle\s*(?:of\s*)?(\d+)", t)
    if m:
        out["v_angle"] = int(m.group(1) or m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:l\b|litre|liter)", t)
    if m:
        out["displacement_l"] = float(m.group(1))
    m = re.search(r"\b(\d{3,5})\s*cc\b", t)
    if m:
        out["cc"] = int(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\s*(?:rpm|limit)?\b|(\d{4,6})\s*rpm", t)
    if m:
        out["rpm"] = int(float(m.group(1)) * 1000) if m.group(1) else int(m.group(2))
    m = re.search(r"bore\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", t)
    if m:
        out["bore_mm"] = float(m.group(1))
    m = re.search(r"stroke\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", t)
    if m:
        out["stroke_mm"] = float(m.group(1))
    m = re.search(r"\b(0\.\d{3,6})\b", t)
    if m:
        out["idle_value"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*mph", t)
    if m:
        out["mph"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*:\s*1", t)
    if m:
        out["compression_ratio"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", t)
    if m:
        out["percent"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\b|\b(double|triple|half)\b", t)
    if m:
        out["multiple"] = {"double": 2.0, "triple": 3.0, "half": 0.5}.get(m.group(2)) if m.group(2) else float(m.group(1))
    m = re.search(r"firing order\s*[:=]?\s*([\d\s\-,]+)|order\s+(?:of\s+)?(\d+(?:[\s\-,]+\d+){3,})", t)
    if m:
        out["order"] = [int(x) for x in re.findall(r"\d+", m.group(1) or m.group(2))]
    m = re.search(r"(?:third|3rd)\s*gear[^.]*?(\d+)\s*mph|(\d+)\s*mph[^.]*?(?:third|3rd)", t)
    if m:
        out["third_mph"] = float(m.group(1) or m.group(2))
    m = re.search(r"groups? of (\d+)|in (?:threes|fours|pairs|twos)", t)
    if m:
        out["group_size"] = int(m.group(1)) if m.group(1) else {"threes": 3, "fours": 4, "pairs": 2, "twos": 2}[m.group(0).split()[-1]]
    return out


# ------------------------------------------------------------------ synthetic data
def _pick(r, xs):
    return r.choice(xs)


def synth(n=3500, seed=1):
    r = random.Random(seed)
    rows = []
    lay = {
        "inline": ["inline {n}", "straight {n}", "i{n}", "{n} cylinder inline", "inline-{n} engine"],
        "v": ["v{n}", "{n} cylinder v engine", "v{n} engine", "{n} cyl v"],
        "flat": ["flat {n}", "boxer {n}", "flat-{n}", "{n} cylinder boxer"],
        "radial": ["radial {n}", "{n} cylinder radial", "single row radial {n}", "radial engine"],
        "radial2": ["twin row radial {n}", "twin bank radial {n}", "{n} cylinder double row radial", "two row radial {n}",
                    "dual row radial {n}"],
    }
    counts = {"inline": [2, 3, 4, 5, 6, 8], "v": [6, 8, 10, 12, 16], "flat": [4, 6, 8, 12], "radial": [5, 7, 9],
              "radial2": [10, 14, 18]}
    fir = {
        "pairs": ["fire two at a time", "cylinders fire two by two", "twin firing pulses", "fire in pairs", "fires in pairs of two", "cylinders fire together in pairs", "paired firing",
                  "make it fire in pairs of two"],
        "even": ["even fire", "evenly spaced firing", "perfectly even spacing", "firing perfectly spaced"],
        "groups": ["fire in groups of three", "groups of {g} fire together", "{g} cylinders fire at once",
                   "each bank fires together"],
        "order": ["firing order {o}", "with a firing order of {o}", "order {o}"],
    }
    tap = ["sizes shrink one after the next", "bores getting progressively smaller", "get smaller along the bank",
           "shrinking cylinder sizes", "each one a bit smaller than the one before", "stepped down in size",
           "each cylinder gets smaller sequentially", "tapering design", "cylinders get smaller one after another",
           "tapered cylinder sizes", "every cylinder smaller than the last", "decreasing bore"]
    inv = ["and the back bank flipped", "reverse the sizes on the second bank", "second row runs the other way round",
           "rear bank in the opposite order", "with the order inverted on the rear bank", "inverted on the second row", "opposite order on the other side",
           "mirror the taper on the second bank"]
    exh = {
        "single": ["single exhaust", "all on one exhaust", "shared exhaust", "one exhaust for everything"],
        "per_bank": ["one exhaust per bank", "separate exhaust for each bank", "each bank gets its own exhaust"],
        "per_cylinder": ["every cylinder on its own exhaust", "individual exhausts", "separate exhaust per cylinder"],
    }
    new_pre = ["make a", "build me a", "i want a", "can you make a", "design a", "create a", "give me a", "make me a"]
    for _ in range(n):
        if r.random() < 0.55:
            L = r.choice(list(lay))
            n_c = r.choice(counts[L])
            parts = [_pick(r, new_pre) + " " + _pick(r, lay[L]).format(n=n_c)]
            lab = dict(intent="new_engine", layout=L, firing="none", taper="no", invert="no", exhaust="none",
                       target="none", dir="none")
            if r.random() < 0.5:
                parts.append(r.choice(["%.1fL" % r.choice([1.0, 2.0, 3.4, 4.7, 6.0, 6.8]), "%dcc" % r.choice([1250, 2000, 5000]),
                                       "%dk rpm" % r.choice([7, 9, 12]), "%d rpm limit" % r.choice([8000, 10000, 12000])]))
            if r.random() < 0.5:
                f = r.choice(list(fir))
                lab["firing"] = f
                parts.append(_pick(r, fir[f]).format(g=r.choice([2, 3, 4]), o="-".join(map(str, r.sample(range(1, 9), 4)))))
            if r.random() < 0.4:
                lab["taper"] = "yes"
                parts.append(_pick(r, tap))
                if r.random() < 0.5:
                    lab["invert"] = "yes"
                    parts.append(_pick(r, inv))
            if r.random() < 0.3:
                e = r.choice(list(exh))
                lab["exhaust"] = e
                parts.append(_pick(r, exh[e]))
            r.shuffle(parts[1:])
            rows.append((" ".join(parts), lab))
        else:
            tgt = r.choice(["redline", "idle", "cam", "ports", "compression", "intake", "exhaust", "gearing", "clutch", "power",
                            "torque", "other"])
            T = {
                "redline": ([("let it spin faster up to %dk" % r.choice([11, 13, 15]), "set"), ("allow higher revs", "up"),
                             ("it hits the limiter too early", "up"), ("cap the revs lower", "down"),("increase the rpm limit to %dk" % r.choice([9, 10, 12, 14]), "set"), ("raise the redline", "up"),
                             ("make it rev higher", "up"), ("more rpm", "up"), ("increase the rpm", "up"),
                             ("lower the rev limit", "down"), ("rpm limit %dk" % r.choice([8, 11]), "set")]),
                "idle": ([("it dies when i let off the throttle", "up"), ("engine cuts out at idle", "up"),
                          ("it idles way too fast", "down"), ("idles too quick", "down"), ("tickover is too high", "down"),
                          ("sits at idle too low", "up"),("make the idle %s" % r.choice(["0.9987", "0.9991", "0.99835", "0.9985"]), "set"),
                          ("it stalls at idle", "up"), ("increase the idle it is struggling", "up"),
                          ("the idle is too high", "down"), ("lower the idle", "down"), ("make it 0.9984", "set"),
                          ("raise the idle to around 1500", "up")]),
                "cam": ([("rework the cam for high rpm", "up"), ("give it longer cam duration", "up"),
                         ("bigger cams", "up"), ("make the cam milder", "down"), ("increase cam duration", "up"),
                         ("rework the main cylinders cam for high rpm", "up")]),
                "ports": ([("improve the ports", "up"), ("increase port flow", "up"), ("more flow through the heads", "up"),
                           ("ports seemed good go further", "up"), ("restrict the ports a bit", "down")]),
                "compression": ([("squeeze the mixture harder", "up"), ("higher static compression", "up"),
                                 ("less squeeze in the cylinder", "down"),("increase compression", "up"), ("make the chamber volume %dcc" % r.choice([10, 37, 46]), "set"),
                                 ("raise the compression ratio to 13:1", "set"), ("lower the compression", "down"),
                                 ("reduce the chamber volume", "up")]),
                "intake": ([("bigger intake", "up"), ("increase the plenum", "up"), ("more intake flow", "up"),
                            ("make the intake runner cross section 5.5cm2", "set"), ("smaller plenum", "down")]),
                "exhaust": ([("bigger exhaust", "up"), ("free flowing exhaust", "up"), ("increase the collector size", "up"),
                             ("make the exhaust runner cross section 40cm2", "set"), ("longer exhaust runner", "up")]),
                "gearing": ([("make the gearbox taller", "up"), ("tighter gear spacing", "down"),("make the gears longer", "up"), ("shorter gearing", "down"),
                             ("make third gear reach 100 mph", "set"), ("rework the weights gearbox and such for this size engine", "set"),
                             ("gears can be lengthened to the 180 mph range", "up")]),
                "clutch": ([("the clutch slips beef it up", "up"), ("stronger clutch", "up"),("increase clutch torque by 10 percent", "up"), ("the clutch torque needs to go up", "up"),
                            ("clutch torque should be around 225", "set"), ("lower the clutch torque", "down")]),
                "power": ([("tune it for maximum horsepower", "up"), ("squeeze out every last bit of horsepower", "up"), ("make it quicker", "up"),("more power", "up"), ("increase power from here", "up"), ("aim it at absolute peak power", "up"),
                           ("make it faster", "up"), ("ultra high performance intake exhaust and cams", "up")]),
                "torque": ([("it lost low end grunt", "up"), ("needs more pull down low", "up"), ("bottom end feels weak", "up"),("more torque", "up"), ("it is making less torque fix that", "up"), ("increase the torque", "up")]),
                "other": ([("extend the rod length by 25mm", "up"), ("make the small cylinder angled 25 degrees", "set"),
                           ("share the same journal", "set"), ("make it a single exhaust", "set"),
                           ("reduce the simulation frequency to 5k", "down")]),
            }[tgt]
            text, d = r.choice(T)
            if r.random() < 0.3:
                text = r.choice(["okay ", "can you ", "please ", "now ", "so "]) + text
            lab = dict(intent="edit", layout="none", firing="none", taper="no", invert="no", exhaust="none", target=tgt, dir=d)
            rows.append((text, lab))
    return rows


def load_rows():
    rows = synth()
    if os.path.exists(SEED_DATA):
        for it in json.load(open(SEED_DATA)):
            rows += [(it["prompt"], it["labels"])] * 6
    if os.path.exists(USER_DATA):
        for line in open(USER_DATA):
            if line.strip():
                it = json.loads(line)
                rows += [(it["prompt"], it["labels"])] * 12
    return rows


# ------------------------------------------------------------------ network
class Brain:
    def __init__(self):
        self.p = None

    def _init(self, rng):
        self.p = {"W1": rng.normal(0, 0.05, (D, H)).astype(np.float32), "b1": np.zeros(H, np.float32)}
        for h, cl in HEADS.items():
            self.p["W_" + h] = rng.normal(0, 0.1, (H, len(cl))).astype(np.float32)
            self.p["b_" + h] = np.zeros(len(cl), np.float32)

    def _forward(self, X):
        z = X @ self.p["W1"] + self.p["b1"]
        hid = np.maximum(z, 0)
        outs = {}
        for h in HEADS:
            lg = hid @ self.p["W_" + h] + self.p["b_" + h]
            lg -= lg.max(1, keepdims=True)
            e = np.exp(lg)
            outs[h] = e / e.sum(1, keepdims=True)
        return z, hid, outs

    def train(self, rows=None, epochs=30, lr=0.003, seed=0, verbose=False):
        rows = rows or load_rows()
        rng = np.random.default_rng(seed)
        X = np.stack([featurize(t) for t, _ in rows])
        Y = {h: np.array([HEADS[h].index(l.get(h, HEADS[h][0])) for _, l in rows]) for h in HEADS}
        self._init(rng)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v) for k, v in self.p.items()}
        t_step = 0
        N = len(rows)
        for ep in range(epochs):
            idx = rng.permutation(N)
            tot = 0.0
            for s in range(0, N, 128):
                b = idx[s:s + 128]
                xb = X[b]
                z, hid, outs = self._forward(xb)
                grads = {k: np.zeros_like(v_) for k, v_ in self.p.items()}
                dhid = np.zeros_like(hid)
                for h in HEADS:
                    y = Y[h][b]
                    pr = outs[h]
                    tot += -np.log(pr[np.arange(len(b)), y] + 1e-9).sum()
                    g = pr.copy()
                    g[np.arange(len(b)), y] -= 1
                    g /= len(b)
                    grads["W_" + h] = hid.T @ g
                    grads["b_" + h] = g.sum(0)
                    dhid += g @ self.p["W_" + h].T
                dz = dhid * (z > 0)
                grads["W1"] = xb.T @ dz
                grads["b1"] = dz.sum(0)
                t_step += 1
                for k in self.p:
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    mh = m[k] / (1 - 0.9 ** t_step)
                    vh = v[k] / (1 - 0.999 ** t_step)
                    self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            if verbose:
                print("epoch %d loss %.4f" % (ep + 1, tot / N))
        np.savez(MODEL_PATH, **self.p)

    def load(self):
        if os.path.exists(MODEL_PATH):
            d = np.load(MODEL_PATH)
            self.p = {k: d[k] for k in d.files}
            return True
        return False

    def ensure(self):
        if not self.load():
            self.train()

    def predict(self, text):
        self.ensure()
        _, _, outs = self._forward(featurize(text)[None, :])
        res = {}
        for h, cl in HEADS.items():
            pr = outs[h][0]
            i = int(pr.argmax())
            res[h] = (cl[i], float(pr[i]))
        return res


def teach(prompt, labels, retrain=True):
    with open(USER_DATA, "a") as f:
        f.write(json.dumps({"prompt": prompt, "labels": labels}) + "\n")
    if retrain:
        Brain().train()


# ------------------------------------------------------------------ meaning -> spec / edits
def interpret(prompt, brain=None):
    """Return dict: intent, labels (with confidences), numbers."""
    b = brain or Brain()
    pr = b.predict(prompt)
    return {"labels": {h: v[0] for h, v in pr.items()}, "conf": {h: round(v[1], 3) for h, v in pr.items()},
            "numbers": tag_numbers(prompt)}


def spec_from_meaning(prompt, meaning):
    lab, num = meaning["labels"], meaning["numbers"]
    spec = {}
    lay = lab["layout"]
    n = num.get("cylinders")
    if lay in ("inline", "v", "flat"):
        spec["layout"] = lay
    elif lay == "radial":
        spec["layout"], spec["rows"] = "radial", 1
    elif lay == "radial2":
        spec["layout"], spec["rows"] = "radial", 2
    if n:
        spec["cylinders"] = n
    elif lay == "radial":
        spec["cylinders"] = 7
    elif lay == "radial2":
        spec["cylinders"] = 14
    if "v_angle" in num:
        spec["v_angle"] = num["v_angle"]
    if "displacement_l" in num:
        spec["displacement_l"] = num["displacement_l"]
    if "cc" in num:
        spec["displacement_l"] = num["cc"] / 1000.0
    if "rpm" in num:
        spec["redline_rpm"] = num["rpm"]
    if "bore_mm" in num:
        spec["bore_mm"] = num["bore_mm"]
    if "stroke_mm" in num:
        spec["stroke_mm"] = num["stroke_mm"]
    if "idle_value" in num:
        spec["idle_plate"] = num["idle_value"]
    if "mph" in num:
        spec["top_speed_mph"] = num["mph"]
    if "compression_ratio" in num:
        spec["compression_ratio"] = num["compression_ratio"]
    f = lab["firing"]
    if f == "pairs":
        spec["firing"] = {"mode": "pairs"}
    elif f == "groups":
        spec["firing"] = {"mode": "groups_of", "size": num.get("group_size", 2)}
    elif f == "order" and "order" in num:
        spec["firing"] = {"mode": "even", "order": num["order"]}
    elif f == "even":
        spec["firing"] = {"mode": "even"}
    if lab["taper"] == "yes":
        if "displacement_l" not in spec and "bore_mm" not in spec:
            spec["displacement_l"] = round(0.486 * spec.get("cylinders", 7), 2)
        spec["bore_mm"] = {"start_mm": spec["bore_mm"] if isinstance(spec.get("bore_mm"), float) else None,
                           "step_mm": -6.0, "invert_second": lab["invert"] == "yes"}
    if lab["exhaust"] in ("single", "per_bank", "per_cylinder"):
        spec["exhaust"] = lab["exhaust"]
    spec["name"] = prompt.strip()[:48]
    return spec


def apply_edit(spec, meaning):
    """Apply an edit meaning to a merged spec. Returns list of human-readable changes."""
    tgt, d = meaning["labels"]["target"], meaning["labels"]["dir"]
    num = meaning["numbers"]
    log = []
    up = d == "up"
    down = d == "down"

    def setv(k, v, why):
        old = spec.get(k)
        spec[k] = v
        log.append("%s: %s -> %s (%s)" % (k, old, v, why))

    def sub(k, key, v, why):
        old = spec[k].get(key)
        spec[k][key] = v
        log.append("%s.%s: %s -> %s (%s)" % (k, key, old, v, why))
    if tgt == "redline" or (tgt == "power" and False):
        if "rpm" in num:
            setv("redline_rpm", num["rpm"], "requested")
        elif down:
            setv("redline_rpm", int(round(spec["redline_rpm"] * 0.8 / 500.0) * 500), "lower rev limit")
        else:
            new = int(round(spec["redline_rpm"] * 1.3 / 500.0) * 500)
            setv("redline_rpm", new, "general 'more rpm': raise limit ~30%; gearing re-solves to keep the same speeds")
            sub("cam", "intake_duration", spec["cam"]["intake_duration"] + 15, "longer duration so the engine can use the rpm")
            sub("cam", "exhaust_duration", spec["cam"]["exhaust_duration"] + 15, "longer duration so the engine can use the rpm")
            setv("port_flow_scale", round(spec["port_flow_scale"] * 1.15, 3), "port flow +15% (raised power without losing torque in your runs)")
    elif tgt == "idle":
        if "idle_value" in num:
            setv("idle_plate", num["idle_value"], "requested")
        else:
            step = 0.0003
            nv = spec["idle_plate"] - step if up else spec["idle_plate"] + step
            setv("idle_plate", round(nv, 5), "idle rpm %s: plate %s" % ("up" if up else "down", "more open" if up else "more closed"))
    elif tgt == "cam":
        k = 15 if not down else -15
        sub("cam", "intake_duration", spec["cam"]["intake_duration"] + k, "longer cam" if k > 0 else "milder cam")
        sub("cam", "exhaust_duration", spec["cam"]["exhaust_duration"] + k, "longer cam" if k > 0 else "milder cam")
        sub("cam", "intake_center", spec["cam"]["intake_center"] - (2 if k > 0 else -2), "more overlap" if k > 0 else "less overlap")
        sub("cam", "exhaust_center", spec["cam"]["exhaust_center"] + (2 if k > 0 else -2), "more overlap" if k > 0 else "less overlap")
    elif tgt == "ports":
        f = num.get("multiple") or (1.15 if not down else 1 / 1.15)
        setv("port_flow_scale", round(spec["port_flow_scale"] * f, 3), "port flow")
    elif tgt == "compression":
        if "compression_ratio" in num:
            setv("compression_ratio", num["compression_ratio"], "requested")
        else:
            setv("compression_ratio", round(spec["compression_ratio"] + (-1.0 if down else 1.0), 2), "compression")
    elif tgt == "intake":
        f = 1.3 if not down else 1 / 1.3
        setv("intake_flow_scale", round(spec["intake_flow_scale"] * f, 3), "intake flow")
        setv("plenum_scale", round(spec["plenum_scale"] * (1.2 if not down else 1 / 1.2), 3), "plenum")
    elif tgt == "exhaust":
        f = 1.3 if not down else 1 / 1.3
        setv("exhaust_flow_scale", round(spec["exhaust_flow_scale"] * f, 3), "exhaust flow and collector")
    elif tgt == "gearing":
        if "third_mph" in num:
            setv("third_gear_mph", num["third_mph"], "requested")
        if "mph" in num and "third_mph" not in num:
            setv("top_speed_mph", num["mph"], "requested")
        if "third_mph" not in num and "mph" not in num:
            cur = spec["top_speed_mph"] or (180 if True else 160)
            setv("top_speed_mph", round(cur * (0.9 if down else 1.1)), "gearing %s" % ("shorter" if down else "longer"))
    elif tgt == "clutch":
        if "percent" in num:
            f = 1 + num["percent"] / 100.0 * (-1 if down else 1)
        else:
            f = 0.8 if down else 1.25
        setv("clutch_scale", round(spec["clutch_scale"] * f, 3), "clutch torque")
    elif tgt == "power":
        setv("port_flow_scale", round(spec["port_flow_scale"] * 1.15, 3), "port flow +15% (the lever that helped power without losing torque in your runs)")
        setv("compression_ratio", round(spec["compression_ratio"] + 0.5, 2), "compression +0.5")
        setv("redline_rpm", int(round(spec["redline_rpm"] * 1.1 / 500.0) * 500), "+10% rpm limit")
    elif tgt == "torque":
        sub("cam", "intake_duration", spec["cam"]["intake_duration"] - 10, "shorter duration for low/mid torque")
        sub("cam", "exhaust_duration", spec["cam"]["exhaust_duration"] - 10, "shorter duration for low/mid torque")
        setv("port_flow_scale", round(spec["port_flow_scale"] * 1.1, 3), "port flow +10%")
    else:
        log.append("edit target '%s' is not something the generator can change automatically; no change made" % tgt)
    return log
