"""Smart-home controller + personal assistant. A small neural intent classifier (hashed n-grams -> MLP) picks the
intent; slots (device, number) are filled from the device registry. Swap SmartHome backends for Home Assistant/MQTT."""
import re, heapq, zlib, numpy as np
from .nn import MLP, softmax
INTENTS = ["light_on", "light_off", "set_temp", "lock", "unlock", "timer", "reminder", "status", "chat"]
_D = 512
def _vec(text):
    w = re.findall(r"[a-z0-9]+", text.lower()); g = w + [a + "_" + b for a, b in zip(w, w[1:])]; v = np.zeros(_D, np.float32)
    for t in g: v[zlib.crc32(t.encode()) % _D] += 1
    return v / (np.linalg.norm(v) + 1e-6)
_SEED = {"light_on": ["turn on the {d}", "switch on {d}", "{d} on please", "lights on in the {d}"],
 "light_off": ["turn off the {d}", "switch off {d}", "{d} off", "kill the {d}", "lights off"],
 "set_temp": ["set the thermostat to {n}", "make it {n} degrees", "heat to {n}", "set temperature {n}"],
 "lock": ["lock the {d}", "please lock up", "secure the {d}"], "unlock": ["unlock the {d}", "open the {d} lock", "let me in, unlock"],
 "timer": ["set a timer for {n} minutes", "timer {n} minutes", "wake me in {n} minutes"],
 "reminder": ["remind me to {x}", "remember to {x}", "reminder {x}"],
 "status": ["is the {d} on", "what's the status of the {d}", "is anything open", "how is everyone doing", "did dad eat"],
 "chat": ["how are you", "tell me a joke", "good morning pip", "what do you think", "thanks pip", "i'm bored"]}

class Device:
    def __init__(s, name, kind, **st): s.name, s.kind, s.state = name, kind, dict(st)

class SmartHome:
    def __init__(s, devices): s.dev = {d.name: d for d in devices}
    def find(s, text):
        t = set(re.findall(r"[a-z]+", text.lower())); best = max(s.dev.values(), key=lambda d: len(t & set(d.name.split())), default=None)
        return best if best and t & set(best.name.split()) else None
    def set(s, name, **kw): s.dev[name].state.update(kw)       # replace body with requests.post(...) to Home Assistant

class Assistant:
    def __init__(s, home, seed=0):
        s.home, s.net, s.timers, s.rng = home, MLP([_D, 64, len(INTENTS)], seed, 5e-3), [], np.random.default_rng(seed); s._train()
    def _train(s):
        names = list(s.home.dev) + ["lamp", "front door", "heating"]; X, Y = [], []
        for k, tpls in _SEED.items():
            for t in tpls:
                for _ in range(8):
                    X.append(_vec(t.format(d=s.rng.choice(names), n=s.rng.integers(15, 30), x="call mom"))); Y.append(INTENTS.index(k))
        X, Y = np.array(X), np.array(Y)
        for _ in range(150):
            p = softmax(s.net.forward(X, cache=True)); d = p.copy(); d[np.arange(len(Y)), Y] -= 1; s.net.step(s.net.backward(d / len(Y)))
    def handle(s, text, now_min=0):
        p = softmax(s.net.forward(_vec(text)[None])[0]); intent = INTENTS[int(p.argmax())]; num = re.search(r"\d+", text); d = s.home.find(text)
        if p.max() < .45: intent = "chat"
        if intent in ("light_on", "light_off") and d: s.home.set(d.name, on=intent == "light_on"); return intent, f"{d.name} {'on' if intent=='light_on' else 'off'}"
        if intent == "set_temp" and num and (d := d or s.home.find("thermostat")): s.home.set(d.name, target=int(num.group())); return intent, f"thermostat {num.group()}°"
        if intent in ("lock", "unlock") and d: s.home.set(d.name, locked=intent == "lock"); return intent, f"{d.name} {intent}ed"
        if intent == "timer" and num: heapq.heappush(s.timers, now_min + int(num.group())); return intent, f"timer {num.group()} min"
        if intent == "reminder": return intent, "ok, I'll remind you"
        if intent == "status" and d: return intent, f"{d.name}: {d.state}"
        return ("chat" if intent == "chat" else "unresolved"), "…"
    def due(s, now_min):
        out = []
        while s.timers and s.timers[0] <= now_min: out.append(heapq.heappop(s.timers))
        return out
