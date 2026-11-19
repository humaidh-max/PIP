"""One PIP control tick: perceive -> feel -> act -> (feedback) -> log. Works with the sim or real perception."""
import numpy as np
from .brain import PipBrain, featurize
from .log import ExperienceLog
from .kinematics import PipBody
from .spec import *

PHRASES = {"greet": ["Hi!", "Hello, you."], "comfort": ["I'm here.", "Hey, it's okay."], "notify": ["Heads up, something's off."],
           "ask": ["What's that?", "Everything alright?"], "cheer": ["Yay!", "Nice one!"]}

class PipAgent:
    def __init__(s, brain=None, log=None):
        s.brain, s.log, s.body = brain or PipBrain(), log or ExperienceLog(), PipBody(); s.last_r = 0.0; s.pend = None
    def tick(s, obs, alert=0.0, explore=True):
        f = featurize(obs, s.last_r, alert); prev, new = s.brain.feel(f)
        c, snd, sp = s.brain.act(f, explore); pa = s.brain.proactive(f, explore)
        s.body.command(c); s.pend = dict(f=f, s_prev=prev, s_new=new, a_c=c, a_s=snd, a_sp=sp, pa=pa)
        say = PHRASES.get(SPEECH[sp], [""])[0] if sp else ""
        return dict(mood=s.brain.mood(), valence=float(new[0]), arousal=float(new[1]), cont=c, sound=SOUNDS[snd],
                    speech=say, proactive=PROACTIVE[pa], _f=f, _snd=snd, _sp=sp)
    def feedback(s, r, emo_target=None, welcomed=None):
        p = s.pend; kw = dict(p, r=r)
        if emo_target is not None: kw["emo_t"] = emo_target
        if welcomed is not None: kw["y"] = welcomed
        s.log.add(**kw); s.last_r = r
