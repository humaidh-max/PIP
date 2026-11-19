"""Shared constants: feature layout, emotion space, action space."""
import numpy as np
MOODS = ["calm", "happy", "curious", "sleepy", "worried", "lonely"]
FEATS = ["voice_v", "voice_a", "face_v", "face_a", "t_sin", "t_cos", "present", "idle_h",
         "pet", "ev_door", "ev_meal", "ev_noise", "ev_visitor", "last_reward", "kids_home", "alert"]
NF, NM = len(FEATS), len(MOODS)
NS = 2 + NM                       # emotion state = [valence, arousal, mood probs...]
SOUNDS = ["silence", "chirp_up", "chirp_down", "purr", "warble"]
SPEECH = ["none", "greet", "comfort", "notify", "ask", "cheer"]
PROACTIVE = ["none", "greet", "notify", "suggest"]
NC = 9                            # x,y,z,roll,pitch,yaw,antL,antR,body_yaw (each in [-1,1])
NA = NC + len(SOUNDS) + len(SPEECH)
LIMITS = np.array([8, 8, 14, 18, 22, 30, 80, 80, 60], float)   # mm, deg
