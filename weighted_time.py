import pandas as pd
import numpy as np
import math

def to_rating(u_ts):
  u_wt = np.zeros((u_ts.shape))
  for i in range(len(u_ts)):
    u_t = u_ts[i]
    hl = np.amin(u_t) + (np.amax(u_t) - np.amin(u_t)) / 2 # get mean timestamp
    t_recent = np.amax(u_t)
    func = lambda t: math.exp(-math.log(2) * (t_recent - t) / hl)
    u_wt[i] = np.array([func(x) for x in u_t])
  return u_wt