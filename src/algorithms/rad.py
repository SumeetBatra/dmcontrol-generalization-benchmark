import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.algorithms.sac import SAC


class RAD(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
