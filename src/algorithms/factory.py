from src.algorithms.sac import SAC
from src.algorithms.rad import RAD
from src.algorithms.curl import CURL
from src.algorithms.pad import PAD
from src.algorithms.soda import SODA
from src.algorithms.drq import DrQ
from src.algorithms.svea import SVEA

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
