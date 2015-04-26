import random

def gibbs_sample(net, evidence, sample_fn, K, burnin, randomize=True):
	"""Does Gibbs sampling on the net. After burnin, calls sample_fn(i, net) after each sweep

	optionally, sample_fn may return True to halt the rest of the sampling
	"""
	net.evidence(evidence)
	non_evidence_nodes = [n for n in net._nodes if n not in evidence]

	for i in xrange(K+burnin):
		if i >= burnin and sample_fn:
			if sample_fn(i-burnin, net):
				return
		if randomize:
			random.shuffle(non_evidence_nodes)
		for n in non_evidence_nodes:
			net.sample_node(n)

def slow_gibbs_sample(net, evidence, sample_fn, K, burnin):
	"""Does something like Gibbs Sampling, but net not updated until all nodes sampled
	"""
	net.evidence(evidence)
	non_evidence_nodes = [n for n in net._nodes if n not in evidence]

	next_state = net.state_map()

	for i in xrange(K+burnin):
		if i >= burnin and sample_fn:
			if sample_fn(i-burnin, net):
				return
		for n in non_evidence_nodes:
			prev_value = n.get_value()
			next_state[n] = net.sample_node(n)
			net.evidence({n: prev_value})
		net.evidence(next_state)