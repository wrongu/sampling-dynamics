import random

def get_non_evidence_nodes(net, ev):
	return [n for n in net._nodes if n not in ev]

def gibbs_sample(net, evidence, sample_fn, K, burnin, randomize=True):
	"""Does Gibbs sampling on the net. After burnin, calls sample_fn(i, net) after each sweep

	optionally, sample_fn may return True to halt the rest of the sampling
	"""
	net.evidence(evidence)
	non_evidence_nodes = get_non_evidence_nodes(net, evidence)

	for i in xrange(K+burnin):
		if i >= burnin and sample_fn:
			if sample_fn(i-burnin, net):
				return
		if randomize:
			random.shuffle(non_evidence_nodes)
		for n in non_evidence_nodes:
			net.sample_node(n)

def gibbs_sample_dynamic_evidence(net, evidence_gen, sample_fn, K, burnin, randomize=True):
	"""Same interface as gibbs_sample(), but at each iteration the generator evidence_gen
	is queried for the next evidence state
	"""
	for i in xrange(K+burnin):
		ev = next(evidence_gen)
		net.evidence(ev)
		non_evidence_nodes = get_non_evidence_nodes(net, ev)
		if i >= burnin and sample_fn:
			if sample_fn(i-burnin, net):
				return
		if randomize:
			random.shuffle(non_evidence_nodes)
		for n in non_evidence_nodes:
			net.sample_node(n)