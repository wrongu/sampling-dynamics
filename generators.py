import random

def alternator(states):
	while True:
		for s in states:
			yield s

def rand(states):
	while True:
		yield random.choice(states)

def constant(state):
	while True:
		yield state

def initial(state):
	yield state
	while True:
		yield {}

def zero_evidence():
	while True:
		yield {}