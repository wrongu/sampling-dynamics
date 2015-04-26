Overview of codebase:
---------------------

* graphs.py: basic Graph and DiGraph classes

* graphical_models.py: currently supports on Bayesian Networks with Discrete variables. The `BayesNet` class is a subclass of `DiGraph`

* visualize.py: using networkx_interface.py and the graphviz module, makes diagrams of BayesNets

* sampling.py: defines gibbs sampling functions

* counting.py: useful combinatorics functions and some sampling procedures

* util.py: other helper functions (e.g. `load_or_run()` which wraps precomputing/saving/loading numpy arrays)

* models.py: functions to create special cases of Graphs or BayesNets

* generators.py: create evidence-generators for sampling

Running scripts:
----------------

Because of the way packages are set up here, from the project directory run `python -m scripts.script_name` without `.py`. For example:

	$ python -m scripts.make_mixing_time_movie
