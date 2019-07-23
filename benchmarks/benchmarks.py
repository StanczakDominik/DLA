# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from DLA import DLA

class SequentialIterationSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    timeout=1000
    def time_sequential(self):
        d = DLA.main_single(1, gotosize=[])

class ParallelIterationSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    timeout=1000
    def time_sequential_then_bunch(self):
        d = DLA.main_single(1, gotosize=[1e4])
