import numpy
import multiprocessing

from tqdm import trange

class Experiment():
    """
    Implements an `Experiment` object that allows to store `Microscope` configurations,
    their respective `Datamap`, and their respective acquisition parameters.

    The `Experiment` object implements `acquire` and `acquire_all` methods. The
    `acquire_all` method allows multiprocessing. The `acquire_all` method should
    called within a `if __name__ == "__main__"` if multiprocessing is desired.
    """
    def __init__(self):
        self.microscopes = {}
        self.datamaps = {}
        self.params = {}

    def add(self, name, microscope, datamap, params):
        """
        Adds a microscope and its corresponding datamap to the experiment

        :param name: A `str` of the name of the microscope
        :param microscope: A `Microscope` object
        :param datamap: A `Datamap` object
        :param datamap: A `dict` of the parameters
        """
        self.microscopes[name] = microscope
        self.datamaps[name] = datamap
        self.params[name] = params

    def acquire(self, name, num_acquisition, bleach=True, verbose=False, seed=None):
        """
        Acquires from a specific microscope

        :param name: A `str` of the name of the microscope
        :param num_acquisition: An `int` of the number of acquired frames
        :param bleach: A `bool` whether to bleach the sample

        :returns : A `str` of the name of the microscope
                   A `dict` of the history
        """
        def trange_(x, *args, **kwargs):
            if verbose:
                return trange(x, *args, **kwargs)
            else:
                return range(x)
        history = {
            "acquisition" : numpy.zeros((num_acquisition, *self.datamaps[name].sub_datamaps_dict["base"][self.datamaps[name].roi].shape), dtype=numpy.uint16),
            "datamap" : numpy.zeros((num_acquisition, *self.datamaps[name].sub_datamaps_dict["base"][self.datamaps[name].roi].shape)),
            # "whole-datamap" : numpy.zeros((num_acquisition, *self.datamaps[name].sub_datamaps_dict["base"].shape)),
            "bleached" : numpy.zeros((num_acquisition, *self.datamaps[name].sub_datamaps_dict["base"][self.datamaps[name].roi].shape)),
            "other" : numpy.ones((num_acquisition, *self.datamaps[name].sub_datamaps_dict["base"][self.datamaps[name].roi].shape))
        }
        # if verbose: print("[----] Acquisition started! {}".format(name))
        for i in trange_(num_acquisition, leave=False, desc="[----] Acquire"):
            history["datamap"][i] = self.datamaps[name].whole_datamap[self.datamaps[name].roi]
            # history["whole-datamap"][i] = self.datamaps[name].whole_datamap
            acquisition, bleached, other = self.microscopes[name].get_signal_and_bleach(self.datamaps[name], self.datamaps[name].pixelsize, **self.params[name],
                                                                                bleach=bleach, update=True, seed=seed)
            history["acquisition"][i] = acquisition
            history["bleached"][i] = bleached["base"][self.datamaps[name].roi]
            if isinstance(other, numpy.ndarray):
                history["other"][i] = other
        # if verbose: print("[----] Acquisition done! {}".format(name))
        return name, history

    def acquire_all(self, num_acquisition, bleach=True, processes=0, verbose=False, seed=None):
        """
        Acquires from all microsopes. This

        :param experiment: An `Experiment` object from which to acquire all
        :param num_acquisition: An `int` of the number of acquired frames
        :param bleach: A `bool` whether to bleach the sample
        :param processes: An `int` of the number of processes to use in `multiprocessing`

        :returns : A `dict` of the history for each microscope
        """
        history = {}
        if processes:
            def log_callback(data):
                name, hist = data
                history[name] = hist
            pool = multiprocessing.Pool(processes=processes)
            calls = [pool.apply_async(self.acquire, kwds={"name":name, "num_acquisition":num_acquisition, "bleach":bleach, "verbose":verbose, "seed":seed}, callback=log_callback)
                        for name in self.microscopes.keys()]
            pool.close()
            pool.join()
            for c in calls: c.get()
        else:
            for name in self.microscopes.keys():
                name, out = self.acquire(name=name, num_acquisition=num_acquisition, bleach=bleach, verbose=verbose, seed=seed)
                history[name] = out
        return history
