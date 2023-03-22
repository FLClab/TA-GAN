
import numpy
import random
import time
import sys
import os
import tifffile
import io
import pickle
import matplotlib
import yaml

from matplotlib import pyplot
from PIL import Image
from tqdm.auto import trange

from pysted import base, utils, raster, bleach_funcs
from pysted import exp_data_gen as dg

import defaults
import experiment

def create_microscope(**kwargs):

    # Creates a default datamap
    delta = 1
    num_mol = 2
    molecules_disposition = numpy.zeros((50, 50))
    for j in range(1,4):
        for i in range(1,4):
            molecules_disposition[
                j * molecules_disposition.shape[0]//4 - delta : j * molecules_disposition.shape[0]//4 + delta + 1,
                i * molecules_disposition.shape[1]//4 - delta : i * molecules_disposition.shape[1]//4 + delta + 1] = num_mol

    # Extracts params
    laser_ex_params = kwargs.get("laser_ex", defaults.LASER_EX)
    laser_sted_params = kwargs.get("laser_sted", defaults.LASER_STED)
    detector_params = kwargs.get("detector", defaults.DETECTOR)
    objective_params = kwargs.get("objective", defaults.OBJECTIVE)
    fluo_params = kwargs.get("fluo", defaults.FLUO)
    datamap_params = kwargs.get("datamap", {
        "whole_datamap" : molecules_disposition,
        "datamap_pixelsize" : 20e-9
    })
    imaging_params = kwargs.get("imaging", {
        "p_sted" : defaults.P_STED,
        "p_ex" : defaults.P_EX,
        "pdt" : defaults.PDT
    })

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(**laser_ex_params)
    laser_sted = base.DonutBeam(**laser_sted_params)
    detector = base.Detector(**detector_params)
    objective = base.Objective(**objective_params)
    fluo = base.Fluorescence(**fluo_params)
    datamap = base.Datamap(**datamap_params)

    microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
    i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)

    datamap.set_roi(i_ex, "max")

    return microscope, datamap, imaging_params

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing", action="store_true",
                        help="Wheter a dry-run is run")
    args = parser.parse_args()

    SAVEFOLDER = "./data/dataset"
    seed = 0
    if args.testing:
        SAVEFOLDER = os.path.join(SAVEFOLDER, "testing")
        seed = 424242

    NUM_EXAMPLES = 200
    n_molecs_in_domain1, min_dist1 = 100, 100

    for n_domains in trange(1, 7, desc="Nanodomains"):
        for _ in trange(NUM_EXAMPLES, desc="Examples", leave=False):
            synapse = dg.Synapse(5, mode="mushroom", seed=seed)
            synapse.add_nanodomains(n_domains, min_dist_nm=min_dist1, n_molecs_in_domain=n_molecs_in_domain1, valid_thickness=7, seed=seed)
            synapse.rotate_and_translate(rot_angle=None, translate=True)

            sted_microscope, datamap, sted_params = create_microscope(
                datamap = {
                    "whole_datamap" : synapse.frame,
                    "datamap_pixelsize" : 20e-9
                },
                imaging = {"p_ex" : defaults.P_EX, "p_sted" : defaults.P_STED, "pdt" : defaults.PDT}
            )
            conf_microscope, _, conf_params = create_microscope(
                datamap = {
                    "whole_datamap" : synapse.frame,
                    "datamap_pixelsize" : 20e-9
                },
                imaging = {"p_ex" : defaults.P_EX, "p_sted" : 0., "pdt" : defaults.PDT}
            )

            if seed == 0:
                effective = sted_microscope.get_effective(20e-9, defaults.P_EX, defaults.P_STED)
                numpy.save(os.path.join(SAVEFOLDER, "effective-STED.npy"), effective)
                effective = sted_microscope.get_effective(20e-9, defaults.P_EX, 0.)
                numpy.save(os.path.join(SAVEFOLDER, "effective-CONF.npy"), effective)

            exp = experiment.Experiment()
            exp.add("CONF", conf_microscope, datamap, conf_params)
            exp.add("STED", sted_microscope, datamap, sted_params)

            history = exp.acquire_all(processes=2, num_acquisition=1, bleach=False, verbose=False, seed=seed)

            # fig, axes = pyplot.subplots(1,2, figsize=(6,3))
            # im = axes[0].imshow(history["CONF"]["acquisition"][-1], cmap="hot", vmax=history["CONF"]["acquisition"][-1].max() * 1.5)
            # pyplot.colorbar(im, ax=axes[0])
            # im = axes[1].imshow(history["STED"]["acquisition"][-1], cmap="hot")
            # pyplot.colorbar(im, ax=axes[1])
            # pyplot.show()

            os.makedirs(SAVEFOLDER, exist_ok=True)
            for key, values in history.items():
                tifffile.imwrite(
                    os.path.join(SAVEFOLDER, f"{seed:06d}-{n_domains}-{key}.tif"),
                    values["acquisition"][-1].astype(numpy.uint16)
                )
            numpy.save(
                os.path.join(SAVEFOLDER, f"{seed:06d}-{n_domains}-POSITIONS.npy"),
                synapse.nanodomains_coords
            )

            seed += 1
