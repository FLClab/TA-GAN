P_STED = 150.0e-3
P_EX = 2.0e-6
PDT = 10.0e-6

LASER_EX = {"lambda_" : 635e-9}
LASER_STED = {"lambda_" : 750e-9, "zero_residual" : 0.005, "anti_stoke": False}
DETECTOR = {"noise" : True, "background" : 2 / PDT}
OBJECTIVE = {
    "transmission" : {488: 0.84, 535: 0.85, 550: 0.86, 585: 0.85, 575: 0.85, 635: 0.85, 690: 0.85, 750: 0.85}
}
FLUO = {
    "lambda_": 6.9e-7,
    "qy": 0.65,
    "sigma_abs": {
        635: 2.14e-20,
        750: 3.5e-25
    },
    "sigma_ste": {
        750: 3.0e-22
    },
    "tau": 3.5e-9,
    "tau_vib": 1e-12,
    "tau_tri": 0.0000012,
    "k0": 0,
    "k1": 1.3e-15,
    "b": 1.6,
    "triplet_dynamics_frac": 0
}
