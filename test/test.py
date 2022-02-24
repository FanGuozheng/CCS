#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 20:46:36 2022

@author: gz_fan
"""
import os
import json
import numpy as np
from ccs.scripts.atom_json import atom_json, parse_cmdline_args
from ccs.fitting.main import twp_fit as ccs_fit

task = {"train": "Li_bcc"}


# CCS fitting, change the absolute path when fitting!!!
path = {
    "Li_bcc": "/home/gz_fan/Documents/ML/train/battery/CCS/CCS/test/Li_bcc"}

path_dft = {
    "Li_bcc": [
        "S0/DFT",
        "S1/DFT",
        "S2/DFT",
        "S3/DFT",
        "S4/DFT",
        "S5/DFT",
        "S6/DFT",
        "S7/DFT",
        "S8/DFT",
        "S9/DFT",
    ],
}
path_dftb = {
    "Li_bcc": [
        "S0/DFTB",
        "S1/DFTB",
        "S2/DFTB",
        "S3/DFTB",
        "S4/DFTB",
        "S5/DFTB",
        "S6/DFTB",
        "S7/DFTB",
        "S8/DFTB",
        "S9/DFTB",
    ],
}

inpaths_dft = {"Li_bcc": [os.path.join(path["Li_bcc"], ii) for ii in path_dft["Li_bcc"]]}
inpaths_dftb = {"Li_bcc": [os.path.join(path["Li_bcc"], ii) for ii in path_dftb["Li_bcc"]]}


if __name__ == "__main__":
    os.system("rm DFT*")

    args = parse_cmdline_args()
    if task["train"] == "Li_bcc":

        args.rcut = 9.0  # BOHR
        atom_json(args, inpaths_dft[task["train"]], inpaths_dftb[task["train"]])

        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {  # ONLY fit first neighbouring atoms
                "Li-Li": {"Rmin": 5.4, "Rcut": 7.4, "Nknots": 10, "Swtype": "rep"}
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }

        # FIT REPULSIVE
        with open("input.json", "w") as f:
            json.dump(input, f, indent=8)
        ccs_fit("input.json")
        error = np.loadtxt("error.out")
