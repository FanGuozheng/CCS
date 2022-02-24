#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 20:46:36 2022

@author: gz_fan
"""
from typing import Union
import os
from ase import Atoms
import ase.io as io
import json
import numpy as np
import matplotlib.pyplot as plt
from ccs.scripts.atom_json import atom_json, parse_cmdline_args
from ccs.fitting.main import twp_fit as ccs_fit

# train: Li6, Li_cub, Limater, Li5PS4Cl2
task = {"train": "Li3Li6", "gen_geo": False, "fit": True, "plot": True}

# Generate geometries
template = {
    "Li_bcc": "../data/raw_geo/Li_bcc_51_primitive.cif",
    "Li_cub": "../data/raw_geo/Li_cubic_135_primitive.cif",
}
geo_type = "aims"
scal_params = np.linspace(0.95, 1.05, 11)  # 12, 56

# CCS fitting
path = {
    "Li_bcc": "/home/gz_fan/Documents/ML/train/battery/CCS/Li_bcc",
    "Li_cub": "/home/gz_fan/Documents/ML/train/battery/CCS/Li_cub",
    "Li6": "/home/gz_fan/Documents/ML/train/battery/CCS/Li6PSCl",
    "Li3PS9": "/home/gz_fan/Documents/ML/train/battery/CCS/Li3PS9",
    "Li5PS4Cl2": "/home/gz_fan/Documents/ML/train/battery/CCS/Li5PS4Cl2",
    "Li5PS4Cl2_md": "/home/gz_fan/Documents/ML/train/battery/CCS/Li5PS4Cl2_md",
    "Li6PSCl_md": "/home/gz_fan/Documents/ML/train/battery/CCS/Li6PSCl_md",
}

path_dft = {
    "Li_cub": [
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
        "S10/DFT",
    ],
    "Li_bcc": [
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
    "Li6": [
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
        "S10/DFT",
    ],
    "Li3PS9": [
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
        "S10/DFT",
    ],
    "Li5PS4Cl2": [
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
        "S10/DFT",
    ],
    "Li5PS4Cl2_md": [
        "S0/DFT",
        "S1/DFT",
        "S2/DFT",
        "S3/DFT",
        "S4/DFT",
        "S5/DFT",
        "S6/DFT",
    ],
    "Li6PSCl_md": [
        "S0/DFT",
        "S1/DFT",
        "S2/DFT",
        "S3/DFT",
        "S4/DFT",
        "S5/DFT",
        "S6/DFT",
    ],
}
path_dftb = {
    "Li_cub": [
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
        "S10/DFTB",
    ],
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
        "S10/DFTB",
    ],
    "Li6": [
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
        "S10/DFTB",
    ],
    "Li3PS9": [
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
        "S10/DFTB",
    ],
    "Li5PS4Cl2": [
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
        "S10/DFTB",
    ],
    "Li5PS4Cl2_md": [
        "S0/DFTB",
        "S1/DFTB",
        "S2/DFTB",
        "S3/DFTB",
        "S4/DFTB",
        "S5/DFTB",
        "S6/DFTB",
    ],
    "Li6PSCl_md": [
        "S0/DFTB",
        "S1/DFTB",
        "S2/DFTB",
        "S3/DFTB",
        "S4/DFTB",
        "S5/DFTB",
        "S6/DFTB",
    ],
}

inpaths_dft = {
    "Li_bcc": [os.path.join(path["Li_bcc"], ii) for ii in path_dft["Li_bcc"]],
    "Li_cub": [os.path.join(path["Li_cub"], ii) for ii in path_dft["Li_cub"]],
    "Li": [os.path.join(path["Li_bcc"], ii) for ii in path_dft["Li_bcc"]] +
          [os.path.join(path["Li_cub"], ii) for ii in path_dft["Li_cub"]],
    "Li6": [os.path.join(path["Li6"], ii) for ii in path_dft["Li6"]],
    "Li3PS9": [os.path.join(path["Li3PS9"], ii) for ii in path_dft["Li3PS9"]],
    "Li5PS4Cl2": [os.path.join(path["Li5PS4Cl2"], ii) for ii in path_dft["Li5PS4Cl2"]],
    "Li5PS4Cl2_md": [os.path.join(path["Li5PS4Cl2_md"], ii) for ii in path_dft["Li5PS4Cl2_md"]],
    "Li6PSCl_md": [os.path.join(path["Li6PSCl_md"], ii) for ii in path_dft["Li6PSCl_md"]],
    "Li3Li6": [os.path.join(path["Li3PS9"], ii) for ii in path_dft["Li3PS9"]]
    + [os.path.join(path["Li6"], ii) for ii in path_dft["Li6"]],
}
inpaths_dftb = {
    "Li_bcc": [os.path.join(path["Li_bcc"], ii) for ii in path_dftb["Li_bcc"]],
    "Li_cub": [os.path.join(path["Li_cub"], ii) for ii in path_dftb["Li_cub"]],
    "Li": [os.path.join(path["Li_bcc"], ii) for ii in path_dftb["Li_bcc"]] +
          [os.path.join(path["Li_cub"], ii) for ii in path_dftb["Li_cub"]],
    "Li6": [os.path.join(path["Li6"], ii) for ii in path_dftb["Li6"]],
    "Li3PS9": [os.path.join(path["Li3PS9"], ii) for ii in path_dftb["Li3PS9"]],
    "Li5PS4Cl2": [os.path.join(path["Li5PS4Cl2"], ii) for ii in path_dftb["Li5PS4Cl2"]],
    "Li5PS4Cl2_md": [os.path.join(path["Li5PS4Cl2_md"], ii) for ii in path_dftb["Li5PS4Cl2_md"]],
    "Li6PSCl_md": [os.path.join(path["Li6PSCl_md"], ii) for ii in path_dftb["Li6PSCl_md"]],
    "Li3Li6": [os.path.join(path["Li3PS9"], ii) for ii in path_dftb["Li3PS9"]]
    + [os.path.join(path["Li6"], ii) for ii in path_dftb["Li6"]],
}


def scale(
    template, #: Union[list, Geometry],
    scale: list,
    to_path: Union[str, list],
    to_type: str,
):
    geo = io.read(template)
    _in = [
        Atoms(
            positions=geo.positions * ii,
            numbers=geo.numbers,
            cell=geo.cell * ii,
            pbc=True,
        )
        for ii in scale
    ]

    if isinstance(to_path, str):
        to_path = [to_path] * len(scale)

    if to_type == "aims":
        os.system("rm " + to_path[0] + "/geometry.in.*")
        [
            io.write(os.path.join(path, "geometry.in." + str(ii)), iin, format="aims")
            for ii, (iin, path) in enumerate(zip(_in, to_path))
        ]
    elif to_type == "dftb":
        os.system("rm " + to_path[0] + "/geo.gen.*")
        [
            io.write(os.path.join(path, "geo.gen." + str(ii)), iin, format="dftb")
            for ii, (iin, path) in enumerate(zip(_in, to_path))
        ]
    elif to_type == "vasp":
        [
            io.write(os.path.join(path, "POSCAR." + str(ii)), iin, format="vasp")
            for ii, (iin, path) in enumerate(zip(_in, to_path))
        ]

    return _in


if __name__ == "__main__":
    os.system("rm DFT*")

    args = parse_cmdline_args()
    if task["train"] == "Li_bcc":
        args.rcut = 9
        atom_json(args, inpaths_dft[task["train"]], inpaths_dftb[task["train"]])
        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                "Li-Li": {"Rmin": 5.4, "Rcut": 7.4, "Nknots": 10, "Swtype": "rep"}
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }
    if task["train"] == "Li_cub":
        args.rcut = 9
        atom_json(args, inpaths_dft[task["train"]], inpaths_dftb[task["train"]])
        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                "Li-Li": {"Rmin": 5.4, "Rcut": 7.4, "Nknots": 10, "Swtype": "rep"}
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li":  # include Li cubic and bcc
        args.rcut = 9
        atom_json(args, inpaths_dft[task["train"]], inpaths_dftb[task["train"]])

        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                "Li-Li": {"Rmin": 5., "Rcut": 9, "Nknots": 10, "Swtype": "rep"}
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li6":
        if task["gen_geo"]:
            geometries = scale(template, scal_params, "../data/geo", geo_type)
        args.rcut = 8.5
        atom_json(args, inpaths_dft["Li6"], inpaths_dftb["Li6"])

        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                "Li-Li": {"Rmin": 5.9, "Rcut": 7.9, "Nknots": 20, "Swtype": "rep"},
                "Li-P": {"Rmin": 6.5, "Rcut": 8.5, "Nknots": 20, "Swtype": "rep"},
                "Li-S": {"Rmin": 4.1, "Rcut": 6.1, "Nknots": 20, "Swtype": "rep"},
                "Li-Cl": {"Rmin": 6.5, "Rcut": 8.5, "Nknots": 20, "Swtype": "rep"},
                # "Li-Li": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-P": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-S": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-Cl": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "P-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "S-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "Cl-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "P-P": {"Rmin": 5.4, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li3PS9":
        args.rcut = 8.0
        atom_json(args, inpaths_dft["Li3PS9"], inpaths_dftb["Li3PS9"])

        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                # "Li-Li": {"Rmin": 5.0, "Rcut": 7.0, "Nknots": 20, "Swtype": "rep"},
                "Li-P": {"Rmin": 4.7, "Rcut": 7.035, "Nknots": 20, "Swtype": "rep"},
                "Li-S": {"Rmin": 3.9, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-Cl": {"Rmin": 6.5, "Rcut": 8.5, "Nknots": 20, "Swtype": "rep"},
                # "Li-Li": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-P": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-S": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-Cl": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "P-Li": {"Rmin": 4.7, "Rcut": 7.7, "Nknots": 20, "Swtype": "rep"},
                # "S-Li": {"Rmin": 3.9, "Rcut": 6.9, "Nknots": 20, "Swtype": "rep"},
                # "Cl-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "P-P": {"Rmin": 5.4, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
            },
            "Onebody": ["S"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li3Li6":
        if task["gen_geo"]:
            geometries = scale(template, scal_params, "../data/geo", geo_type)
        args.rcut = 8.5
        atom_json(args, inpaths_dft["Li3Li6"], inpaths_dftb["Li3Li6"])

        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                # "Li-Li": {"Rmin": 5.9, "Rcut": 7.9, "Nknots": 20, "Swtype": "rep"},
                "Li-P": {"Rmin": 6.5, "Rcut": 8.5, "Nknots": 20, "Swtype": "rep"},
                "Li-S": {"Rmin": 4.1, "Rcut": 6.1, "Nknots": 20, "Swtype": "rep"},
                "Li-Cl": {"Rmin": 6.5, "Rcut": 8.5, "Nknots": 20, "Swtype": "rep"},
                # "Li-Li": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-P": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-S": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-Cl": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "P-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "S-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "Cl-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                "S-S": {"Rmin": 3.4, "Rcut": 6.4, "Nknots": 20, "Swtype": "rep"},
            },
            "Onebody": ["S"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li5PS4Cl2":
        if task["gen_geo"]:
            geometries = scale(template, scal_params, "../data/geo", geo_type)
        args.rcut = 8.5
        atom_json(args, inpaths_dft["Li5PS4Cl2"], inpaths_dftb["Li5PS4Cl2"])

        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                "Li-Li": {"Rmin": 5.1, "Rcut": 7.9, "Nknots": 20, "Swtype": "rep"},
                "Li-P": {"Rmin": 6.5, "Rcut": 8.5, "Nknots": 20, "Swtype": "rep"},
                "Li-S": {"Rmin": 4.1, "Rcut": 6.1, "Nknots": 20, "Swtype": "rep"},
                "Li-Cl": {"Rmin": 6.5, "Rcut": 8.5, "Nknots": 20, "Swtype": "rep"},
                # "Li-Li": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-P": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-S": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "Li-Cl": {"Rmin": 3.0, "Rcut": 6.0, "Nknots": 20, "Swtype": "rep"},
                # "P-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "S-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "Cl-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "P-P": {"Rmin": 5.4, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "P-S": {"Rcut": 6.0, "Resolution": 0.05, "Swtype": "rep"},
                # "P-Cl": {"Rcut": 6.0, "Resolution": 0.05, "Swtype": "rep"},
                # "S-S": {"Rcut": 6.0, "Resolution": 0.05, "Swtype": "rep"},
                # "S-Cl": {"Rcut": 6.0, "Resolution": 0.05, "Swtype": "rep"},
                # "Cl-Cl": {"Rcut": 6.0, "Resolution": 0.05, "Swtype": "rep"}
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li5PS4Cl2_md":
        if task["gen_geo"]:
            geometries = scale(template, scal_params, "../data/geo", geo_type)
        args.rcut = 8.5
        atom_json(args, inpaths_dft["Li5PS4Cl2_md"], inpaths_dftb["Li5PS4Cl2_md"])

        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                "Li-Li": {"Rmin": 1.1, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                "Li-P": {"Rmin": 2.1, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                "Li-S": {"Rmin": 1.0, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                "Li-Cl": {"Rmin": 0.9, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                # "P-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "S-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "Cl-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                "P-P": {"Rmin": 7.7, "Rcut": 10.0, "Nknots": 5, "Swtype": "rep"},
                "P-S": {"Rmin": 1.8, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                "P-Cl": {"Rmin": 4.6, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"},
                "S-S": {"Rmin": 4.0, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"},
                "S-Cl": {"Rmin": 3.6, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"},
                "Cl-Cl": {"Rmin": 4.1, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"}
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li6PSCl_md":
        if task["gen_geo"]:
            geometries = scale(template, scal_params, "../data/geo", geo_type)
        args.rcut = 8.5
        atom_json(args, inpaths_dft["Li6PSCl_md"], inpaths_dftb["Li6PSCl_md"])

        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                "Li-Li": {"Rmin": 3.0, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                "Li-P": {"Rmin": 3.0, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                "Li-S": {"Rmin": 3.0, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                "Li-Cl": {"Rmin": 3.0, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                # "P-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "S-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                # "Cl-Li": {"Rmin": 3.0, "Rcut": 9.0, "Nknots": 20, "Swtype": "rep"},
                "P-P": {"Rmin": 10.0, "Rcut": 14.0, "Nknots": 10, "Swtype": "rep"},
                "P-S": {"Rmin": 3.0, "Rcut": 8.0, "Nknots": 20, "Swtype": "rep"},
                "P-Cl": {"Rmin": 5.0, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"},
                "S-S": {"Rmin": 4.0, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"},
                "S-Cl": {"Rmin": 4.0, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"},
                "Cl-Cl": {"Rmin": 4.0, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"}
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }

    if task["fit"]:
        with open("input.json", "w") as f:
            json.dump(input, f, indent=8)
        ccs_fit("input.json")
        error = np.loadtxt("error.out")

    if task["plot"]:
        EDFT = error[:, 0]
        EDFTB = error[:, 1]
        x_line = [min(EDFT), max(EDFT)]
        plt.plot(x_line, x_line, "-")
        plt.xlim(x_line)
        plt.ylim(x_line)
        plt.xlabel("Target repulsive energy (eV)")
        plt.ylabel("Predicted repulsive energy (eV)")
        plt.scatter(EDFT, EDFTB)
        plt.show()
