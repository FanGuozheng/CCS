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
task = {"train": "Li5PS4Cl2PSCl", "gen_geo": False, "fit": True, "plot": True}

# Generate geometries
template = {
    "Li_bcc": "../data/raw_geo/Li_bcc_51_primitive.cif",
    "Li_cub": "../data/raw_geo/Li_cubic_135_primitive.cif",
}
geo_type = "aims"
scal_params = np.linspace(0.95, 1.05, 11)  # 12, 56

# CCS fitting
_path = os.getcwd()
path = {
    "Li_bcc": os.path.join(_path, "Li_bcc"),
    "Li_cub": os.path.join(_path, "Li_cub"),
    "LiP": os.path.join(_path, "LiP"),
    "LiS": os.path.join(_path, "LiS"),
    "LiCl": os.path.join(_path, "LiCl"),
    "Li6": os.path.join(_path, "Li6PSCl"),
    "Li3PS9": os.path.join(_path, "Li3PS9"),
    "Li5PS4Cl2": os.path.join(_path, "Li5PS4Cl2"),
    "Li5PS4Cl2_trans": os.path.join(_path, "Li5PS4Cl2_trans"),
}
volrange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
path_dft = {
    "Li_cub": [f"S{ii}/DFT" for ii in volrange],
    "Li_bcc": [f"S{ii}/DFT" for ii in volrange],
    "LiP": [f"S{ii}/DFT" for ii in volrange],
    "LiS": [f"S{ii}/DFT" for ii in volrange],
    "LiCl": [f"S{ii}/DFT" for ii in volrange],
    "Li6": [f"S{ii}/DFT" for ii in volrange],
    "Li3PS9": [f"S{ii}/DFT" for ii in volrange],
    "Li5PS4Cl2": [f"S{ii}/DFT" for ii in volrange],
    "Li5PS4Cl2_trans": [f"S{ii}/DFT" for ii in volrange],
}

path_dftb = {
    "Li_cub": [f"S{ii}/DFTB" for ii in volrange],
    "Li_bcc": [f"S{ii}/DFTB" for ii in volrange],
    "LiP": [f"S{ii}/DFTB" for ii in volrange],
    "LiS": [f"S{ii}/DFTB" for ii in volrange],
    "LiCl": [f"S{ii}/DFTB" for ii in volrange],
    "Li6": [f"S{ii}/DFTB" for ii in volrange],
    "Li3PS9": [f"S{ii}/DFTB" for ii in volrange],
    "Li5PS4Cl2": [f"S{ii}/DFTB" for ii in volrange],
    "Li5PS4Cl2_trans": [f"S{ii}/DFTB" for ii in volrange],
}

inpaths_dft = {
    "Li_bcc": [os.path.join(path["Li_bcc"], ii) for ii in path_dft["Li_bcc"]],
    "Li_cub": [os.path.join(path["Li_cub"], ii) for ii in path_dft["Li_cub"]],
    "Li": [os.path.join(path["Li_bcc"], ii) for ii in path_dft["Li_bcc"]] +
          [os.path.join(path["Li_cub"], ii) for ii in path_dft["Li_cub"]],
    "LiP": [os.path.join(path["LiP"], ii) for ii in path_dft["LiP"]],
    "LiS": [os.path.join(path["LiS"], ii) for ii in path_dft["LiS"]],
    "LiCl": [os.path.join(path["LiCl"], ii) for ii in path_dft["LiCl"]],
    "Li2ele": [os.path.join(path["LiP"], ii) for ii in path_dft["LiP"]] +
              [os.path.join(path["LiS"], ii) for ii in path_dft["LiS"]] +
              [os.path.join(path["LiCl"], ii) for ii in path_dft["LiCl"]],
    "Li6": [os.path.join(path["Li6"], ii) for ii in path_dft["Li6"]],
    "Li3PS9": [os.path.join(path["Li3PS9"], ii) for ii in path_dft["Li3PS9"]],
    "Li5PS4Cl2": [os.path.join(path["Li5PS4Cl2"], ii) for ii in path_dft["Li5PS4Cl2"]],
    "Li3Li6": [os.path.join(path["Li3PS9"], ii) for ii in path_dft["Li3PS9"]]
    + [os.path.join(path["Li6"], ii) for ii in path_dft["Li6"]],
    "Li5PS4Cl2_trans": [os.path.join(path["Li5PS4Cl2_trans"], ii) for ii in path_dft["Li5PS4Cl2_trans"]],
    "Li5PS4Cl2PSCl": [os.path.join(path["Li5PS4Cl2"], ii) for ii in path_dft["Li5PS4Cl2"]] +
                     [os.path.join(path["LiP"], ii) for ii in path_dft["LiP"]] +
                     [os.path.join(path["LiS"], ii) for ii in path_dft["LiS"]] +
                     [os.path.join(path["LiCl"], ii) for ii in path_dft["LiCl"]],
    "Li3Li5Li6": [os.path.join(path["Li3PS9"], ii) for ii in path_dft["Li3PS9"]] +
                 [os.path.join(path["Li5PS4Cl2_trans"], ii) for ii in path_dft["Li5PS4Cl2_trans"]] +
                 [os.path.join(path["Li6"], ii) for ii in path_dft["Li6"]]
                 # + [os.path.join(path["LiCl"], ii) for ii in path_dft["LiCl"]]
    ,
}
inpaths_dftb = {
    "Li_bcc": [os.path.join(path["Li_bcc"], ii) for ii in path_dftb["Li_bcc"]],
    "Li_cub": [os.path.join(path["Li_cub"], ii) for ii in path_dftb["Li_cub"]],
    "Li": [os.path.join(path["Li_bcc"], ii) for ii in path_dftb["Li_bcc"]] +
          [os.path.join(path["Li_cub"], ii) for ii in path_dftb["Li_cub"]],
    "LiP": [os.path.join(path["LiP"], ii) for ii in path_dftb["LiP"]],
    "LiS": [os.path.join(path["LiS"], ii) for ii in path_dftb["LiS"]],
    "LiCl": [os.path.join(path["LiCl"], ii) for ii in path_dftb["LiCl"]],
    "Li2ele": [os.path.join(path["LiP"], ii) for ii in path_dftb["LiP"]] +
              [os.path.join(path["LiS"], ii) for ii in path_dftb["LiS"]] +
              [os.path.join(path["LiCl"], ii) for ii in path_dftb["LiCl"]],
    "Li6": [os.path.join(path["Li6"], ii) for ii in path_dftb["Li6"]],
    "Li3PS9": [os.path.join(path["Li3PS9"], ii) for ii in path_dftb["Li3PS9"]],
    "Li5PS4Cl2": [os.path.join(path["Li5PS4Cl2"], ii) for ii in path_dftb["Li5PS4Cl2"]],
    "Li5PS4Cl2_trans": [os.path.join(path["Li5PS4Cl2_trans"], ii) for ii in path_dftb["Li5PS4Cl2_trans"]],
    "Li5PS4Cl2PSCl": [os.path.join(path["Li5PS4Cl2"], ii) for ii in path_dftb["Li5PS4Cl2"]] +
                     [os.path.join(path["LiP"], ii) for ii in path_dftb["LiP"]] +
                     [os.path.join(path["LiS"], ii) for ii in path_dftb["LiS"]] +
                     [os.path.join(path["LiCl"], ii) for ii in path_dftb["LiCl"]],
    "Li3Li5Li6": [os.path.join(path["Li3PS9"], ii) for ii in path_dftb["Li3PS9"]] +
                 [os.path.join(path["Li5PS4Cl2_trans"], ii) for ii in path_dftb["Li5PS4Cl2_trans"]] +
                 [os.path.join(path["Li6"], ii) for ii in path_dftb["Li6"]]
                 # + [os.path.join(path["LiCl"], ii) for ii in path_dftb["LiCl"]]
    ,
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
                "Li-Li": {"Rmin": 4., "Rcut": 6, "Nknots": 10, "Swtype": "rep"}
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }
    if task["train"] == "Li2ele":  # include Li cubic and bcc
        args.rcut = 9
        atom_json(args, inpaths_dft[task["train"]], inpaths_dftb[task["train"]])

        input = {
            "General": {"interface": "DFTB", "smooth": True},
            "Twobody": {
                "Li-Li": {"Rmin": 3.9, "Rcut": 5.5, "Nknots": 8, "Swtype": "rep"},
                "Li-Cl": {"Rmin": 3.8, "Rcut": 5.5, "Nknots": 8, "Swtype": "rep"},
                "Li-S": {"Rmin": 3.8, "Rcut": 5.5, "Nknots": 8, "Swtype": "rep"},
                "Li-P": {"Rmin": 3.8, "Rcut": 5.5, "Nknots": 8, "Swtype": "rep"},
            },
            "Onebody": ["P", "S", "Cl"],
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
                "S-S": {"Rmin": 3.4, "Rcut": 6.4, "Nknots": 20, "Swtype": "rep"},
            },
            "Onebody": ["S"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li3Li5Li6":
        args.rcut = 8.5
        atom_json(args, inpaths_dft["Li3Li5Li6"], inpaths_dftb["Li3Li5Li6"])

        input = {
            "General": {"interface": "DFTB"},
            "Twobody": {
                "Li-Li": {"Rmin": 3.6, "Rcut": 6.0, "Nknots": 8, "Swtype": "rep"},
                "Li-P": {"Rmin": 3.8, "Rcut": 7.4, "Nknots": 8, "Swtype": "rep"},
                "Li-S": {"Rmin": 3.6, "Rcut": 6.0, "Nknots": 8, "Swtype": "rep"},
                "Li-Cl": {"Rmin": 3.6, "Rcut": 5.6, "Nknots": 8, "Swtype": "rep"},
                # "S-S": {"Rmin": 3.8, "Rcut": 5.0, "Nknots": 8, "Swtype": "rep"},
            },
            "Onebody": ["Li", "S", "Cl"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li5PS4Cl2_trans":
        args.rcut = 8.5
        atom_json(args, inpaths_dft["Li5PS4Cl2_trans"], inpaths_dftb["Li5PS4Cl2_trans"])

        input = {
            "General": {"interface": "DFTB"},
            "Twobody": {
                "Li-Li": {"Rmin": 3.6, "Rcut": 6, "Nknots": 10, "Swtype": "rep"},
                "Li-P": {"Rmin": 3.6, "Rcut": 7.4, "Nknots": 10, "Swtype": "rep"},
                "Li-S": {"Rmin": 3.4, "Rcut": 5.8, "Nknots": 10, "Swtype": "rep"},
                "Li-Cl": {"Rmin": 3.4, "Rcut": 5.8, "Nknots": 10, "Swtype": "rep"},
                # "S-S": {"Rmin": 3.8, "Rcut": 5.0, "Nknots": 8, "Swtype": "rep"},
            },
            "Onebody": ["Li"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li5PS4Cl2PSCl":
        args.rcut = 8.5
        atom_json(args, inpaths_dft["Li5PS4Cl2PSCl"], inpaths_dftb["Li5PS4Cl2PSCl"])

        input = {
            "General": {"interface": "DFTB"},
            "Twobody": {
                "Li-Li": {"Rmin": 4.0, "Rcut": 6.0, "Nknots": 10, "Swtype": "rep"},
                "Li-P": {"Rmin": 3.7, "Rcut": 6.0, "Nknots": 10, "Swtype": "rep"},
                "Li-S": {"Rmin": 3.6, "Rcut": 5.4, "Nknots": 10, "Swtype": "rep"},
                "Li-Cl": {"Rmin": 3.6, "Rcut": 5.4, "Nknots": 10, "Swtype": "rep"},
                "P-P": {"Rmin": 6.0, "Rcut": 8.5, "Nknots": 10, "Swtype": "rep"},
                "P-S": {"Rmin": 3.6, "Rcut": 5.5, "Nknots": 10, "Swtype": "rep"},
                "P-Cl": {"Rmin": 7.0, "Rcut": 9.0, "Nknots": 10, "Swtype": "rep"},
                "S-S": {"Rmin": 5.5, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"},
                "S-Cl": {"Rmin": 6.5, "Rcut": 8.5, "Nknots": 10, "Swtype": "rep"},
                "Cl-Cl": {"Rmin": 5.2, "Rcut": 8.0, "Nknots": 10, "Swtype": "rep"},
            },
            "Onebody": ["Li", "P", "S", "Cl"],
            "Reference": "structures.json",
        }

    if task["train"] == "Li5PS4Cl2":
        if task["gen_geo"]:
            geometries = scale(template, scal_params, "../data/geo", geo_type)
        args.rcut = 8.5
        atom_json(args, inpaths_dft["Li5PS4Cl2"], inpaths_dftb["Li5PS4Cl2"])
        if task["train"] == "Li5PS4Cl2":
            args.rcut = 8.5
            atom_json(args, inpaths_dft["Li5PS4Cl2"], inpaths_dftb["Li5PS4Cl2"])
            input = {
                "General": {"interface": "DFTB", "smooth": True},
                "Twobody": {
                    "Li-Li": {"Rmin": 5.1, "Rcut": 7.9, "Nknots": 10, "Swtype": "rep"},
                    "Li-P": {"Rmin": 4.0, "Rcut": 6.0, "Nknots": 10, "Swtype": "rep"},
                    "Li-S": {"Rmin": 4.0, "Rcut": 6.0, "Nknots": 10, "Swtype": "rep"},
                    "Li-Cl": {"Rmin": 4.0, "Rcut": 6.0, "Nknots": 10, "Swtype": "rep"},
                    "P-P": {"Rmin": 4.0, "Rcut": 6.0, "Nknots": 10, "Swtype": "rep"},
                    "P-S": {"Rmin": 3.8, "Rcut": 4.2, "Nknots": 10, "Swtype": "rep"},
                    "P-Cl": {"Rmin": 4.0, "Rcut": 6.0, "Nknots": 10, "Swtype": "rep"},
                    "S-S": {"Rmin": 4.0, "Rcut": 6.0, "Nknots": 10, "Swtype": "rep"},
                    "S-Cl": {"Rmin": 6.4, "Rcut": 9.0, "Nknots": 10, "Swtype": "rep"},
                    "Cl-Cl": {"Rmin": 4.0, "Rcut": 6.0, "Nknots": 10, "Swtype": "rep"},
                },
                "Onebody": ["P"],
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

        edft = np.loadtxt('DFT.txt')
        edftb = np.loadtxt('DFTB.txt')
        edftb += EDFTB
        plt.plot(np.linspace(1, 11, 11), edft[:11] - min(edft[:11]), label='DFT')
        plt.plot(np.linspace(1, 11, 11), edftb[:11] - min(edftb[:11]), 'rx')
        plt.title('Li5PS4Cl2')
        plt.show()
        plt.plot(np.linspace(1, 11, 11), edft[11:22] - min(edft[11:22]), label='DFT')
        plt.plot(np.linspace(1, 11, 11), edftb[11:22] - min(edftb[11:22]), 'rx')
        plt.show()
        plt.plot(np.linspace(1, 11, 11), edft[22:33] - min(edft[22:33]), label='DFT')
        plt.plot(np.linspace(1, 11, 11), edftb[22:33] - min(edftb[22:33]), 'rx')
        plt.show()
        plt.plot(np.linspace(1, 11, 11), edft[33:44] - min(edft[33:44]), label='DFT')
        plt.plot(np.linspace(1, 11, 11), edftb[33:44] - min(edftb[33:44]), 'rx')
        plt.show()
