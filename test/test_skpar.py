#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 22:08:18 2022

@author: gz_fan
"""
from typing import Union
import os
import re

import ase.io.aims
import torch
from ase import Atoms
import ase.io as io
import numpy as np
import matplotlib.pyplot as plt

from tbmalt import Geometry
from tbmalt.io.dataset import GeometryTo


task = {"gen_geo": False, "gen_band": False, "get_dftb_energy": False, "plot": False,
        "plot_ccs_error": False, "distance_distribution": True}


# plot
energies = {
    "bcc": np.array(
        [
            -204.68549931,
            -204.69435132,
            -204.70103732,
            -204.70568191,
            -204.70841043,
            -204.70935564,
            -204.70865075,
            -204.70642134,
            -204.70277470,
            -204.69780453,
            -204.69160456,
        ]
    )
}
energy = energies["bcc"]
aims_energy = [
    -622.39875445,
    -622.42540082,
    -622.44562178,
    -622.45981541,
    -622.46836077,
    -622.47163706,
    -622.47002478,
    -622.46388065,
    -622.45351286,
    -622.43921115,
    -622.42125445,
    -622.39990568,
]
aims_li6 = np.array(
    [
        -77204.69755699,
        -77205.05283465,
        -77205.31584302,
        -77205.49472604,
        -77205.59702887,
        -77205.62971999,
        -77205.59918423,
        -77205.51129562,
        -77205.37147146,
        -77205.1846876,
        -77204.95556232,
    ]
)
aims_cub = np.array(
    [
        -205.89991774,
        -205.89189785,
        -205.88228386,
        -205.87120413,
        -205.85881538,
        -205.84529833,
        -205.83087933,
        -205.81590623,
        -205.80109471,
        -205.78940324,
        -205.77397351,
    ]
)


def scale(
    template: Union[list, Geometry],
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

    return _in


def get_energy_single(detail, unit='H'):
    text = "".join(open(detail, "r").readlines())
    E_tot_ = re.search(
        "(?<=Total energy:).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
    ).group(0)
    E_tot = re.findall(r"[-+]?\d*\.\d+", E_tot_)

    E_rep_ = re.search(
        "(?<=Repulsive energy:).+(?=\n)", text, flags=re.DOTALL | re.MULTILINE
    ).group(0)
    E_rep = re.findall(r"[-+]?\d*\.\d+", E_rep_)

    if unit == 'H':
        return float(E_tot[0]), float(E_rep[0])
    else:
        return float(E_tot[1]), float(E_rep[1])


def get_energy_dft_single(aims, unit='H'):
    """Read FHI-aims output."""
    text = "".join(open(aims, "r").readlines())
    E_tot_ = re.findall("^.*\| Total energy                  :.*$", text, re.MULTILINE)[-1]
    E_tot = re.findall(r"[-+]?\d*\.\d+", E_tot_)

    return float(E_tot[0]) if unit == 'H' else float(E_tot[1])


def get_dft_energy(path, scal_params, unit='H'):
    energy = []
    for num in range(len(scal_params)):
        file = os.path.join(path, 'aims.out.' + str(num))
        energy.append(get_energy_dft_single(file, unit))
    return energy


def get_dftb_energy(path, scal_params, unit='H'):
    rep, tot = [], []
    for ii in range(len(scal_params)):
        detail = path + "/detailed.out." + str(ii)
        try:
            it, ir = get_energy_single(detail, unit)
        except:
            it, ir = 0, 0
        rep.append(ir)
        tot.append(it)
    return tot, rep


if __name__ == "__main__":
    scal_params = np.linspace(0.95, 1.05, 11)  # 11, 51

    if task["gen_geo"]:
        # generate geometries
        # template = "./data/raw_geo/Li_cubic_135_primitive.cif"
        # geo_type = "dftb"
        # path = ("./data/test_rep",)

        # template = "./data/raw_geo/Li3PS9_discharged_cathode.vasp"
        # geo_type = "aims"
        # path = "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/test_band/Li3PS9/volscan"

        # template = "./data/raw_geo/Li3PS9_discharged_cathode.vasp"
        # geo_type = "dftb"
        # path = "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/battery/Li3PS9/volscan"

        template = "/home/gz_fan/Documents/ML/train/battery/CCS/CCS/test/LiCl/LiCl_mp-22905_conventional_standard.cif"
        geo_type = "aims"
        scal_params = np.linspace(0.8, 1.1, 11)  # 11, 51
        to_path = "/home/gz_fan/Documents/ML/train/battery/CCS/CCS/test/LiCl/aims_geo"
        geometries = scale(template, scal_params, to_path, geo_type)

    if task["get_dftb_energy"]:
        tot, rep = get_dftb_energy(path, scal_params)


    if task["plot"]:
        # path_to_dft = '/home/gz_fan/Documents/ML/train/battery/CCS/Li3PS9/DFT_std_tight'
        # path_to_dftb = '/home/gz_fan/Downloads/dftb/dftbplus/work/battery/Li3PS9/test_rep'
        path_to_dft = '/Users/gz_fan/Documents/dftb/CCS/test/Li5PS4Cl2_trans/aims2'
        path_to_dftb = '/Users/gz_fan/Downloads/software/dftbplus/dftbplus/work/battery/Li5PS4Cl2/li_trans2'
        scal_params_test = np.linspace(0.95, 1.05, 11)  # 11, 51

        ref_e = get_dft_energy(path_to_dft, scal_params)
        plt.show()
        tot, rep = get_dftb_energy(path_to_dftb, scal_params_test)
        plt.plot(scal_params, np.array(ref_e) - min(ref_e), "rx", label="FHI-aims")
        plt.plot(scal_params_test, np.array(tot) - min(tot), label="DFTB+")
        plt.ylim(-0.1, 0.8)
        print(np.array(ref_e) - min(ref_e))
        print(np.array(tot) - min(tot))
        # plt.yticks([])
        # plt.xlabel("Grid points with scaling params")
        plt.ylabel("E (eV)")
        plt.legend()
        plt.show()
        plt.plot(np.arange(len(rep)), rep, label="rep")
        plt.xlabel("Grid points with scaling params")
        plt.ylabel("E (eV)")
        plt.legend()
        plt.show()

    if task["gen_band"]:
        # generate band structures data
        path_dftb = (
            "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/battery"
        )
        # geometry_in_files = ['./data/raw_geo/S_cubic_10869_primitive.cif']
        # path_input_template = '/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/test_band/S_cub/tmp/control.in'
        # path_input_template = '/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/battery/Li_tri/tmp/dftb_in.hsd.band'
        # path_input_template = '/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/battery/S_cub/tmp/dftb_in.hsd.band'
        # to_geometry_type = 'aims'
        # to_geometry_path = '/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/test_band/S_cub/band'
        # to_geometry_path = '/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/battery/Li_tri/band'
        # to_geometry_path = '/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/battery/S_cub/band'

        # geometry_in_files = ["data/raw_geo/Li3PS9_discharged_cathode.vasp"]
        # path_input_template = os.path.join(path_dftb, "Li3PS9/tmp/dftb_in.hsd.band")
        # to_geometry_type = "dftbplus"
        # to_geometry_path = os.path.join(path_dftb, "Li3PS9/band")

        # geometry_in_files = ["data/raw_geo/Li5PS4Cl2.vasp"]
        # path_input_template = os.path.join(path_dftb, "Li5PS4Cl2/tmp/dftb_in.hsd.band")
        # to_geometry_type = "dftbplus"
        # to_geometry_path = os.path.join(path_dftb, "Li5PS4Cl2/band")

        # geometry_in_files = ["data/raw_geo/Li5PS6.vasp"]
        # path_input_template = os.path.join(path_dftb, "Li5PS6/tmp/dftb_in.hsd.band")
        # to_geometry_type = "dftbplus"
        # to_geometry_path = os.path.join(path_dftb, "Li5PS6/band")

        # geometry_in_files = ["../lcdftb/Germanium/Ge_mp-32_primitive.cif"]
        # path_input_template = "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/lcdftb/ge/tmp/dftb_in.hsd.band"
        # to_geometry_type = "dftbplus"
        # to_geometry_path = (
        #     "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/lcdftb/ge/band"
        # )

        # geometry_in_files = ["../lcdftb/Germanium/Ge_mp-32_primitive.cif"]
        # path_input_template = "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/test_band/lcdftb/Ge/tmp/control.in"
        # to_geometry_type = "aims"
        # to_geometry_path = "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/test_band/lcdftb/Ge/band"

        # geometry_in_files = [
        #     "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/lcdftb/C/C_mp-66_primitive.cif"
        # ]
        # path_input_template = "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/lcdftb/C/tmp/dftb_in.hsd.band"
        # to_geometry_type = "dftbplus"
        # to_geometry_path = (
        #     "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/lcdftb/C/band"
        # )
        # geometry_in_files = [
        #     "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/lcdftb/C/C_mp-66_primitive.cif"
        # ]
        # path_input_template = "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/test_band/lcdftb/C/diamond/tmp/control.in"
        # to_geometry_type = "aims"
        # to_geometry_path = "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/test_band/lcdftb/C/diamond/band"

        # geometry_in_files = [
        #     "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/lcdftb/Si/data/Si_mp-149_primitive.cif"
        # ]
        # path_input_template = "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/lcdftb/Si/diamond/tmp/control.in"
        # to_geometry_type = "aims"
        # to_geometry_path = "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/lcdftb/Si/diamond/band"
        # geometry_in_files = [
        #     "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/lcdftb/Si/data/Si_mp-149_primitive.cif"
        # ]
        # path_input_template = "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/lcdftb/Si/diamond/tmp/dftb_in.hsd.band"
        # to_geometry_type = "dftbplus"
        # to_geometry_path = "/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/test/lcdftb/Si/diamond/band"

        # geometry_in_files = [
        #     "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/lcdftb/Benchmark_Geometries/Al2O3/POSCAR_Conventional_Cell"
        # ]
        # path_input_template = "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/lcdftb/Al2O3/tmp/control.in"
        # to_geometry_type = "aims"
        # to_geometry_path = "/home/gz_fan/Downloads/software/fhiaims/fhiaims/work/lcdftb/Al2O3/band"

        # geometry_in_files = [
        #     "/home/gz_fan/Documents/ML/train/battery/data/lipscl/vasp/Li3PS9.vasp",
        #     "/home/gz_fan/Documents/ML/train/battery/data/lipscl/vasp/Li5PS4Cl2.vasp",
        #     "/home/gz_fan/Documents/ML/train/battery/data/lipscl/vasp/Li5PS6.vasp",
        #     "/home/gz_fan/Documents/ML/train/battery/data/lipscl/vasp/Li6PS4Cl2.vasp"
        # ]
        geometry_in_files = [os.path.join('/Users/gz_fan/Documents/work/band/geo', ii)
                             for ii in os.listdir('/Users/gz_fan/Documents/work/band/geo')]
        # path_input_template = "/home/gz_fan/Downloads/dftb/dftbplus/work/battery/band/dftb_in.hsd.band"
        path_input_template = "/Users/gz_fan/Documents/work/band/dftb_in.hsd.band"
        to_geometry_type = "dftbplus"
        to_geometry_path = "/Users/gz_fan/Documents/work/band/input_dftb"

        geot = GeometryTo(
            in_geometry_files=geometry_in_files,
            path_to_input_template=path_input_template,
            to_geometry_type=to_geometry_type,
            to_geometry_path=to_geometry_path,
            calculation_properties=["band"],
        )
        geot()

    if task["plot_ccs_error"]:
        error = np.loadtxt('/Users/gz_fan/Documents/dftb/CCS/test/error.out')
        print(error)
        dft = error[..., 0]
        pred = error[..., 1]
        plt.plot(np.arange(len(error)), pred - dft, 'rx')
        plt.ylabel('predicted error (H)')
        plt.show()

    if task['distance_distribution']:
        # init = ase.io.read('/home/gz_fan/Downloads/dftb/dftbplus/work/battery/Li5PS4Cl2/tmp_mdscc/geo.cif')
        # fina = ase.io.read('/home/gz_fan/Downloads/dftb/dftbplus/work/battery/Li5PS4Cl2/tmp_mdscc/geo_end.cif')
        # num = torch.unique(torch.from_numpy(init.numbers)).numpy()
        # dist0 = {(ii, jj): init.get_all_distances()[init.numbers == ii][:, init.numbers == jj]
        #          for ii in num for jj in num}
        # dist1 = {(ii, jj): fina.get_all_distances()[fina.numbers == ii][:, fina.numbers == jj]
        #          for ii in num for jj in num}
        # for (key0, val0), (key1, val1) in zip(dist0.items(), dist1.items()):
        #     plt.plot(val0, val1, '.')
        #     plt.title(key0)
        #     plt.plot([0, 10], [0, 10])
        #     plt.xlabel('perfect')
        #     plt.ylabel('MD')
        #     plt.show()
        #     # print(key0, np.average(val0[val0 != 0]))
        #     # print(key1, np.average(val1[val1 != 0]))

        md_geo = ase.io.read('/home/gz_fan/Downloads/dftb/dftbplus/work/battery/Li5PS4Cl2/li_trans2/geo_end.xyz',
                             index=':')
        all_dist = np.stack([ii.get_all_distances() for ii in md_geo])
        number = md_geo[0].numbers
        print('_dist', all_dist.shape, all_dist.transpose().shape)
        cutoff = {(3, 3): 6.0,
                  (3, 15): 6.0, (15, 3): 6.0,
                  (3, 16): 5.5, (16, 3): 5.5,
                  (3, 17): 5.5, (17, 3): 5.5,
                  (15, 15): 8.5,
                  (15, 16): 5.5, (16, 15): 5.5,
                  (15, 17): 5.5, (17, 15): 5.5,
                  (16, 16): 8.0,
                  (16, 17): 8.5, (17, 16): 8.5,
                  (17, 17): 8.0
                  }

        for ii in np.unique(number):
            for jj in np.unique(number):

                if ii > jj:
                    continue

                mask = np.expand_dims(number == ii, axis=0) * np.expand_dims(number == jj, axis=1)
                _dist = all_dist.transpose()[mask]

                _mask_d = (_dist < cutoff[(ii, jj)]) * (_dist > 1)
                select_dist = _dist[_mask_d].flatten()
                _dist_split = np.split(select_dist, np.add.accumulate(_mask_d.sum(0)))[:-1]
                print('_dist_split', len(_dist_split), [(ii.sum(), len(ii)) for ii in _dist_split])
                _dist_split = [ii.sum() / len(ii) for ii in _dist_split]
                plt.plot(np.arange(len(_dist_split)), _dist_split, label=(ii, jj))

        plt.legend()
        plt.show()
