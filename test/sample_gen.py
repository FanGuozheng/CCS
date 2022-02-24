import ase
import ase.io as io

import matplotlib.pyplot as plt
import numpy as np

from ccs.scripts.atom_json import atom_json


def gen_distance(path_to_file, index):
    """Generate new geometries and make sure the distributions reasonable."""
    ase_obj = io.read(path_to_file)
    ase_list = []

    shift_dist = [[np.array([0, 0, -2.0]), np.array([0, 0, -1]), np.array([0, 0, 1]),
                   np.array([0, 0, 2])],
                  [np.array([0, 0, -2.0]), np.array([0, 0, -1]), np.array([0, 0, 1]),
                   np.array([0, 0, 2])]]

    for ish in shift_dist[0]:
        for jsh in shift_dist[1]:
            positions = ase_obj.positions
            positions[index[0]] = positions[index[0]] + ish
            positions[index[1]] = positions[index[1]] + jsh
            ase_list.append(ase.Atoms(
                numbers=ase_obj.numbers, positions=positions, cell=ase_obj.cell))

    numbers = np.pad([ii.numbers for ii in ase_list], pad_width=0)
    all_dist = [ii.get_all_distances() for ii in ase_list]
    uniq_numbers = np.unique(numbers)

    fig, axs = plt.subplots(len(uniq_numbers), len(uniq_numbers))
    for ii, inum in enumerate(uniq_numbers):
        for jj, jnum in enumerate(uniq_numbers):
            pair = [inum, jnum]
            _dist = [dist[inum==num][:, jnum==num] for num, dist in zip(numbers, all_dist)][0]
            axs[ii, jj].hist(_dist[_dist != 0], bins=100)
            axs[ii, jj].set_title(pair)
            axs[ii, jj].set_xlim(0, 5)
    plt.show()


def gen_distance_md(path_to_file, index, write=True):
    """Generate new geometries and make sure the distributions reasonable."""
    ase_obj = io.read(path_to_file, index=':', format='xyz')
    for ii in ase_obj:
        ii.cell = [9.897999763500000, 9.897999763500000, 9.897999763500000]
        ii.pbc = True
    print(ase_obj[0].cell)
    numbers = ase_obj[0].numbers
    all_dist = [ii.get_all_distances() for ii in ase_obj]
    uniq_numbers = np.unique(numbers)

    count = 0
    for ii, iase in enumerate(ase_obj):
        if ii % 2 == 0 and ii > 2 and ii < 26:
            print(iase.cell)
            io.write(f'geometry.in.{count}', iase, format='aims')
            io.write(f'geo.gen.{count}', iase, format='dftb')
            count += 1

    fig, axs = plt.subplots(len(uniq_numbers), len(uniq_numbers))
    for ii, inum in enumerate(uniq_numbers):
        for jj, jnum in enumerate(uniq_numbers):
            pair = [inum, jnum]
            _dist = all_dist[0][inum==numbers][:, jnum==numbers]
            axs[ii, jj].hist(_dist[_dist != 0], bins=200)
            axs[ii, jj].set_title(pair, fontsize='small')
            axs[ii, jj].set_xlim(0, 5)
    plt.show()

    fig, axs = plt.subplots(len(uniq_numbers), len(uniq_numbers))
    for ii, inum in enumerate(uniq_numbers):
        for jj, jnum in enumerate(uniq_numbers):
            pair = [inum, jnum]
            for kk, dist in enumerate(all_dist):
                if kk % 2 == 0 and kk > 2 and kk < 26:
                    _dist = dist[inum == numbers][:, jnum == numbers]
                    axs[ii, jj].hist(_dist[_dist != 0], bins=200)
            axs[ii, jj].set_title(pair, fontsize='small')
            axs[ii, jj].set_xlim(0, 5)
    plt.show()


if __name__ == "__main__":
    # path_to_file = 'Li5PS4Cl2/geo/Li5PS4Cl2.vasp'
    # gen_distance(path_to_file, index=[9, 19])

    path_to_mdgeo = '/Users/gz_fan/Downloads/software/dftbplus/dftbplus/work/battery/Li5PS4Cl2/li_md2/geo_end.xyz'
    gen_distance_md(path_to_mdgeo, index=[9, 19])
