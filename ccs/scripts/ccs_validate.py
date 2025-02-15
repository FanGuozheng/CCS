import sys
from ase.db import connect
from ase.io import Trajectory, read, write
import re
import numpy as np
import ase.db as db
import json
import random
from tqdm import tqdm
try:
    from pymatgen import Lattice, Structure
    from pymatgen.analysis import ewald
except:
    pass
from ccs.ase_calculator.ccs_ase_calculator import CCS


def validate(mode=None, CCS_params='CCS_params.json', Ns='all', DFT_DB=None, CCS_DB='CCS_DB.db', DFTB_DB=None, charge=False, q=None, charge_scaling=False):

    DFT_DB = db.connect(DFT_DB)
    CCS_DB = db.connect(CCS_DB)

    f = open("CCS_validate.dat", 'w')
    print("#Reference      Predicted      Error      No_of_atoms structure_no", file=f)

    calc = CCS(CCS_params=CCS_params, charge=charge,
               q=q, charge_scaling=charge_scaling)

    if(Ns != 'all'):
        mask = [a <= Ns for a in range(len(DFT_DB))]
        random.shuffle(mask)
    else:
        mask = len(DFT_DB)*[True]

    counter = -1
    for row in tqdm(DFT_DB.select(), total=len(DFT_DB), colour='#800000'):
        counter += 1
        if(mask[counter]):
            key = row.key
            structure = row.toatoms()
            EDFT = structure.get_total_energy()
            structure.calc = calc
            ECCS = structure.get_potential_energy()
            print(EDFT, ECCS, EDFT-ECCS, len(structure), counter,  file=f)
            CCS_DB.write(structure, key=key)


def main():
    print("--- USAGE:  ccs_validate MODE [...] --- ")
    print(" ")
    print("       The following modes and inputs are supported:")
    print("       CCS:   CCS_params_file(string) NumberOfSamples(int) DFT.db(string)")
    print("       CCS+Q: CCS_params_file(string) NumberOfSamples(int) DFT.db(string) charge_dict(string) charge_scaling(bool)")
    print("       DFTB:  Not yet supported...")
    print(" ")

    assert sys.argv[1] in ['CCS', 'CCS+Q', 'DFTB'], 'Mode not supproted.'

    mode = sys.argv[1]
    CCS_params_file = sys.argv[2]
    Ns = int(sys.argv[3])
    DFT_data = sys.argv[4]
    with open(CCS_params_file, 'r') as f:
        CCS_params = json.load(f)

    print("    Mode: ", mode)
    if(mode == "CCS"):
        print("    Number of samples: ", Ns)
        print("    DFT reference data base: ", DFT_data)
        print("")
        print("-------------------------------------------------")
        validate(mode=mode, CCS_params=CCS_params, Ns=Ns, DFT_DB=DFT_data)
    if(mode == "CCS+Q"):
        print("        NOTE: charge_dict should use double quotes to enclose property nanes. Example:")
        print("        \'{\"Zn\":2.0,\"O\" : -2.0 } \'")
        charge_dict = sys.argv[5]
        charge_scaling = bool(sys.argv[6])
        print("    Number of samples: ", Ns)
        print("    DFT reference data base: ", DFT_data)
        print("    Charge dictionary: ", charge_dict)
        print("    Charge scaling: ", charge_scaling)
        print("")
        print("-------------------------------------------------")
        charge_dict = json.loads(charge_dict)
        validate(mode=mode, CCS_params=CCS_params, Ns=Ns, DFT_DB=DFT_data, charge=True,
                 q=charge_dict, charge_scaling=charge_scaling)


if __name__ == "__main__":
    main()
