# ********************************************************
# *   ----------------------------------------------     *
# * Yoochan Myung                                        *
# * The University of Melbourne                          *
# * yuchan.m@gmail.com                                   *
# *   ----------------------------------------------     *
# ********************************************************
from Bio.SeqUtils import seq1
from Bio.PDB import *

import argparse
import sys
import os
import re

def acc_to_bfactor(args):
    input_pdb = args.input_pdb
    dssp_input = input_pdb[:-3] + 'dssp'

    dssp_out_dict = dict()
    aa3to1 = {
        'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
        'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
        'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
    }
    try:
        with open(dssp_input,'r') as dssp:
            while "  #  RESIDUE AA STRUCTURE" not in dssp.readline():
                continue
            lines = dssp.readlines()
    except:
        print("Cannot find {}".format(dssp_input))
        print("Please check your input files.")
        sys.exit()

    for each in ''.join(lines).split('\n'):
        _temp_list = re.sub(' +',',',each[0:39].strip()).split(',')
        if len(_temp_list) > 3:
            _resn = _temp_list[1]
            _chain = _temp_list[2]
            _res = _temp_list[3]
            _acc = int(_temp_list[-1])

            if _chain.isalpha():
                dssp_out_dict['{}_{}'.format(_chain,_resn)] = _acc 

    max_acc = max(dssp_out_dict.values())

    parser = PDBParser()
    structure = parser.get_structure('input',input_pdb)
    models = structure[0]
    for model in models:
        for residue in model:
            _chain = str()
            _resn = int()
            _res = str()
            try:
                _chain = residue.get_full_id()[2]
                _resn = residue.get_full_id()[3][1]
                _acc = int(dssp_out_dict['{}_{}'.format(_chain,_resn)])
                for atom in residue:
                    atom.bfactor = _acc
            except:
                pass
    
    new_pdb = os.path.join('{}_dssp.pdb'.format(input_pdb[:-4]))
    io = PDBIO()
    io.set_structure(structure)
    io.save(new_pdb)
    print('{} is successfully created.'.format(new_pdb))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map ACC(solvent accessibility) of DSSP to b-factor column of input PDB file. ')
    parser.add_argument('input_pdb', type=str, help='Provide an input pdb.')
    args = parser.parse_args()
    acc_to_bfactor(args)