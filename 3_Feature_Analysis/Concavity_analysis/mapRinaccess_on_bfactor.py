# ********************************************************
# *   ----------------------------------------------     *
# * Yoochan Myung                                        *
# * The University of Melbourne                          *
# * yuchan.m@gmail.com                                   *
# *   ----------------------------------------------     *
# ********************************************************
from Bio.PDB import PDBIO, PDBParser, PDBList, Select, Selection

import os
import sys
import pandas as pd
import argparse
import numpy as np

class HetSelect(Select):
    def accept_residue(self, residue):
        if residue.id[0] == ' ' or residue.id[0] == 'H_MSE':
            return True
        else:
            return False

def map_AtomRinsaccess_on_Bfactor(pdb_file, ghecom_atom_file):
    result_dict = dict() # {'chainID_resNumber' : avr_Rinaccess}

    with open(ghecom_atom_file,'r') as ofile:
        for line in ofile.readlines():
            _identifier = str()
            if line.startswith('HETATM') or line.startswith('ATOM'):
                _atom = line.strip()[12:17].strip()
                _resn = line.strip()[17:21].strip()
                _chain = line.strip()[21:22].strip()
                _resi = line.strip()[22:27].strip()
                _shellaccess = line.strip()[67:73].strip()
                _rinaccess = line.strip()[73:78].strip()
                # _normalised_rinaccess = float(_rinaccess)*100/float(_shellaccess)
                _identifier = '{}_{}_{}'.format(_chain,_resi,_atom)
                result_dict[_identifier] = _rinaccess

    parser = PDBParser()
    structure = parser.get_structure('input',pdb_file)
    models = structure[0]
    for model in models:
        for residue in model:
            for atom in residue:
                _chain = residue.get_full_id()[2]
                _resi = residue.get_full_id()[3][1]
                _resn = residue.get_resname()
                _hetatm = atom.get_full_id()[3][0]
                _atom = atom.get_full_id()[4][0]
                if len(_hetatm) >1 and _hetatm != 'H_MSE':
                    print("hetatm",residue.get_id())
                else:
                    _rinaccess = float(result_dict['{}_{}_{}'.format(_chain,_resi,_atom)])
                    atom.bfactor = _rinaccess
       
    new_pdb = os.path.join('{}_Rinaccess_atom.pdb'.format(pdb_file[:-4]))
    io = PDBIO()
    io.set_structure(structure)
    io.save(new_pdb,HetSelect())
    print("{} is successfully created.".format(new_pdb))
    return True

def map_ResRinsaccess_on_Bfactor(pdb_file, ghecom_atom_file, avr_opt):
    result_dict = dict() # {'chainID_resNumber' : avr_Rinaccess}

    with open(ghecom_atom_file,'r') as ofile:
        for line in ofile.readlines():
            _identifier = str()
            if line.startswith('HETATM') or line.startswith('ATOM'):
                _atom = line.strip()[12:17].strip()
                _resn = line.strip()[17:21].strip()
                _chain = line.strip()[21:22].strip()
                _resi = line.strip()[22:27].strip()
                _identifier = '{}_{}'.format(_chain,_resi)
                result_dict[_identifier] = list()

    with open(ghecom_atom_file,'r') as ofile:
        for line in ofile.readlines():
            _identifier = str()
            if line.startswith('HETATM') or line.startswith('ATOM'):
                _atom = line.strip()[12:17].strip()
                _resn = line.strip()[17:21].strip()
                _chain = line.strip()[21:22].strip()
                _resi = line.strip()[22:27].strip()
                _rinaccess = line.strip()[73:79].strip()
                _identifier = '{}_{}'.format(_chain,_resi)

                for key in result_dict.keys():
                    if avr_opt == 'whole':
                        if _identifier == key: # whole
                            result_dict[key].append(_rinaccess)
                    else:
                        if _identifier == key and _atom not in ['N','CA','C','O']: # whole
                            result_dict[key].append(_rinaccess)

    for key,value in result_dict.items():
        _list = list(map(float,value))
        if len(_list) == 0:
            result_dict[key] = 0
        else:
            avr_rinaccess = np.average(_list).round(2)
            result_dict[key] = avr_rinaccess
    
    parser = PDBParser()
    structure = parser.get_structure('input',pdb_file)
    models = structure[0]
    for model in models:
        for residue in model:
            _chain = residue.get_full_id()[2]
            _resi = residue.get_full_id()[3][1]
            _resn = residue.get_resname()
            _hetatm = residue.get_full_id()[3][0]
            if len(_hetatm) >1 and _hetatm != 'H_MSE':
                pass
            else:
                _rinaccess = float(result_dict['{}_{}'.format(_chain,_resi)])
                for atom in residue:
                    atom.bfactor = _rinaccess
    
    new_pdb = os.path.join('{}_Rinaccess_avr_{}.pdb'.format(pdb_file[:-4],avr_opt))
    io = PDBIO()
    io.set_structure(structure)
    io.save(new_pdb,HetSelect())
    print("{} is successfully created.".format(new_pdb))
    return True

def main():
    parser = argparse.ArgumentParser(description="This is a script for mapping Rinaccess\
     on b-factor column of a given pdb.")
    parser.add_argument('type_of_run',type=str,\
        help='choose a type of run',
        choices=['atom','residue'])
    parser.add_argument('pdb_file',type=str,\
        help='a PDB file')
    parser.add_argument('ghecom_file',type=str,\
        help='choose a ghecom result file')
    parser.add_argument('--avr',type=str,\
        choices=['sidechain','whole'],
        default='whole',
        help='choose between sidechain or whole atoms for residue-level Rinaccess mapping')

    args = parser.parse_args()
    type_of_run = args.type_of_run
    pdb_file = args.pdb_file
    ghecom_atom_file = args.ghecom_file
    avr_opt = args.avr

    if type_of_run == 'residue':
        map_ResRinsaccess_on_Bfactor(pdb_file, ghecom_atom_file, avr_opt)
    else:
        map_AtomRinsaccess_on_Bfactor(pdb_file, ghecom_atom_file)

if __name__ == "__main__":
    main()
