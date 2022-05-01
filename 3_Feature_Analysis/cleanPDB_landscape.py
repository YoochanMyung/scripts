# ********************************************************
# *   ----------------------------------------------     *
# * Yoochan Myung                                        *
# * The University of Melbourne                          *
# * yuchan.m@gmail.com                                   *
# *   ----------------------------------------------     *
# ********************************************************
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser, PDBIO, Select
import sys
import os
import warnings

class cleanFilter(Select):
    def accept_atom(self, atom):
        if (not atom.is_disordered()) or atom.get_altloc() == 'A': # Filter only A conformation if alternatives exist
            if not atom.full_id[3][0] == 'W' and not len(atom.full_id[3][0]) == 4:   # Remove Water and Ion
                atom.set_altloc(" ") # Replace "A" for A conformation
                return True
            else:
                return False
        else:
            return False

def cleanPDB(input_pdb):
    folder = os.path.dirname(input_pdb)
    filename_list = os.path.basename(input_pdb).split('.')
    new_pdb = os.path.join(folder,'{}_cleaned.{}'.format(filename_list[0],filename_list[1]))
    print("Processing {}....".format(new_pdb))
    parser = PDBParser()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',PDBConstructionWarning)
        structure = parser.get_structure('input_pdb',input_pdb)[0] # Filter only MODEL 0
        io = PDBIO()
        io.set_structure(structure)
        io.save(new_pdb,select=cleanFilter())
        return new_pdb

if __name__ == '__main__':
    input_pdb = sys.argv[1]
    cleanPDB(input_pdb)
