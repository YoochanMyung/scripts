# ********************************************************
# *   ----------------------------------------------     *
# * Yoochan Myung                                        *
# * The University of Melbourne                          *
# * yuchan.m@gmail.com                                   *
# *   ----------------------------------------------     *
# ********************************************************
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser, PDBIO, Select
from subprocess import Popen, PIPE
from multiprocessing import Pool
import sys
import os
import warnings
import pandas as pd
import numpy as np

class SelectFilter(Select):
    def __init__(self, chain_letters):
        self.chain_letters = chain_letters    
    def accept_atom(self, atom):        
        if (not atom.is_disordered()) or atom.get_altloc() == 'A': # Filter only A conformation if alternatives exist
            if not atom.full_id[3][0] == 'W' and not len(atom.full_id[3][0]) == 4:   # Remove Water and Ion
                atom.set_altloc(" ") # Replace "A" for A conformation
                return True
            else:
                return False
        else:
            return False
    def accept_chain(self, chain):
        return (chain.get_id() in self.chain_letters)

def cleanPDB(input_pd):
    for ind,row in input_pd.iterrows():
        pdb_id = str()
        ab_chains = str()
        ag_chains = str()

        pdb_id,ab_chains,ag_chains = row['ID'].split('_')
        input_pdb = os.path.join(pdb_dir,'{}.pdb'.format(pdb_id))
        chains = ab_chains + ag_chains
        
        if not chains.isalpha():
            chains.replace('!','')

        complex_pdb = os.path.join(pdb_dir,'{}_{}_{}.pdb'.format(pdb_id,ab_chains,ag_chains))
        ab_pdb = os.path.join(pdb_dir,'{}_{}.pdb'.format(pdb_id,ab_chains))
        ag_pdb = os.path.join(pdb_dir,'{}_{}.pdb'.format(pdb_id,ag_chains))

        print("Processing {}....".format(complex_pdb))
        parser = PDBParser()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',PDBConstructionWarning)
            structure = parser.get_structure('input_pdb',input_pdb)[0] # Filter only MODEL 0
            io = PDBIO()
            io.set_structure(structure)
            io.save(complex_pdb,select=SelectFilter(chains))
            io.save(ab_pdb,select=SelectFilter(ab_chains))
            io.save(ag_pdb,select=SelectFilter(ag_chains))


def parallelize_dataframe(df, func, num_cores):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pool.map(func, df_split)
    pool.close()
    pool.join()

if __name__ == '__main__':
    global pdb_dir
    parser = argparse.ArgumentParser(description="This is a script for cleaning PDB files.")
    parser.add_argument('input_tsv',type=str,\
        help='input_cluster pandas tsv')
    parser.add_argument('pdb_dir',type=str,\
        help='location of PDBs')        
    parser.add_argument('cores',type=str,\
        default=4,
        help='Choose the number of cores for parallelization')

    args = parser.parse_args()
    input_tsv = args.input_tsv
    pdb_dir = args.pdb_dir
    cores = args.cores
    
    input_pd = pd.read_csv(input_tsv,sep='\t')
    parallelize_dataframe(input_pd,cleanPDB,cores)

