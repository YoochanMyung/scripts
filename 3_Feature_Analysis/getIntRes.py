# ********************************************************
# *   ----------------------------------------------     *
# * Yoochan Myung                                        *
# * The University of Melbourne                          *
# * yuchan.m@gmail.com                                   *
# *   ----------------------------------------------     *
# ********************************************************
from multiprocessing import Pool
from pymol import cmd

import os
import sys
import argparse
import pandas as pd
import numpy as np

def getInterfaceResidue(input_pd):
    for ind,row in input_pd.iterrows():
        print('Working on .... {}'.format(row['pdb']))
        pdb_file = os.path.join(pdb_dir,row['ID']+'.pdb')
        ab_chains = row['Hchain']+row['Lchain']
        ag_chains = row['antigen_chain'].replace(' | ','')
        output_filename_paratope = os.path.join(os.path.dirname(pdb_file),os.path.basename(pdb_file).split('.')[0]+"_paratope.txt")
        output_filename_epitope = os.path.join(os.path.dirname(pdb_file),os.path.basename(pdb_file).split('.')[0]+"_epitope.txt")
        intResidues = {'paratope':[],'epitope':[]}

        paratope_list = list()
        epitope_list = list()

        selection_ag_chains = ' or ' .join(['chain '+ ag for ag in list(ag_chains)])
        selection_ab_chains = ' or ' .join(['chain '+ ab for ab in list(ab_chains)])

        cmd.load(pdb_file,'input_pdb')
        cmd.select('ab_chains',selection='{}'.format(selection_ab_chains))
        cmd.select('ag_chains',selection='{}'.format(selection_ag_chains))
        cmd.select('ab_intResidues',selection='(ab_chains within 5.1 of ag_chains) and not hetatm')
        cmd.select('ag_intResidues',selection='(ag_chains within 5.1 of ab_chains) and not hetatm')
        cmd.iterate(selection='ab_intResidues',expression='paratope.append([chain,resi])',space=intResidues)
        cmd.iterate(selection='ag_intResidues',expression='epitope.append([chain,resi])',space=intResidues)
        cmd.remove(selection='all')

        for each in intResidues['paratope']:
            paratope_list.append('_'.join(each))
        paratope_list = list(set(paratope_list))

        for each in intResidues['epitope']:
            epitope_list.append('_'.join(each))
        epitope_list = list(set(epitope_list))

        with open(output_filename_paratope,"w") as f:
            f.write(",".join(paratope_list))
            f.close()
        with open(output_filename_epitope,"w") as f:
            f.write(",".join(epitope_list))
            f.close()

def parallelize_dataframe(df, func, num_cores):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pool.map(func, df_split)
    pool.close()
    pool.join()

if __name__ == '__main__':
    global pdb_dir
    parser = argparse.ArgumentParser(description="This is a script for getting interface residues using Pymol")
    parser.add_argument('input_tsv',type=str,\
        help='input_cluster pandas tsv')
    parser.add_argument('pdb_dir',type=str,\
        help='location of PDBs')                
    parser.add_argument('cores',type=str,\
        default=4,
        help='Choose the number of cores for parallelization')

    args = parser.parse_args()
    input_tsv = args.input_tsv
    cores = args.cores
    input_pd = pd.read_csv(input_tsv,sep='\t')
    parallelize_dataframe(input_pd,getInterfaceResidue,cores)

    # python getIntRes.py test.tsv 8

