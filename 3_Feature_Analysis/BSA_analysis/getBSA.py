# ********************************************************
# *   ----------------------------------------------     *
# * Yoochan Myung                                        *
# * The University of Melbourne                          *
# * yuchan.m@gmail.com                                   *
# *   ----------------------------------------------     *
# ********************************************************
from subprocess import Popen, PIPE
from itertools import combinations
from multiprocessing import Pool

import re
import time
import pickle
import argparse
import pandas as pd
import numpy as np
import os
import sys

def run_Freesasa(input_pd):
    result_pd = pd.DataFrame()
    for ind,row in input_pd.iterrows():
        ab_chains = str()
        ag_chains = str()
        ab_type = str()
        pdb_id,ab_chains,ag_chains = row['ID'].split('_')
        pdb_file = os.path.join(pdb_dir,row['ID']+'.pdb')
        ab_type = getAb_type(ab_chains)

        if ab_type == 'scFv':
            ab_chains = ab_chains[0]
        
        freesasa_chain_groups = ab_chains.replace('!','') + '+' + ag_chains.replace('!','')
        print("Working on...{}".format(pdb_file))
        subprocess_freesasa = Popen(['freesasa',pdb_file,'--chain-groups',freesasa_chain_groups,'--radii=naccess'],stdout=PIPE, stderr=PIPE)
        stdout, stderr = subprocess_freesasa.communicate()
        subprocess_freesasa.wait()

        freesasa_result_list = list()

        for each in stdout.splitlines():
            if str(each).startswith('b\'Total'):
                freesasa_result_list.append(float(str(each).strip().split(':')[1].strip()[0:-1]))
        result_pd.loc['{}'.format(row['ID']),'sasaTotal'] = freesasa_result_list[0]
        result_pd.loc['{}'.format(row['ID']),'sasaAb'] = freesasa_result_list[1]
        result_pd.loc['{}'.format(row['ID']),'sasaAg'] = freesasa_result_list[2]
        result_pd.loc['{}'.format(row['ID']),'BSA'] = (freesasa_result_list[1]+freesasa_result_list[2]) - freesasa_result_list[0]
        result_pd.loc['{}'.format(row['ID']),'type'] = ab_type

    return result_pd

def getAb_type(HL_chains):
    if HL_chains[0] == '!':
        ab_type = 'Fll'
    elif HL_chains[1] == '!':
        ab_type = 'Nanobody'
    elif HL_chains[0] == HL_chains[1]:
        ab_type = 'scFv'
    else:
        ab_type = 'Fab'
    return ab_type

def parallelize_dataframe(df, func, num_cores):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

if __name__ == "__main__":
    print("Please check the 'pdb_dir' in the file!")
    global pdb_dir
    
    parser = argparse.ArgumentParser(description="This is a script for calculating Buried Surface Area using FreeSASA.")
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
    pdb_dir = args.pdb_dir

    input_pd = pd.read_csv(input_tsv,sep='\t')
    bsa_pd = parallelize_dataframe(input_pd,run_Freesasa,cores)
    bsa_pd = bsa_pd.round(2)
    bsa_pd.to_csv('result_BSA.csv',index_label='ID')


    # python getBSA.py test.tsv 8

