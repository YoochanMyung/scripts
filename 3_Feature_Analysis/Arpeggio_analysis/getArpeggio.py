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
import numpy as np
import pandas as pd
import os
import sys
import argparse
pdb_dir = './SabDab/'

def run_Arpeggio_interface(input_pd):
    for ind,row in input_pd.iterrows():
        pdb_file = os.path.join(pdb_dir,row['ID']+'.pdb')
        ab_chains = row['Hchain']+row['Lchain']
        ag_chains = row['antigen_chain'].replace(' | ','')

        epitope_list = list()
        paratope_list = list()

        epitope_file = os.path.join(pdb_dir,os.path.basename(pdb_file)[0:4]+"_{}_{}_epitope.txt".format(ab_chains,ag_chains))
        paratope_file = os.path.join(pdb_dir,os.path.basename(pdb_file)[0:4]+"_{}_{}_paratope.txt".format(ab_chains,ag_chains))

        epitope_list = open(epitope_file).read()
        paratope_list = open(paratope_file).read()
        interface_list = "/"+ epitope_list.replace("_","/").replace(",","/ /") + "/ /" + paratope_list.replace("_","/").replace(",","/ /")+ "/"

        print("Working on...{}".format(pdb_file))
        try:
            os.system('python {}src/Arpeggio/arpeggio.py {} -s {} -op _interface'.format(pdb_dir,pdb_file,interface_list))
        except Exception as e:
            print('Got an Error from {}'.format(pdb_file))
            print(e)


def run_Arpeggio_CDR(input_pd):
    for ind,row in input_pd.iterrows():
        pdb_file = os.path.join(pdb_dir,row['ID']+'.pdb')
        ab_chains = row['Hchain']+row['Lchain']
        ag_chains = row['antigen_chain'].replace(' | ','')

        CDR_list = list()
        CDR_file = os.path.join(pdb_dir,os.path.basename(pdb_file)[0:4]+"_{}_{}_CDR.txt".format(ab_chains,ag_chains))
        CDR_list = open(CDR_file).read()
        residue_list = "/"+ CDR_list.replace("_","/").replace(",","/ /") + "/"

        print("Working on...{}".format(pdb_file))
        try:
            os.system('python {}src/Arpeggio/arpeggio.py {} -s {} -op _CDR'.format(pdb_dir,pdb_file,residue_list))
        except Exception as e:
            print('Got an Error from {}'.format(pdb_file))
            print(e)

def run_Arpeggio_whole(input_pd):
    for ind,row in input_pd.iterrows():
        pdb_file = os.path.join(pdb_dir,row['ID']+'.pdb')
        print("Working on...{}".format(pdb_file))        
        try:
            os.system('python {}src/Arpeggio/arpeggio.py {} -op _whole'.format(pdb_dir, pdb_file))
        except Exception as e:
            print('Got an Error from {}'.format(pdb_file))
            print(e)

def parallelize_dataframe(df, func, num_cores):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pool.map(func, df_split)
    pool.close()
    pool.join()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a script for running Arpeggio in parallel.\
     To run in interface mode, you have to have paratope and interface information in text file \
     which can be obtained by getIntRes.py.")
    parser.add_argument('input_tsv',type=str,\
        help='input_cluster pandas tsv')
    parser.add_argument('type_of_run',type=str,\
        choices=['whole','interface','CDR'],
        default='whole',
        help='choose between interface or whole')

    args = parser.parse_args()

    input_tsv = args.input_tsv
    type_of_run = args.type_of_run
    input_pd = pd.read_csv(input_tsv,sep='\t')

    if type_of_run == 'interface':
        parallelize_dataframe(input_pd,run_Arpeggio_interface,4)
    elif type_of_run == 'CDR':
        parallelize_dataframe(input_pd,run_Arpeggio_CDR,4)
    else:
        parallelize_dataframe(input_pd,run_Arpeggio_whole,4)

