# ********************************************************
# *   ----------------------------------------------     *
# * Yoochan Myung                                        *
# * The University of Melbourne                          *
# * yuchan.m@gmail.com                                   *
# *   ----------------------------------------------     *
# ********************************************************
# Create a bash script to run Ghecom (Rinaccess)
import os
import argparse
import pandas as pd
import sys
from subprocess import Popen, PIPE

def runGhecom_Rinaccess(input_tsv):
    input_pd = pd.read_csv(input_tsv,sep='\t',index_col='ID')
    index_list = input_pd.index.tolist()

    uniq_ab_list = list()
    uniq_ag_list = list()

    for each in index_list:
        pdb_id = str()
        ab_chains = str()
        ag_chains = str()
        pdb_id, ab_chains, ag_chains = each.split('_')
        uniq_ab_list.append("{}_{}".format(pdb_id,ab_chains))
        uniq_ag_list.append("{}_{}".format(pdb_id,ag_chains))

    uniq_ab_list = list(set(uniq_ab_list))
    uniq_ag_list = list(set(uniq_ag_list))

    output_run_file = open('run_Rinaccess.sh','w+')

    for each in uniq_ab_list:
        pdb_id, ab_chains = each.split('_')
        input_pdb = '{}.pdb'.format(pdb_id)
        output_pdb = '{}_{}_ab_rinaccess.pdb'.format(pdb_id,ab_chains)    
        output_run_file.write("ghecom M -ipdb {} -ch {} -opdb {} -tfac R \n".format(input_pdb, ab_chains.replace('!',''), output_pdb))
    
    for each in uniq_ag_list:
        pdb_id, ag_chains = each.split('_')
        input_pdb = '{}.pdb'.format(pdb_id)
        output_pdb = '{}_{}_ag_rinaccess.pdb'.format(pdb_id,ag_chains)
        output_run_file.write("ghecom M -ipdb {} -ch {} -opdb {} -tfac R \n".format(input_pdb, ag_chains, output_pdb))

    for each in index_list:
        pdb_id, ab_chains, ag_chains = each.split('_')
        input_pdb = '{}.pdb'.format(pdb_id)
        output_pdb = '{}_rinaccess.pdb'.format(each)
        chain = ab_chains.replace('!','') + ag_chains
        output_run_file.write("ghecom M -ipdb {} -ch {} -opdb {} -tfac R \n".format(input_pdb, chain, output_pdb))
    
    output_run_file.close()
    print("DONE")

def runGhecom_Rinaccess_as_ligand(input_tsv):
    input_pd = pd.read_csv(input_tsv,sep='\t',index_col='ID')
    index_list = input_pd.index.tolist()

    output_run_file = open('run_Rinaccess_lig.sh','w+')

    for each in index_list:
        pdb_id = str()
        ab_chains = str()
        ag_chains = str()
        pdb_id, ab_chains, ag_chains = each.split('_')
        antibody_pdb =  '{}_{}.pdb'.format(pdb_id, ab_chains)
        antigen_pdb = '{}_{}.pdb'.format(pdb_id, ag_chains)

        antibody_rinaccess_pdb = '{}_ab_rinaccess.pdb'.format(each)
        antigen_rinaccess_pdb =  '{}_ag_rinaccess.pdb'.format(each)

        output_run_file.write("ghecom M -ipdb \'{}\' -iligpdb \'{}\' -oligpdb \'{}\' -tfac R -rli 2.5 -rlx 10.0  \n".format(antigen_pdb, antibody_pdb, antibody_rinaccess_pdb)) # For Ab Rinaccess
        output_run_file.write("ghecom M -ipdb \'{}\' -iligpdb \'{}\' -oligpdb \'{}\' -tfac R -rli 2.5 -rlx 10.0  \n".format(antibody_pdb, antigen_pdb, antigen_rinaccess_pdb)) # For Ag Rinaccess

    output_run_file.close()
    print("DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('type_of_run',type=str,\
        choices=['ligand','general'],
        default='ligand',
        help='choose between ligand or general')
    parser.add_argument('input_tsv',type=str,\
        help='input_cluster pandas tsv')

    args = parser.parse_args()
    input_tsv = args.input_tsv
    type_of_run = args.type_of_run

    if type_of_run == 'ligand':
        runGhecom_Rinaccess_as_ligand(input_tsv)
    else:        
        runGhecom_Rinaccess(input_tsv)

if __name__ == "__main__":
    main()
