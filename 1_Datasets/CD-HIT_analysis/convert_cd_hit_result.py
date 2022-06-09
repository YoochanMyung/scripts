import re
import os
import sys
import ast
import pandas as pd
import requests
from Bio.PDB.PDBList import PDBList
from Bio import pairwise2
from Bio import SeqIO
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

def convert_cd_hit_result(cd_hit_clstr_file):
    loc_of_result = cd_hit_clstr_file
    dir_loc = os.path.dirname(loc_of_result)
    out_filename = os.path.join(dir_loc,loc_of_result+'_pandas.csv')

    result_dic = dict()
    result_pd = pd.DataFrame()
    
    with open (loc_of_result,'r') as f:
        temp_list = list()
        for each in f.readlines():
            each = each.strip()
            # input("Press the <ENTER> key to continue...")
            if each.startswith('>'):
                if len(temp_list) >1:
                    ttemp_list = list()
                    repr_pdb = str()
                    cluster_ID = temp_list[0].split(' ')[1]
                    cluster_contents = temp_list[1:]
                    result_dic[cluster_ID] = cluster_contents

                    for eeach in temp_list[1:]:
                        if eeach.split(',')[1][-1] == '*':
                            repr_pdb = eeach.split(', >')[1].split('|')[0]
                            repr_pdb_chain = eeach.split(', >')[1].split('|')[1][0]
                            repr_info = repr_pdb+'_'+repr_pdb_chain
                        else:
                            not_repr_pdb = eeach.split(', >')[1].split('|')[0]
                            not_repr_pdb_chain = eeach.split(', >')[1].split('|')[1][0]
                            not_repr_info = not_repr_pdb+'_'+ not_repr_pdb_chain
                            ttemp_list.append(not_repr_info)
                        result_dic[temp_list[0].split(' ')[1]] = ','.join(ttemp_list)
                    result_pd = result_pd.append({'cluster':temp_list[0].split(' ')[1],'rep':repr_info,'others':','.join(ttemp_list)},ignore_index=True)
                temp_list = list()
            temp_list.append(each)
    result_pd.to_csv(out_filename,index=False,sep=';')
    return True

if __name__ == '__main__':
    cd_hit_result = sys.argv[1]
    convert_cd_hit_result(cd_hit_result)