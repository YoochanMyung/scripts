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
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB import *
from Bio import SeqIO
import re
import time
import pickle
import argparse
import os
import sys
import pandas as pd
import numpy as np
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3
pdb_dir = './SabDab/'

def RabinKarp_search(pattern, text):
    d = len(text)
    q = len(text) + len(pattern)
    m = len(pattern)
    n = len(text)
    p = 0
    t = 0
    h = 1
    i = 0
    j = 0

    for i in range(m-1):
        h = (h*d) % q

    # Calculate hash value for pattern and text
    for i in range(m):
        p = (d*p + ord(pattern[i])) % q
        t = (d*t + ord(text[i])) % q

    # Find the match
    for i in range(n-m+1):
        if p == t:
            for j in range(m):
                if text[i+j] != pattern[j]:
                    break

            j += 1
            if j == m:
                # print("Pattern is found at position: " + str(i+1))
                return int(i)

        if i < n-m:
            t = (d*(t-ord(text[i])*h) + ord(text[i+m])) % q

            if t < 0:
                t = t+q

def runDirectly(fasta, scheme):
    # only for chothia
    try:
        anarci_run = Popen(["ANARCI", "--sequence", fasta, "--scheme", scheme], stdout=PIPE, stderr=PIPE)
        stdout, stderr = anarci_run.communicate();  # print(len(stdout.decode('ascii')))
        return stdout.decode('ascii')
    except:
        print("GOT ERROR while running runDirectly")
    
def getAllChains(pdb_file):
    chain_list = []
    p = PDBParser()
    structure = p.get_structure('input_pdb', pdb_file)
    for each in structure[0]:
        chain_list.append(each.get_id())
    return ''.join(chain_list)

def getFASTAnPDBnumberMap(input_pdb,chain_id):
    pdb_id = input_pdb.split('/')[-1]
    new_dic = dict()
    nn = list()
    out_list = pd.DataFrame()
    chain_id = str(chain_id)

    for each in new_dic.values():
        nn.append(" ".join(each).split(" "))

    structure = PDBParser(QUIET=True).get_structure('input_pdb', input_pdb)
    tempp_list = list()

    # for model in structure:
    model = structure[0]
    new_number = 1
    for residue in model[chain_id]:
        tempp_list.append([new_number, str(residue)])
        chain_ID = str(model[chain_id]).rsplit('id=')[1][0].strip()
        amino_acid_name = str(residue).split('Residue')[1].strip().split(' ')[0]
        amino_acid_number = str(residue).split('resseq=')[1].split('icode')[0].strip()
        icode_code = str(residue).rsplit('icode=')[1].strip()

        if len(icode_code) != 1:
            icode_code = icode_code[0].strip()
            amino_acid_number = amino_acid_number + icode_code
        out_list = out_list.append({'chain':chain_id,'wild':amino_acid_name,'pdb_numb':amino_acid_number,'fasta_numb':str(new_number)},ignore_index=True)
        new_number += 1
    
    return out_list

def readFasta(fasta):
    buf = StringIO(fasta)
    new_pd = pd.DataFrame(list(fasta), columns=['fasta'])
    new_pd.index += 1
    new_pd['fasta_numbering'] = new_pd.index

    return new_pd

def three_2_one(three):
    aa1to3 = {
        'A':'ALA', 'V':'VAL', 'F':'PHE', 'P':'PRO', 'M':'MET',
        'I':'ILE', 'L':'LEU', 'D':'ASP', 'E':'GLU', 'K':'LYS',
        'R':'ARG', 'S':'SER', 'T':'THR', 'Y':'TYR', 'H':'HIS',
        'C':'CYS', 'N':'ASN', 'Q':'GLN', 'W':'TRP', 'G':'GLY',
    }
    return aa1to3[three]

def pdbtofasta(pdb_file, selected_chain):
    nucleic_acids=['DA','DC','DG','DT','DI','A','C','G','U','I']
    aa3to1 = {
        'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
        'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
        'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
    }
    chain_pattern = re.compile("^ATOM\s{2,6}\d{1,5}[\sA-Z1-9]{10}([\w])")
    ca_pattern = re.compile("^ATOM\s{2,6}\d{1,5}\s{2}CA\s[\sA]([A-Z]{3})\s([\s\w])")
    nucleic_acid_pattern = re.compile("^ATOM\s{2,6}\d{1,5}\s{2}P [\sA]{3}([\s\w]{2})")
    chain_dict = dict()
    chain_list = []

    fp = open(pdb_file, 'r')
    for line in fp.read().splitlines():
        if line.startswith("ENDMDL"):
            break
        if ''.join(chain_pattern.findall(line)) == selected_chain:
            match_list = ca_pattern.findall(line)
            na_match_list = nucleic_acid_pattern.findall(line)
            if na_match_list:
                return 'nucleic_acid'
            elif match_list:
                resn = match_list[0][0]
                chain = match_list[0][1]
                try:
                    if chain in chain_dict:
                        chain_dict[chain] += aa3to1[resn]
                    else:
                        chain_dict[chain] = aa3to1[resn]
                        chain_list.append(chain)
                except:
                    pass
    fp.close()
    result = chain_dict.get(selected_chain[0])
    return result

def addChainLetter(letter, row):
    row = str(row)
    if row == 'nan':
        return '-'
    else:
        return letter.upper() + row

def numberIMGTCDR(IMGT_number):
    IMGT_number = str(IMGT_number)
    position = int()

    if IMGT_number[-1].isalpha():
        if IMGT_number == 'nan':
            return '-'
        else:
            position = int(IMGT_number[:-1])
    else:
        position = int(IMGT_number)

    if position < 27:
        return "FR1"
    if (27 <= position) & (position <= 38):
        return "CDR-1"
    if (39 <= position) & (position <= 55):
        return "FR2"
    if (56 <= position) & (position <= 65):
        return "CDR-2"
    if (66 <= position) & (position <= 104):
        return "FR3"
    if (105 <= position) & (position <= 117):
        return "CDR-3"
    if 118 <= position:
        return "FR4"
    else:
        return "-"

def getCDRnumbForHighlight_IMGT(fasta,fastanPDBMap,chainID,pdbID,chainType):
    CDR_dictionary = dict()
    
    annotated_seq = runDirectly(fasta, 'imgt')
    buf = StringIO(annotated_seq)

    temp_pd = pd.DataFrame()
    actual_fasta = fasta
    fasta = readFasta(fasta)

    if len(annotated_seq) < 30:
        return 'none'
    else:
        for each in buf.readlines():
            if not each.startswith('#'):
                stripted = each.strip()
                trimed = list(filter(None, stripted.split(' ')))
                if len(trimed) == 3:
                    if trimed[2] != '-':
                        temp_pd = temp_pd.append({'chainType':trimed[0],'IMGT':trimed[1],'{}_{}'.format(pdbID,chainID):trimed[2]},ignore_index=True)

                if len(trimed) == 4:
                    if trimed[3] != '-':
                        temp_pd = temp_pd.append({'chainType':trimed[0],'IMGT':trimed[1] + trimed[2],'{}_{}'.format(pdbID,chainID):trimed[3]},ignore_index=True)

            if each.startswith('# Domain 2') and not temp_pd.query('chainType == @chainType').empty:
                print("{}_{} has more than single {} chain domains.".format(pdbID,chainID,chainType))
                # break
                pass

        temp_pd = temp_pd.query('chainType == @chainType').reset_index(drop=True)
        temp_pd.index += 1
        _fasta = ''.join(temp_pd['{}_{}'.format(pdbID,chainID)].to_list())
        offset = RabinKarp_search(_fasta[0:10],actual_fasta)
        temp_pd.index += offset

        result = pd.merge(temp_pd, fasta, left_index=True, right_index=True, how='outer')
        result['CDR-{}'.format(chainType)] = result.apply(lambda row: numberIMGTCDR(row['IMGT']), axis=1)
        result['imgt_numbering'] = result.apply(lambda row: addChainLetter('{}'.format(chainType), row['IMGT']), axis=1)

        CDR_1 = list()
        CDR_2 = list()
        CDR_3 = list()
        for index, row in result.iterrows():
            if row['CDR-{}'.format(chainType)].startswith('CDR'):
                pdb_numb = fastanPDBMap.query('fasta_numb == {}'.format(row['fasta_numbering']))['pdb_numb'].values[0]
                imgt_number = row['imgt_numbering']
                amino_acid = three_2_one(row['fasta'])

            if row['CDR-{}'.format(chainType)].startswith('CDR-1'):
                # CDR_1.append('{}_{}'.format(chainID,pdb_numb))
                CDR_1.append('{}_{}/{}_{}'.format(chainID,pdb_numb,imgt_number,amino_acid))
            elif row['CDR-{}'.format(chainType)].startswith('CDR-2'):
                # CDR_2.append('{}_{}'.format(chainID,pdb_numb))
                CDR_2.append('{}_{}/{}_{}'.format(chainID,pdb_numb,imgt_number,amino_acid))
            elif row['CDR-{}'.format(chainType)].startswith('CDR-3'):
                # CDR_3.append('{}_{}'.format(chainID,pdb_numb))
                CDR_3.append('{}_{}/{}_{}'.format(chainID,pdb_numb,imgt_number,amino_acid))

        CDR_dictionary['CDR-{}1'.format(chainType)] = CDR_1
        CDR_dictionary['CDR-{}2'.format(chainType)] = CDR_2
        CDR_dictionary['CDR-{}3'.format(chainType)] = CDR_3

        return CDR_dictionary

def changeForm(row, column_name):
    resname = row[column_name].split('/')[1]
    chain = row[column_name].split('/')[0]
    atomname = row[column_name].split('/')[2]

    try:
        int(resname)
    except ValueError:
        resname = resname[0:len(resname) - 1] + "^" + resname[-1]

    return resname + ":" + chain + "." + atomname

def checkInterface_residue(row, column1_name, column2_name, interface_residue_list):
    chain_Atom1 = row[column1_name].split('/')[0]
    resnum_Atom1 = row[column1_name].split('/')[1]
    chain_Atom2 = row[column2_name].split('/')[0]
    resnum_Atom2 = row[column2_name].split('/')[1]

    try:
        int(resnum_Atom1)
    except ValueError:
        resnum_Atom1 = resnum_Atom1[0:len(resnum_Atom1) - 1] + "^" + resnum_Atom1[-1]

    try:
        int(resnum_Atom2)
    except ValueError:
        resnum_Atom2 = resnum_Atom2[0:len(resnum_Atom2) - 1] + "^" + resnum_Atom2[-1]

    set_intersection = set([chain_Atom1+'|'+resnum_Atom1,chain_Atom2+'|'+resnum_Atom2]) & set (interface_residue_list)

    # 2 for both, 1 for only one side.
    if len(set_intersection) > 0:
        return True
    else:
        return False

def changeFormForTarget(row, column_name):
    resname = row[column_name].split('/')[1]
    chain = row[column_name].split('/')[0]
    atomname = row[column_name].split('/')[2]

    try:
        int(resname)
    except ValueError:
        resname = resname[0:len(resname) - 1] + "^" + resname[-1]

    return resname + ":" + chain

def roundupList(given_list):
    result_list = []
    for each in given_list.split(','):
        result_list.append(round(float(each), 3))

    return ','.join(str(result_list)[1:-1].strip('\"').split(','))

def stringToListOfString(given_list):
    result_list = []

    try:

        for each in given_list.split(','):
            result_list.append(each.replace("'", "").replace('"', '').strip())

        return ','.join(str(result_list)[1:-1].split(',')).replace("'", "")

    except:
        print("exceptions")

        return result_list[1:-1]

def parallelize_dataframe(df, func, num_cores):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

######################################
######################################

def analyze_CDRs(input_pd):
    # two_groups_only=True
    # remove_redundant=False
    arpeggio_cols = ['Amide-Amide', 'Amide-Ring', 'Aromatic', 'Carbon-PI', 'Carbonyl', \
        'Cation-PI', 'Clash', 'Covalent', 'Donor-PI', 'Halogen', 'Halogen-PI', \
            'Hbond', 'Hydrophobic', 'Ionic', 'Metal', 'MetalSulphur-PI', 'PI-PI', \
                'Polar', 'Proximal', 'VDW', 'VDWClash', 'WeakHbond', 'WeakPolar']

    result_Arpeggio_pd = pd.DataFrame(columns=arpeggio_cols)

    for ind,row in input_pd.iterrows():
        fastanPDBMap = pd.DataFrame()
        resultH_pd = pd.DataFrame()
        resultL_pd = pd.DataFrame()
        try:
            pdb_id, HL_chains, ag_chains = row['ID'].split('_')
            pdb_file = os.path.join(pdb_dir,'{}.pdb'.format(pdb_id))

            ab_type = str()
            heavy_chain = str(row['Hchain'])
            light_chain = str(row['Lchain'])        
            
            # Define ab_type
            if HL_chains[0] == '!':
                ab_type = 'Fll'
            elif HL_chains[1] == '!':
                ab_type = 'Nanobody'
            elif HL_chains[0] == HL_chains[1]:
                ab_type = 'scFv'
            else:
                ab_type = 'Fab'

            for each_chain in getAllChains(pdb_file):
                fastanPDBMap = fastanPDBMap.append([getFASTAnPDBnumberMap(pdb_file,each_chain)],ignore_index=True, sort=False)
            fastanPDBMap.fasta_numb = fastanPDBMap.fasta_numb.astype(int)
            fastanPDBMap.pdb_numb = fastanPDBMap.pdb_numb.astype(str)
            fastanPDBMap.chain = fastanPDBMap.chain.astype(str)

            if not ab_type == 'Fll':
                _Hindex = '{}_{}'.format(pdb_id,heavy_chain)
                resultH_pd.loc[_Hindex,'ID'] = row['ID']
                resultH_pd.loc[_Hindex,'type'] = ab_type

                CDR_H_dict = getCDRnumbForHighlight_IMGT(pdbtofasta(pdb_file,heavy_chain),fastanPDBMap.query('chain == "{}"'.format(heavy_chain)),heavy_chain,pdb_id,'H')
                
                if CDR_H_dict != 'none':
                    resultH_pd.loc[_Hindex,'CDR-H1'] = ','.join(CDR_H_dict['CDR-H1'])
                    resultH_pd.loc[_Hindex,'nCDR-H1'] = len(CDR_H_dict['CDR-H1'])
                    resultH_pd.loc[_Hindex,'CDR-H2'] = ','.join(CDR_H_dict['CDR-H2'])
                    resultH_pd.loc[_Hindex,'nCDR-H2'] = len(CDR_H_dict['CDR-H2'])
                    resultH_pd.loc[_Hindex,'CDR-H3'] = ','.join(CDR_H_dict['CDR-H3'])
                    resultH_pd.loc[_Hindex,'nCDR-H3'] = len(CDR_H_dict['CDR-H3'])
                    resultH_pd.loc[_Hindex,'nCDR-H'] =  len(CDR_H_dict['CDR-H1']) + len(CDR_H_dict['CDR-H2']) + len(CDR_H_dict['CDR-H3'])
                else:
                    resultH_pd.loc[_Hindex,'CDR-H1'] = 'might_not_H_chain'
                    resultH_pd.loc[_Hindex,'nCDR-H1'] = 0
                    resultH_pd.loc[_Hindex,'CDR-H2'] = 'might_not_H_chain'
                    resultH_pd.loc[_Hindex,'nCDR-H2'] = 0
                    resultH_pd.loc[_Hindex,'CDR-H3'] = 'might_not_H_chain'
                    resultH_pd.loc[_Hindex,'nCDR-H3'] = 0
                    resultH_pd.loc[_Hindex,'nCDR-H'] = 0
                
                for each_CDR in ['CDR-H1','CDR-H2','CDR-H3']:
                    _residue_list = [each.split('/')[0] for each in resultH_pd.loc[_Hindex,each_CDR].split(',')]
                    residue_list = list()
                    for each in _residue_list:
                        _chainID,_resNum = each.split('_')
                        residue_list.append('{}|{}'.format(_chainID,_resNum))
                    _temp_pd = getInteractions_SelectedRes(pdb_file,HL_chains, ag_chains, residue_list,'CDR')
                    _temp_pd['ID'] = '{}_{}'.format(_temp_pd.index[0],each_CDR)
                    _temp_pd['type'] = resultH_pd.iloc[0]['type']
                    _temp_pd['{}'.format(each_CDR)] = resultH_pd.iloc[0][each_CDR]
                    _temp_pd['n{}'.format(each_CDR)] = resultH_pd.iloc[0]['n{}'.format(each_CDR)]
                    _temp_pd.set_index('ID',inplace=True)
                    result_Arpeggio_pd = result_Arpeggio_pd.append(_temp_pd)

            if not ab_type == 'Nanobody':
                _Lindex = '{}_{}'.format(pdb_id,light_chain)
                resultL_pd.loc[_Lindex,'ID'] = row['ID']
                resultL_pd.loc[_Lindex,'type'] = ab_type            

                CDR_L_dict = getCDRnumbForHighlight_IMGT(pdbtofasta(pdb_file,light_chain),fastanPDBMap.query('chain == "{}"'.format(light_chain)),light_chain,pdb_id,'L')

                if CDR_L_dict != 'none':
                    resultL_pd.loc[_Lindex,'CDR-L1'] = ','.join(CDR_L_dict['CDR-L1'])
                    resultL_pd.loc[_Lindex,'nCDR-L1'] = len(CDR_L_dict['CDR-L1'])
                    resultL_pd.loc[_Lindex,'CDR-L2'] = ','.join(CDR_L_dict['CDR-L2'])
                    resultL_pd.loc[_Lindex,'nCDR-L2'] = len(CDR_L_dict['CDR-L2'])
                    resultL_pd.loc[_Lindex,'CDR-L3'] = ','.join(CDR_L_dict['CDR-L3'])
                    resultL_pd.loc[_Lindex,'nCDR-L3'] = len(CDR_L_dict['CDR-L3'])
                    resultL_pd.loc[_Lindex,'nCDR-L'] =  len(CDR_L_dict['CDR-L1']) + len(CDR_L_dict['CDR-L2']) + len(CDR_L_dict['CDR-L3'])
                else:
                    resultL_pd.loc[_Lindex,'light_chain'] = 'might_not_L_chain'
                    resultL_pd.loc[_Lindex,'CDR-L1'] = 'might_not_L_chain'
                    resultL_pd.loc[_Lindex,'nCDR-L1'] = 0
                    resultL_pd.loc[_Lindex,'CDR-L2'] = 'might_not_L_chain'
                    resultL_pd.loc[_Lindex,'nCDR-L2'] = 0
                    resultL_pd.loc[_Lindex,'CDR-L3'] = 'might_not_L_chain'
                    resultL_pd.loc[_Lindex,'nCDR-L3'] = 0
                    resultL_pd.loc[_Lindex,'nCDR-L'] = 0

                for each_CDR in ['CDR-L1','CDR-L2','CDR-L3']:
                    _residue_list = [each.split('/')[0] for each in resultL_pd.loc[_Lindex,each_CDR].split(',')]
                    residue_list = list()
                    for each in _residue_list:
                        _chainID,_resNum = each.split('_')
                        residue_list.append('{}|{}'.format(_chainID,_resNum))
                    _temp_pd = getInteractions_SelectedRes(pdb_file,HL_chains, ag_chains, residue_list,'CDR')
                    _temp_pd['ID'] = '{}_{}'.format(_temp_pd.index[0],each_CDR)
                    _temp_pd['type'] = resultL_pd.iloc[0]['type']
                    _temp_pd['{}'.format(each_CDR)] = resultL_pd.iloc[0][each_CDR]
                    _temp_pd['n{}'.format(each_CDR)] = resultL_pd.iloc[0]['n{}'.format(each_CDR)]    
                    _temp_pd.set_index('ID',inplace=True)
                    result_Arpeggio_pd = result_Arpeggio_pd.append(_temp_pd)

        except Exception as e:
            print(row['ID'])
            print(e)
        
    return result_Arpeggio_pd

def getInteractions_SelectedRes(pdb_file, antibody_chain_string, antigen_chain_string, residue_list, type_of_run):
    two_groups_only=True
    remove_redundant=True

    antibody = set(antibody_chain_string)
    antigen = set(antigen_chain_string)

    contacts_file = pdb_file[:-4] + "_whole.contacts"  # for contacts
    ari_file = pdb_file[:-4] + "_whole.ari"  # for carbonpi, cationpi, donorpi, halogenpi, metsulphurpi
    ri_file = pdb_file[:-4] + "_whole.ri"  # for ring-ring
    amam_file = pdb_file[:-4] + "_whole.amam"  # for amideamide
    amri_file = pdb_file[:-4] + "_whole.amri"  # for amidering

    # Reformatted Arpeggio result files
    new_contacts_file = pdb_file[:-4]+ "_trimed.contacts"
    new_ari_file = pdb_file[:-4]+ "_trimed.ari"
    new_ri_file = pdb_file[:-4]+ "_trimed.ri"
    new_amam_file = pdb_file[:-4]+ "_trimed.amam"
    new_amri_file = pdb_file[:-4]+ "_trimed.amri"

    all_interacting_residues = []
    all_pseudo_coords = []

    type_of_pi = ['carbonpi', 'cationpi', 'donorpi', 'halogenpi', 'metsulphurpi', 'pipi', 'amideamide', 'amidering']
    all_PIinteractions = dict.fromkeys(type_of_pi, list())

    try:
        if os.path.getsize(contacts_file) > 0:
            contacts_pd = pd.read_csv(contacts_file, sep='\t', header=None)
            contacts_pd.columns = ['Atom1', 'Atom2', 'Clash', 'Covalent', 'VDWClash', 'VDW', 'Proximal', 'Hbond',
                                    'WeakHbond', 'Halogen', 'Ionic', 'Metal', 'Aromatic', 'Hydrophobic', 'Carbonyl',
                                    'Polar', 'WeakPolar', 'int_type']
            contacts_pd['given_residue'] = contacts_pd.apply(lambda row: checkInterface_residue(row,'Atom1','Atom2',residue_list), axis=1)
            # contacts_pd[contacts_pd['given_residue']].drop('given_residue',axis=1).to_csv(new_contacts_file,sep='\t',header=None,index=False)
            contacts_pd['Atom1'] = contacts_pd.apply(lambda row: changeForm(row, 'Atom1'), axis=1)
            contacts_pd['Atom2'] = contacts_pd.apply(lambda row: changeForm(row, 'Atom2'), axis=1)
            contacts_pd['test_query'] = contacts_pd.apply(
                lambda row: set(row['Atom1'].split(':')[1].split('.')[0] + row['Atom2'].split(':')[1].split('.')[0]),
                axis=1)
            contacts_pd['antibody'] = contacts_pd.apply(lambda row: len(row['test_query'].intersection(antibody)) != 0, axis=1)
            contacts_pd['antigen'] = contacts_pd.apply(lambda row: len(row['test_query'].intersection(antigen)) != 0, axis=1)
            contacts_pd['is_Ab_Ag_int'] = contacts_pd.apply(
                lambda row: row['antibody'] == True and row['antigen'] == True, axis=1)
            contacts_pd.drop(['test_query'],inplace=True,axis=1)

            if two_groups_only == True:
                contacts_pd = contacts_pd.query('given_residue == True and is_Ab_Ag_int == True')  # This part distinguish AbAg interaction.
            else:
                pass

            contacts_sum = contacts_pd.sum().drop(['Atom1', 'Atom2', 'int_type', 'antibody', 'antigen', 'is_Ab_Ag_int','given_residue'])
            dict_contacts = contacts_sum.to_dict()

        if os.path.getsize(ari_file) > 0:

            carbonpi = []
            cationpi = []
            donorpi = []
            halogenpi = []
            metsulphurpi = []

            ari_pd = pd.read_csv(ari_file, sep='\t', header=None)
            ari_pd.columns = ['target_atom', 'unknown', 'pseudo_residue', 'pseudo_coord', 'int_type', 'target_type',
                                'pseudo_type']

            ari_pd['given_residue'] = ari_pd.apply(
                lambda row: checkInterface_residue(row,'target_atom','pseudo_residue',residue_list), axis=1)
            ari_pd['target_residue'] = ari_pd.apply(lambda row: changeFormForTarget(row, 'target_atom'), axis=1)
            ari_pd['target_atom'] = ari_pd.apply(lambda row: changeForm(row, 'target_atom'), axis=1)
            ari_pd['pseudo_residue'] = ari_pd.apply(lambda row: changeForm(row, 'pseudo_residue')[:-1], axis=1)
            ari_pd['pseudo_coord'] = ari_pd.apply(
                lambda row: roundupList(','.join(row['pseudo_coord'][1:-1].split(','))), axis=1)  # for rounding up
            ari_pd['int_type'] = ari_pd.apply(lambda row: stringToListOfString(row['int_type'][2:-2]),
                                                axis=1)  # for removing quote marks
            ari_pd.assign(int_type=ari_pd.int_type.str.split(','))
            ari_pd = (ari_pd.set_index(ari_pd.columns.drop('int_type', 1).tolist())
                            .int_type.str.split(',', expand=True)
                            .stack()
                            .reset_index()
                            .rename(columns={0: 'int_type'})
                            .loc[:, ari_pd.columns]
                            )
            ari_pd.index += 1
            ari_pd = ari_pd.drop(['target_type', 'pseudo_type'], axis=1)
            ari_pd['test_query'] = ari_pd.apply(
                lambda row: set(row['pseudo_residue'].split(':')[-1] + row['target_residue'].split(':')[-1]), axis=1)
            ari_pd['antibody'] = ari_pd.apply(lambda row: len(row['test_query'].intersection(antibody)) != 0, axis=1)
            ari_pd['antigen'] = ari_pd.apply(lambda row: len(row['test_query'].intersection(antigen)) != 0, axis=1)
            ari_pd['is_Ab_Ag_int'] = ari_pd.apply(lambda row: row['antibody'] == True and row['antigen'] == True, axis=1)

            if two_groups_only == True:
                ari_pd = ari_pd.query('given_residue == True and is_Ab_Ag_int == True')
            else:
                pass

            if ari_pd.empty:
                pass
            else:
                grouped = ari_pd.groupby('int_type')

                for group_name, table in grouped:
                    if group_name == 'CARBONPI':
                        for index, row in table.iterrows():
                            carbonpi.append([row.tolist()[0], row.tolist()[3]])

                    if group_name == 'CATIONPI':
                        for index, row in table.iterrows():
                            cationpi.append([row.tolist()[0], row.tolist()[3]])

                    if group_name == 'DONORPI':
                        for index, row in table.iterrows():
                            donorpi.append([row.tolist()[0], row.tolist()[3]])

                    if group_name == 'HALOGENPI':
                        for index, row in table.iterrows():
                            halogenpi.append([row.tolist()[0], row.tolist()[3]])
                    if group_name == 'METSULPHURPI':
                        for index, row in table.iterrows():
                            metsulphurpi.append([row.tolist()[0], row.tolist()[3]])

                # save each of interaction into separate list.
                all_PIinteractions['carbonpi'] = carbonpi
                all_PIinteractions['cationpi'] = cationpi
                all_PIinteractions['donorpi'] = donorpi
                all_PIinteractions['halogenpi'] = halogenpi
                all_PIinteractions['metsulphurpi'] = metsulphurpi

                # for showing interacting residues
                all_interacting_residues.extend(list(set(ari_pd['target_residue'].tolist())))
                all_interacting_residues.extend(list(set(ari_pd['pseudo_residue'].tolist())))
                # print(all_interacting_residues)

                # for drawing pseudo_coord
                # all_pseudo_coords.extend(list(set(ari_pd['pseudo_coord'].tolist())))

        # ring_ring(ri_file)
        if os.path.getsize(ri_file) > 0:
            pipi = []

            ri_pd = pd.read_csv(ri_file, sep='\t', header=None)
            ri_pd.columns = ['ring_num1', 'pseudo_residue1', 'pseudo_coord1', 'ring_num2', 'pseudo_residue2',
                            'pseudo_coord2', 'int_type', 'target_type',
                            'pseudo_type']
            ri_pd['given_residue'] = ri_pd.apply(
                lambda row: checkInterface_residue(row, 'pseudo_residue1', 'pseudo_residue2', residue_list), axis=1)
            ri_pd = ri_pd.drop(['target_type', 'pseudo_type'], axis=1)
            ri_pd['pseudo_residue1'] = ri_pd.apply(lambda row: changeForm(row, 'pseudo_residue1')[:-1], axis=1)
            ri_pd['pseudo_residue2'] = ri_pd.apply(lambda row: changeForm(row, 'pseudo_residue2')[:-1], axis=1)
            ri_pd['pseudo_coord1'] = ri_pd.apply(
                lambda row: roundupList(','.join(row['pseudo_coord1'][1:-1].split(','))), axis=1)
            ri_pd['pseudo_coord2'] = ri_pd.apply(
                lambda row: roundupList(','.join(row['pseudo_coord2'][1:-1].split(','))), axis=1)
            ri_pd['test_query'] = ri_pd.apply(
                lambda row: set(row['pseudo_residue1'].split(':')[-1] + row['pseudo_residue2'].split(':')[-1]), axis=1)
            ri_pd['antibody'] = ri_pd.apply(lambda row: len(row['test_query'].intersection(antibody)) != 0, axis=1)
            ri_pd['antigen'] = ri_pd.apply(lambda row: len(row['test_query'].intersection(antigen)) != 0, axis=1)
            ri_pd['is_Ab_Ag_int'] = ri_pd.apply(lambda row: row['antibody'] == True and row['antigen'] == True, axis=1)

            if two_groups_only == True:
                ri_pd = ri_pd.query('given_residue == True and is_Ab_Ag_int == True')
            else:
                pass

            if ri_pd.empty:
                pass
            else:

                try:
                    if remove_redundant == 'TRUE':
                        ri_pd['redundancy'] = ri_pd['ring_num1'] > ri_pd['ring_num2']
                        ri_pd = ri_pd[~ri_pd['redundancy']]

                    for index, row in ri_pd.iterrows():
                        pipi.append([row.tolist()[2], row.tolist()[5]])

                    # save interactions into one list
                    all_PIinteractions['pipi'] = pipi

                    # for showing interacting residues
                    all_interacting_residues.extend(list(set(ri_pd['pseudo_residue1'].tolist())))
                    all_interacting_residues.extend(list(set(ri_pd['pseudo_residue2'].tolist())))

                    # for drawing pseudo_coord
                    # all_pseudo_coords.extend(list(set(ri_pd['pseudo_coord1'].tolist())))
                    # all_pseudo_coords.extend(list(set(ri_pd['pseudo_coord2'].tolist())))

                except ValueError:
                    pass

        # amide_amide(amam_file)
        if os.path.getsize(amam_file) > 0:
            amideamide = []

            amam_pd = pd.read_csv(amam_file, sep='\t', header=None)

            amam_pd.index += 1
            amam_pd.columns = ['amide_num1', 'pseudo_residue1', 'pseudo_coord1', 'amide_num2', 'pseudo_residue2',
                                'pseudo_coord2',
                                'int_type', 'target_type',
                                'pseudo_type']
            amam_pd['given_residue'] = amam_pd.apply(
                lambda row: checkInterface_residue(row, 'pseudo_residue1', 'pseudo_residue2', residue_list), axis=1)
            # amam_pd[amam_pd['given_residue']].drop('given_residue', axis=1).to_csv(new_amam_file,
            #                                                                              sep='\t', header=None,
            #                                                                              index=False)
            amam_pd = amam_pd.drop(['target_type', 'pseudo_type'], axis=1)
            amam_pd['pseudo_residue1'] = amam_pd.apply(lambda row: changeForm(row, 'pseudo_residue1')[:-1], axis=1)
            amam_pd['pseudo_residue2'] = amam_pd.apply(lambda row: changeForm(row, 'pseudo_residue2')[:-1], axis=1)
            amam_pd['pseudo_coord1'] = amam_pd.apply(
                lambda row: roundupList(','.join(row['pseudo_coord1'][1:-1].split(','))), axis=1)
            amam_pd['pseudo_coord2'] = amam_pd.apply(
                lambda row: roundupList(','.join(row['pseudo_coord2'][1:-1].split(','))), axis=1)
            amam_pd['test_query'] = amam_pd.apply(
                lambda row: set(row['pseudo_residue1'].split(':')[-1] + row['pseudo_residue2'].split(':')[-1]), axis=1)
            amam_pd['antibody'] = amam_pd.apply(lambda row: len(row['test_query'].intersection(antibody)) != 0, axis=1)
            amam_pd['antigen'] = amam_pd.apply(lambda row: len(row['test_query'].intersection(antigen)) != 0, axis=1)
            amam_pd['is_Ab_Ag_int'] = amam_pd.apply(lambda row: row['antibody'] == True and row['antigen'] == True, axis=1)

            if two_groups_only == True:
                amam_pd = amam_pd.query('given_residue == True and is_Ab_Ag_int == True')

            else:
                pass

            if amam_pd.empty:
                pass
            else:
                try:

                    if remove_redundant == 'TRUE':
                        amam_pd['redundancy'] = amam_pd['amide_num1'] > amam_pd['amide_num2']
                        amam_pd = amam_pd[~amam_pd['redundancy']]

                    for index, row in amam_pd.iterrows():
                        amideamide.append([row.tolist()[2], row.tolist()[5]])

                    # save interactions into one list
                    all_PIinteractions['amideamide'] = amideamide

                    # for showing interacting residues
                    all_interacting_residues.extend(list(set(amam_pd['pseudo_residue1'].tolist())))
                    all_interacting_residues.extend(list(set(amam_pd['pseudo_residue2'].tolist())))

                    # for drawing pseudo_coord
                    # all_pseudo_coords.extend(list(set(amam_pd['pseudo_coord1'].tolist())))
                    # all_pseudo_coords.extend(list(set(amam_pd['pseudo_coord2'].tolist())))

                except ValueError:
                    pass

        # amide_ring(amri_file)
        if os.path.getsize(amri_file) > 0:

            amidering = []
            amri_pd = pd.read_csv(amri_file, sep='\t', header=None)

            amri_pd.index += 1
            amri_pd.columns = ['unknown', 'pseudo_residue1', 'pseudo_coord1', 'unknown2', 'pseudo_residue2',
                                'pseudo_coord2',
                                'int_type', 'target_type',
                                'pseudo_type']
            amri_pd['given_residue'] = amri_pd.apply(
                lambda row: checkInterface_residue(row, 'pseudo_residue1', 'pseudo_residue2', residue_list),
                axis=1)
            amri_pd = amri_pd.drop(['target_type', 'pseudo_type'], axis=1)
            amri_pd['pseudo_residue1'] = amri_pd.apply(lambda row: changeForm(row, 'pseudo_residue1')[:-1], axis=1)
            amri_pd['pseudo_residue2'] = amri_pd.apply(lambda row: changeForm(row, 'pseudo_residue2')[:-1], axis=1)
            amri_pd['pseudo_coord1'] = amri_pd.apply(
                lambda row: roundupList(','.join(row['pseudo_coord1'][1:-1].split(','))), axis=1)
            amri_pd['pseudo_coord2'] = amri_pd.apply(
                lambda row: roundupList(','.join(row['pseudo_coord2'][1:-1].split(','))), axis=1)
            amri_pd['test_query'] = amri_pd.apply(
                lambda row: set(row['pseudo_residue1'].split(':')[-1] + row['pseudo_residue2'].split(':')[-1]), axis=1)
            amri_pd['antibody'] = amri_pd.apply(lambda row: len(row['test_query'].intersection(antibody)) != 0, axis=1)
            amri_pd['antigen'] = amri_pd.apply(lambda row: len(row['test_query'].intersection(antigen)) != 0, axis=1)
            amri_pd['is_Ab_Ag_int'] = amri_pd.apply(lambda row: row['antibody'] == True and row['antigen'] == True, axis=1)

            if two_groups_only == True:
                amri_pd = amri_pd.query('given_residue == True and is_Ab_Ag_int == True')
            else:
                pass

            if amri_pd.empty:
                pass
            else:

                for index, row in amri_pd.iterrows():
                    amidering.append([row.tolist()[2], row.tolist()[5]])

                # save interactions into one list
                all_PIinteractions['amidering'] = amidering

                # for showing interacting residues
                all_interacting_residues.extend(list(set(amri_pd['pseudo_residue1'].tolist())))
                all_interacting_residues.extend(list(set(amri_pd['pseudo_residue2'].tolist())))

                # for drawing pseudo_coord
                # all_pseudo_coords.extend(list(set(amri_pd['pseudo_coord1'].tolist())))
                # all_pseudo_coords.extend(list(set(amri_pd['pseudo_coord2'].tolist())))

        all_interacting_residues = list(set(all_interacting_residues))
        all_pseudo_coords = list(set(all_pseudo_coords))

        dict_PI = {'PI-PI': len(all_PIinteractions['pipi']),
                    'Carbon-PI': len(all_PIinteractions['carbonpi']),
                    'Cation-PI': len(all_PIinteractions['cationpi']),
                    'Donor-PI': len(all_PIinteractions['donorpi']),
                    'MetalSulphur-PI': len(all_PIinteractions['metsulphurpi']),
                    'Halogen-PI': len(all_PIinteractions['halogenpi']),
                    'Amide-Amide': len(all_PIinteractions['amideamide']),
                    'Amide-Ring': len(all_PIinteractions['amidering'])}

        result = dict_contacts.copy()
        result.update(dict_PI)

        final_result = pd.DataFrame([result.values()], columns=result.keys())
        final_result['ID'] = os.path.basename(pdb_file)[:-4]
        final_result.set_index('ID',inplace=True)
        final_result = final_result.reindex(sorted(final_result.columns),axis=1)

        return final_result
    except Exception as e:
        print(e)

def analyze_Interface(input_pd):
    result_pd = pd.DataFrame()

    # Parallellise
    for ind,row in input_pd.iterrows():
        pdb_id,ab_chains,ag_chains = row['ID'].split('_')
        pdb_file = os.path.join(pdb_dir,'{}.pdb'.format(row['ID']))

        epitope_list = list()
        paratope_list = list()

        epitope_file = os.path.join(pdb_file[0:-4]+"_epitope.txt")
        paratope_file = os.path.join(pdb_file[0:-4]+"_paratope.txt")

        epitope_list = open(epitope_file).read().replace('_','|').split(',')
        paratope_list = open(paratope_file).read().replace('_','|').split(',')
        interface_list = epitope_list + paratope_list

        try:
            result_pd = result_pd.append(getInteractions_SelectedRes(pdb_file, ab_chains, ag_chains, interface_list, 'interface'))
        except Exception as e:
            print("Got error:{}".format(row['ID']))
            print(e)

    return result_pd

def do_int_analysis(input_pd,num_of_cores):
    output_fname = 'result_Arpeggio_interface.csv'
    result_pd = pd.DataFrame()

    result_pd = result_pd.append(parallelize_dataframe(input_pd, analyze_Interface, num_of_cores))
    result_pd.to_csv(output_fname, sep=',',index_label='ID')

def do_CDR_analysis(input_pd,num_of_cores):
    output_fname_Arpeggio = 'result_Arpeggio_CDRs_ints.csv'
    result_Arpeggio_pd = pd.DataFrame()
    
    result_Arpeggio_pd = parallelize_dataframe(input_pd, analyze_CDRs, num_of_cores)
    result_Arpeggio_pd.to_csv(output_fname_Arpeggio, sep=';',index_label='ID')

def main():
    parser = argparse.ArgumentParser(description="This is a script for analysing Arpeggio results.")
    parser.add_argument('input_tsv',type=str,\
        help='input_cluster pandas tsv')
    parser.add_argument('core',type=str,\
        default=4,
        help='Choose the number of cores for parallelization')

    args = parser.parse_args()
    input_tsv = args.input_tsv
    num_of_cores = args.core
    input_pd = pd.read_csv(input_tsv,sep='\t')

    do_int_analysis(input_pd,num_of_cores)
    do_CDR_analysis(input_pd,num_of_cores)

if __name__ == "__main__":
    main()

