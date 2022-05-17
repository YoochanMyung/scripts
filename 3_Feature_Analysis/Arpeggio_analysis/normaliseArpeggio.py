# ********************************************************
# *   ----------------------------------------------     *
# * Yoochan Myung                                        *
# * The University of Melbourne                          *
# * yuchan.m@gmail.com                                   *
# *   ----------------------------------------------     *
# ********************************************************
import argparse
import pandas as pd
import sys

def overall_CDRs(arpeggio_result_pd):
    result_pd = pd.DataFrame()

    for ind, row in arpeggio_result_pd.iterrows():
        arpeggio_result_pd.loc[ind,'groupID'] = ind.split('_CDR')[0]
    
    for ind, group in arpeggio_result_pd.groupby('groupID',dropna=False):
        _temp_pd = pd.DataFrame()
        _temp_pd = group.drop(['type','groupID','CDR-L1','CDR-L2','CDR-L3','CDR-H1','CDR-H2','CDR-H3'],axis=1).sum()
        _temp_pd['ID'] = ind
        _temp_pd['type'] = group['type'].to_list()[0]
        result_pd = result_pd.append(_temp_pd,ignore_index=True)
    result_pd.set_index('ID',inplace=True)
    result_pd = result_pd.convert_dtypes()
    
    return result_pd

def specific_CDR(arpeggio_result_pd):
    result_pd = pd.DataFrame()

    for ind, row in arpeggio_result_pd.iterrows():
        arpeggio_result_pd.loc[ind,'groupID'] = ind.split('_CDR')[0]
        arpeggio_result_pd.loc[ind,'CDR_group1'] = ind.split('CDR-')[1]
        arpeggio_result_pd.loc[ind,'CDR_group2'] = ind.split('CDR-')[1][0]
    
    arpeggio_result_pd['nCDR'] = arpeggio_result_pd[['nCDR-L1','nCDR-L2','nCDR-L3','nCDR-H1','nCDR-H2','nCDR-H3']].sum(axis=1)
    arpeggio_result_pd.drop(['CDR-L1','CDR-L2','CDR-L3','CDR-H1','CDR-H2','CDR-H3','nCDR-L1','nCDR-L2','nCDR-L3','nCDR-H1','nCDR-H2','nCDR-H3'], axis=1, inplace= True)
    result_pd = arpeggio_result_pd.convert_dtypes()
    
    return result_pd


def normalise_interface(arpeggio_result_pd, bsa_result_pd):
    result_pd = pd.DataFrame()
    merged_pd = arpeggio_result_pd.join(bsa_result_pd)
    for ind, row in merged_pd.iterrows():
        for col in row.keys():
            if not isinstance(row[col],str) and col != 'BSA':
                result_pd.loc[ind,col] = row[col]/row['BSA']*100

        result_pd.loc[ind,'BSA'] = row['BSA']
        result_pd.loc[ind,'type'] = row['type']
    result_pd.drop(['sasaTotal','sasaAb','sasaAg'],axis=1,inplace=True)
    result_pd = result_pd.convert_dtypes()

    return result_pd

def main():
    parser = argparse.ArgumentParser(description="This is a script for getting normalised Arpeggio interactions based on BSA.")
    parser.add_argument('arpeggio_csv',type=str,\
        help='Result file of Arpeggio interface/CDR analysis')
    parser.add_argument('bsa_csv',type=str,\
        help='Result file of BSA analysis')        
    parser.add_argument('--analysis',type=str,\
        choices=['interface','CDR'],
        default='interface',
        help='choose between interface or CDR')

    args = parser.parse_args()
    sep = str()
    bsa_result_pd = pd.read_csv(args.bsa_csv,index_col='ID')

    if args.analysis == 'interface':
        sep = ','
        arpeggio_result_pd = pd.read_csv(args.arpeggio_csv,sep=sep,index_col='ID')
        normalise_interface(arpeggio_result_pd, bsa_result_pd).to_csv('{}_normalised_by_BSA.csv'.format(args.arpeggio_csv), index_label='ID', sep=';')
        print('{}_normalised_by_BSA.csv was successfully created.'.format(args.arpeggio_csv))
    else:
        sep = ';'
        arpeggio_result_pd = pd.read_csv(args.arpeggio_csv,sep=sep,index_col='ID')
        overall_CDRs(arpeggio_result_pd).to_csv('{}_overall_CDRs.csv'.format(args.arpeggio_csv), index_label='ID', sep=';')
        specific_CDR(arpeggio_result_pd).to_csv('{}_specific_CDR.csv'.format(args.arpeggio_csv), index_label='ID', sep=';')
        print('{}_overall_CDRs.csv was successfully created.'.format(args.arpeggio_csv))
        print('{}_specific_CDR.csv was successfully created.'.format(args.arpeggio_csv))

if __name__ == "__main__":
    main()

# python normaliseArpeggio.py result_Arpeggio_interface.csv result_BSA.csv
# python normaliseArpeggio.py result_Arpeggio_CDRs_ints.csv result_BSA.csv --analysis CDR
