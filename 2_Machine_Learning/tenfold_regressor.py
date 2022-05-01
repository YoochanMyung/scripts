# ********************************************************
# *               University of Melbourne                *
# *   ----------------------------------------------     *
# * Yoochan Myung - ymyung@student.unimelb.edu.au        *
# *   ----------------------------------------------     *
# ********************************************************
import pandas as pd
import numpy as np
import scipy as sp
import time
import sys
import os
import re
import argparse
from math import sqrt
from scipy import stats
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.tree import export_graphviz

timestr = time.strftime("%Y%m%d_%H%M%S")

def runML(algorithm, fname, train_set, blind_set, output_dir, target_label, error_type, n_cores,
          num_of_shuffling, random_state, save_model):
    result_ML = pd.DataFrame()
    blind_kfold_result = pd.DataFrame()
    blind_result = pd.DataFrame()

    distance_analysis = False
    train_ID = pd.DataFrame(train_set['ID'])

    overall_train_result_filename = '{}_{}_{}_{}_10CV_train_scatter_plot.csv'.format(timestr,fname,algorithm,str(random_state))
    train_label = np.array(train_set[target_label])
    train_set = train_set.drop('ID', axis=1)
    train_features = train_set.drop(target_label, axis=1)
    headers = list(train_features.columns.values)

    if algorithm == 'GB':
        regressor = GradientBoostingRegressor(n_estimators=300, random_state=1)

    elif (algorithm == 'XGBOOST'):
        regressor = XGBRegressor(objective ='reg:squarederror',n_estimators=300, random_state=1, n_jobs=n_cores)

    elif (algorithm == 'RF'):
        regressor = RandomForestRegressor(n_estimators=300, random_state=1, n_jobs=n_cores)

    elif (algorithm == 'ExtraTrees'):
        regressor = ExtraTreesRegressor(n_estimators=300, random_state=1, n_jobs=n_cores)

    elif (algorithm == 'GAUSSIAN'):
        regressor = GaussianProcessRegressor(random_state=1)

    elif (algorithm == 'ADABOOST'):
        regressor = AdaBoostRegressor(n_estimators=300, random_state=1)

    elif (algorithm == 'KNN'):
        regressor = KNeighborsRegressor(n_neighbors=5, n_jobs=n_cores)

    elif (algorithm == 'SVR'):
        regressor = svm.SVR(kernel='rbf')

    elif (algorithm == 'NEURAL'):
        regressor = MLPRegressor(random_state=1)

    else:
        print("Algorithm Selection ERROR!!")
        sys.exit()

    # 10-fold cross validation
    predicted_n_actual_pd = pd.DataFrame(columns=['ID', 'predicted', 'actual', 'fold'])
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    fold = 1
    for train, test in kf.split(train_features):
        # train and test number(row number) are based on train_features.
        # For example, if '1' from train or test, it would be '1' in train_features

        train_cv_features, test_cv_features, train_cv_label, test_cv_label = train_features.iloc[train], \
                                                                             train_features.iloc[test], train_label[
                                                                                 train], train_label[test]

        if algorithm == 'GB':
            temp_regressor = GradientBoostingRegressor(n_estimators=300, random_state=1)

        elif (algorithm == 'XGBOOST'):
            temp_regressor = XGBRegressor(objective ='reg:squarederror',n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'RF'):
            temp_regressor = RandomForestRegressor(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'ExtraTrees'):
            temp_regressor = ExtraTreesRegressor(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'GAUSSIAN'):
            temp_regressor = GaussianProcessRegressor(random_state=1)

        elif (algorithm == 'ADABOOST'):
            temp_regressor = AdaBoostRegressor(n_estimators=300, random_state=1)

        elif (algorithm == 'KNN'):
            temp_regressor = KNeighborsRegressor(n_neighbors=5, n_jobs=n_cores)

        elif (algorithm == 'SVR'):
            temp_regressor = svm.SVR(kernel='rbf')

        elif (algorithm == 'NEURAL'):
            temp_regressor = MLPRegressor(random_state=1)
        temp_regressor.fit(train_cv_features, train_cv_label)
        temp_prediction = temp_regressor.predict(test_cv_features)
        temp_new_pd = pd.DataFrame(
            np.column_stack([train_features.index[test].tolist(), test_cv_label, temp_prediction]),
            columns=['ID', 'actual', 'predicted'])
        temp_new_pd['fold'] = fold
        predicted_n_actual_pd = predicted_n_actual_pd.append(temp_new_pd, ignore_index=True, sort=True)

        if not isinstance(blind_set,str):
            temp_blind_kfold_result,aa = run_blind(algorithm,fold,headers,target_label,temp_regressor,blind_set,1,0,error_type)
            blind_kfold_result = blind_kfold_result.append(temp_blind_kfold_result,ignore_index=True)

        fold += 1

    # Get Pearson's correlation of 10-fold C.V. on both original and 10% outlier removed dataset.
    predicted_n_actual_pd.ID += 1
    predicted_n_actual_pd = predicted_n_actual_pd.sort_values('ID')
    predicted_n_actual_pd['old_ID'] = train_ID['ID'].tolist()

    predicted_n_actual_pd['ID'] = predicted_n_actual_pd['ID'].astype(int)
    predicted_n_actual_pd = predicted_n_actual_pd.sort_values('ID')
    predicted_n_actual_pd['error'] = abs(predicted_n_actual_pd['predicted'] - predicted_n_actual_pd['actual'])

    train_slope, train_intercept, train_r_value, train_p_value, train_std_err = stats.linregress(
        predicted_n_actual_pd['actual'], predicted_n_actual_pd['predicted'])
    predicted_n_actual_pd['dist.to_bestfit'] = abs(
        (train_slope * predicted_n_actual_pd['actual']) - predicted_n_actual_pd['predicted'] + train_intercept) / sqrt(
        pow(train_slope, 2) + 1)

    if error_type == 'Absolute':
        predicted_n_actual_pd = predicted_n_actual_pd.sort_values('error', ascending=False)

    elif error_type == 'BestFit':
        predicted_n_actual_pd = predicted_n_actual_pd.sort_values('dist.to_bestfit', ascending=False)

    # Separate results for distance-based analysis
    ten_percent_of_train = int(round(len(predicted_n_actual_pd['ID']) * 0.1))
    train_non_outliers = predicted_n_actual_pd[ten_percent_of_train:]
    train_outliers = predicted_n_actual_pd[:ten_percent_of_train]
    train_outliers.insert(len(train_outliers.columns), 'outlier', 'True')
    # Pearson's correlation
    train_w_outliers_pearsons = round(
        sp.stats.pearsonr(predicted_n_actual_pd['actual'], predicted_n_actual_pd['predicted'])[0], 3)
    train_wo_outliers_pearsons = round(
        sp.stats.pearsonr(train_non_outliers['actual'], train_non_outliers['predicted'])[0], 3)
    # Spearman's rank-order correlation
    train_w_outliers_spearman = round(
        sp.stats.spearmanr(predicted_n_actual_pd['actual'], predicted_n_actual_pd['predicted'])[0], 3)
    train_wo_outliers_spearman = round(
        sp.stats.spearmanr(train_non_outliers['actual'], train_non_outliers['predicted'])[0], 3)
    # Kendall's tau
    train_w_outliers_kendalltau = round(
        sp.stats.kendalltau(predicted_n_actual_pd['actual'], predicted_n_actual_pd['predicted'])[0], 3)
    train_wo_outliers_kendalltau = round(
        sp.stats.kendalltau(train_non_outliers['actual'], train_non_outliers['predicted'])[0], 3)
    # MSE
    train_w_outliers_mse = round(
        mean_squared_error(predicted_n_actual_pd['actual'], predicted_n_actual_pd['predicted']), 2)
    train_wo_outliers_mse = round(
        mean_squared_error(train_non_outliers['actual'], train_non_outliers['predicted']), 2)
    # RMSE
    train_w_outliers_rmse = round(
        sqrt(mean_squared_error(predicted_n_actual_pd['actual'], predicted_n_actual_pd['predicted'])), 2)
    train_wo_outliers_rmse = round(
        sqrt(mean_squared_error(train_non_outliers['actual'], train_non_outliers['predicted'])), 2)

    # if outlier, it will have 'true' else 'false'
    overall_training = pd.merge(predicted_n_actual_pd, train_outliers[['ID', 'outlier']], how='left',
                                on=['ID', 'ID'])
    overall_training = overall_training.fillna(False)
    overall_training['dist.to_bestfit'] = overall_training['dist.to_bestfit'].round(2)
    overall_training['predicted'] = overall_training['predicted'].round(2)
    overall_training['error'] = overall_training['error'].round(2)

    # Preparation for Evaluation/Assessment steps
    model = regressor.fit(train_features, train_label)
    if num_of_shuffling == random_state and save_model != 'False':
        model_filename = '{}_{}_{}_model.sav'.format(timestr,fname,algorithm)
        model_feature_list = '{}_{}_{}_feature_order.sav'.format(timestr,fname,algorithm)
        joblib.dump(model, os.path.join(output_dir,model_filename))
        joblib.dump(train_features.columns, os.path.join(model_feature_list))

    if error_type == 'Absolute':
        overall_training = overall_training.sort_values('error', ascending=False)
        overall_training.rename(columns={"outlier": "outlier(Absolute)"}, inplace=True)

    elif error_type == 'BestFit':
        overall_training.rename(columns={"outlier": "outlier(BestFit)"}, inplace=True)
        overall_training = overall_training.sort_values('dist.to_bestfit', ascending=False)
    else:
        print("You gave wrong error type.")
        sys.exit()

    overall_training.to_csv(os.path.join(output_dir,overall_train_result_filename), index=False)

    if not isinstance(blind_set,str):
        blind_kfold_result_filename = '{}_{}_{}_{}_kfold_blind_results.csv'.format(timestr,fname,algorithm,str(random_state))
        blind_kfold_result.to_csv(os.path.join(output_dir,blind_kfold_result_filename), index=False)

        if num_of_shuffling == random_state:
            blind_scatter_filename = '{}_{}_{}_10CV_blind_scatter_plot.csv'.format(timestr,fname,algorithm)
            blind_result, blind_scatter = run_blind(algorithm,fname,headers,target_label,regressor,blind_set,train_slope,train_intercept,error_type)
            # blind_result.to_csv(os.path.join(output_dir,blind_result_filename), index=False)

            result_ML = result_ML.append(
            {'filename': fname,
             'Training(Pearson)': train_w_outliers_pearsons,
             'Training(Spearman)': train_w_outliers_spearman,
             'Training(Kendall)': train_w_outliers_kendalltau,
             'Training(MSE)': train_w_outliers_mse,
             'Training(RMSE)': train_w_outliers_rmse,
             'Training(Pearson_90%)': train_wo_outliers_pearsons,
             'Training(Spearman_90%)': train_wo_outliers_spearman,
             'Training(Kendall_90%)': train_wo_outliers_kendalltau,
             'Training(MSE_90%)': train_wo_outliers_mse,
             'Training(RMSE_90%)': train_wo_outliers_rmse,
             'Blindtest(Pearson)': blind_result.iloc[0]['Blindtest(Pearson)'],
             'Blindtest(Spearman)': blind_result.iloc[0]['Blindtest(Spearman)'],
             'Blindtest(Kendall)': blind_result.iloc[0]['Blindtest(Kendall)'],
             'Blindtest(MSE)': blind_result.iloc[0]['Blindtest(MSE)'],
             'Blindtest(RMSE)': blind_result.iloc[0]['Blindtest(RMSE)'],
             'Blindtest(Pearson_90%)': blind_result.iloc[0]['Blindtest(Pearson_90%)'],
             'Blindtest(Spearman_90%)': blind_result.iloc[0]['Blindtest(Spearman_90%)'],
             'Blindtest(Kendall_90%)': blind_result.iloc[0]['Blindtest(Kendall_90%)'],
             'Blindtest(MSE_90%)': blind_result.iloc[0]['Blindtest(MSE_90%)'],
             'Blindtest(RMSE_90%)': blind_result.iloc[0]['Blindtest(RMSE_90%)']}, ignore_index=True) 
            blind_scatter.to_csv(os.path.join(output_dir,blind_scatter_filename), index=False)
        else:
            result_ML = result_ML.append(
            {'filename': fname,
             'Training(Pearson)': train_w_outliers_pearsons,
             'Training(Spearman)': train_w_outliers_spearman,
             'Training(Kendall)': train_w_outliers_kendalltau,
             'Training(MSE)': train_w_outliers_mse,
             'Training(RMSE)': train_w_outliers_rmse,
             'Training(Pearson_90%)': train_wo_outliers_pearsons,
             'Training(Spearman_90%)': train_wo_outliers_spearman,
             'Training(Kendall_90%)': train_wo_outliers_kendalltau,
             'Training(MSE_90%)': train_wo_outliers_mse,
             'Training(RMSE_90%)': train_wo_outliers_rmse,
             'Blindtest(Pearson)': np.nan,
             'Blindtest(Spearman)': np.nan,
             'Blindtest(Kendall)': np.nan,
             'Blindtest(MSE)': np.nan,
             'Blindtest(RMSE)': np.nan,
             'Blindtest(Pearson_90%)': np.nan,
             'Blindtest(Spearman_90%)': np.nan,
             'Blindtest(Kendall_90%)': np.nan,
             'Blindtest(MSE_90%)': np.nan,
             'Blindtest(RMSE_90%)': np.nan}, ignore_index=True)
    else:
        result_ML = result_ML.append(
        {'filename': fname,
         'Training(Pearson)': train_w_outliers_pearsons,
         'Training(Spearman)': train_w_outliers_spearman,
         'Training(Kendall)': train_w_outliers_kendalltau,
         'Training(MSE)': train_w_outliers_mse,
         'Training(RMSE)': train_w_outliers_rmse,
         'Training(Pearson_90%)': train_wo_outliers_pearsons,
         'Training(Spearman_90%)': train_wo_outliers_spearman,
         'Training(Kendall_90%)': train_wo_outliers_kendalltau,
         'Training(MSE_90%)': train_wo_outliers_mse,
         'Training(RMSE_90%)': train_wo_outliers_rmse}, ignore_index=True)

    return result_ML

def run_blind(algorithm,fname,headers,target_label,regressor,blind_set,train_slope,train_intercept,error_type):
    result_ML = pd.DataFrame()
    # blind_set = pd.read_csv(blind_csv,header=0,low_memory=False)
    blind_set_ID = pd.DataFrame(blind_set['ID'])
    blind_label = np.array(blind_set[target_label])
    blind_features = blind_set[headers]
    # outerblindtest_result_filename = '{}_{}_{}_{}_10CV_blind_scatter_plot.csv'.format(timestr,fname,algorithm,str(random_state))

    ## outerblind-Test
    blind_prediction = regressor.predict(blind_features)
    blind_pd = pd.DataFrame(np.column_stack([blind_set_ID, blind_label, blind_prediction]),
                                 columns=['ID', 'actual', 'predicted'])

    blind_pd['ID'] = blind_pd['ID'].astype(int)
    blind_pd['old_ID'] = blind_set_ID.sort_values('ID')['ID'].tolist()
    blind_pd['error'] = abs(blind_pd['predicted'] - blind_pd['actual'])
    blind_pd['trans_predicted'] = (blind_pd['predicted']-train_intercept)/train_slope

    blind_slope, blind_intercept, blind_r_value, blind_p_value, blind_std_err = stats.linregress(
        blind_pd['actual'], blind_pd['predicted'])
    blind_pd['dist.to_bestfit'] = abs(
        (blind_slope * blind_pd['actual']) - blind_pd['predicted'] + blind_intercept) / sqrt(
        pow(blind_slope, 2) + 1)

    if error_type == 'Absolute':
        blind_pd = blind_pd.sort_values('error', ascending=False)
    elif error_type == 'BestFit':
        blind_pd = blind_pd.sort_values('dist.to_bestfit', ascending=False)

    ten_percent_of_outerblind = int(round(len(blind_pd['ID']) * 0.1))

    blind_non_outliers = blind_pd[ten_percent_of_outerblind:]
    blind_outliers = blind_pd[:ten_percent_of_outerblind]
    blind_outliers.insert(len(blind_outliers.columns), 'outlier', 'True')
    # Pearson's correlation
    blind_w_outliers_pearsons = round(
        sp.stats.pearsonr(blind_pd['actual'], blind_pd['predicted'])[0], 3)
    blind_wo_outliers_pearsons = round(
        sp.stats.pearsonr(blind_non_outliers['actual'], blind_non_outliers['predicted'])[0], 3)
    # Spearman's rank-order correlation
    blind_w_outliers_spearman = round(
        sp.stats.spearmanr(blind_pd['actual'], blind_pd['predicted'])[0], 3)
    blind_wo_outliers_spearman = round(
        sp.stats.spearmanr(blind_non_outliers['actual'], blind_non_outliers['predicted'])[0], 3)
    # Kendall's tau
    blind_w_outliers_kendalltau = round(
        sp.stats.kendalltau(blind_pd['actual'], blind_pd['predicted'])[0], 3)
    blind_wo_outliers_kendalltau = round(
        sp.stats.kendalltau(blind_non_outliers['actual'], blind_non_outliers['predicted'])[0], 3)
    # MSE
    blind_w_outliers_mse = round(mean_squared_error(blind_pd['actual'],blind_pd['predicted']), 2)
    blind_wo_outliers_mse = round(mean_squared_error(blind_non_outliers['actual'], blind_non_outliers['predicted']), 2)
    # RMSE
    blind_w_outliers_rmse = round(sqrt(mean_squared_error(blind_pd['actual'],blind_pd['predicted'])), 2)
    blind_wo_outliers_rmse = round(sqrt(mean_squared_error(blind_non_outliers['actual'], blind_non_outliers['predicted'])), 2)

    # if outlier, it will have 'true' else 'false'
    # blind_outliers = blind_outliers.drop(['predicted', 'actual', 'dist.to_bestfit', 'old_ID'], axis=1) # Maybe unnecessary
    overall_outerblind = pd.merge(blind_pd, blind_outliers[['ID', 'outlier']], how='left',
                                  on=['ID', 'ID'])
    overall_outerblind = overall_outerblind.fillna(False)
    overall_outerblind['dist.to_bestfit'] = overall_outerblind['dist.to_bestfit'].round(2)
    overall_outerblind['predicted'] = overall_outerblind['predicted'].round(2)
    overall_outerblind['error'] = overall_outerblind['error'].round(2)
    overall_outerblind['trans_predicted'] = blind_pd['trans_predicted'].round(2)

    result_ML = pd.DataFrame({'filename': fname,
         'Blindtest(Pearson)': blind_w_outliers_pearsons,
         'Blindtest(Spearman)': blind_w_outliers_spearman,
         'Blindtest(Kendall)': blind_w_outliers_kendalltau,
         'Blindtest(MSE)': blind_w_outliers_mse,
         'Blindtest(RMSE)': blind_w_outliers_rmse,
         'Blindtest(Pearson_90%)': blind_wo_outliers_pearsons,
         'Blindtest(Spearman_90%)': blind_wo_outliers_spearman,
         'Blindtest(Kendall_90%)': blind_wo_outliers_kendalltau,
         'Blindtest(MSE_90%)': blind_wo_outliers_mse,
         'Blindtest(RMSE_90%)': blind_wo_outliers_rmse},index=[0])

    return result_ML, overall_outerblind
####

def main(algorithm,train_csv,blind_csv,output_dir,target_label,error_type,n_cores,num_shuffle,save_model):
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    fname = os.path.split(train_csv.name)[1]
    sorting_order = list()
    train_pd = pd.read_csv(train_csv, sep=',', quotechar='\'', header=0)
    train_pd.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train_pd.columns.values]

    print("filename : {} |  algorithm : {}".format(fname,algorithm))

    result_ML = pd.DataFrame()
    result_of_blinds = pd.DataFrame()
    result_ML_output_name = '{}_{}_{}_10CV_{}_result.csv'.format(timestr,fname,algorithm,error_type)

    if blind_csv != 'False':
        blind_pd = pd.read_csv(blind_csv, header=0)
        blind_pd.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in blind_pd.columns.values]
    else:
        blind_pd = "False"

    if train_pd.columns[0] != 'ID':
        print("'ID' column should be given as 1st column.")
        sys.exit()

    for each in range(1, int(num_of_shuffling) + 1):
        each_result_ML = runML(algorithm, fname, train_pd, blind_pd, output_dir, target_label,
                               error_type, n_cores, num_of_shuffling,each,save_model)
        result_ML = result_ML.append([each_result_ML], ignore_index=True)  # for general results

    print(result_ML)
    result_ML.to_csv(os.path.join(output_dir, result_ML_output_name), index=True)


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='ex) python tenfold_regressor.py ExtraTrees ../test_files/training_regressor.csv dG')
    # REQUIRED
    parser.add_argument("algorithm", help="Choose algorithm between RF,GB,XGBOOST and ExtraTrees")
    parser.add_argument("train_csv", help="Choose input CSV(comma-separated values) format file",\
                        type=argparse.FileType('rt'))
    parser.add_argument("target_label", help="Type the name of label")
    # OPTIONAL
    parser.add_argument("-n_cores", help="Choose the number of cores to use", type=int, default=4)    
    parser.add_argument("-output_dir", help="Choose folder to save result(CSV)",default=output_dir)
    parser.add_argument("-num_shuffle", help="Choose the number of shuffling", type=int,default=1)
    parser.add_argument("-error_type", help="Absolute for Absolute Error and BestFit for distance to Best-Fit curve",default="BestFit")
    parser.add_argument("-blind_csv", help="Choose input CSV(comma-separated values) format file",default='False')
    parser.add_argument("-save_model", help="Save trained models",default="False")

    args = parser.parse_args()

    # REQUIRED
    algorithm = args.algorithm
    train_csv = args.train_csv
    target_label = args.target_label

    # OPTIONAL    
    n_cores = args.n_cores
    output_dir = args.output_dir
    num_of_shuffling = args.num_shuffle
    error_type = args.error_type
    blind_csv = args.blind_csv
    save_model = args.save_model

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(algorithm,train_csv,blind_csv,output_dir,target_label,error_type,n_cores,num_of_shuffling,save_model)
