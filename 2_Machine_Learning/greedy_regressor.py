# ********************************************************
# *               University of Melbourne                *
# *   ----------------------------------------------     *
# * Yoochan Myung - ymyung@student.unimelb.edu.au        *
# *   ----------------------------------------------     *
# ********************************************************
import pandas as pd
import numpy as np
import argparse
import csv
import sys
import time
import os
import scipy as sp

from terminalplot import plot
from terminalplot import get_terminal_size
import collections
from sklearn import preprocessing
import re
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
# import tenfold_regressor

def forwardGreedyFeature(training_csv, algorithm, output_dir, target_label, metric , n_cores, outerblind_set, cutoff):
    input_filename = os.path.split(training_csv)[1]

    ## DataFrame
    training_set = pd.read_csv(training_csv, header=0)
    training_set_ID= training_set.pop('ID')
    train_label = training_set[target_label]

    training_features = training_set.drop(target_label, axis=1)
    training_features_original = training_features.copy()

    greedy_features = pd.DataFrame()
    total_score_book = collections.OrderedDict()
    scored_feature = collections.OrderedDict()
    score_book = pd.DataFrame(columns=['training','blind','diff'])
    score_book_cutoff = pd.DataFrame(columns=['training','blind','diff'])

    total_feature_list = pd.DataFrame()
    scatter_plot = pd.DataFrame()
    stop_signal = False
    greedy_feature_num = 1
    running_counts = 1

    while stop_signal != True:

        num_of_features = len(training_features.columns)

        for x in range(0,num_of_features):
            # name of feature
            feature_name = training_features.iloc[:,x].name

            if len(greedy_features) > 0 :
                temp_feature_set = pd.concat([training_set_ID, greedy_features, training_features.iloc[:,x], train_label], axis=1)
            else:
                temp_feature_set = pd.concat([training_set_ID, training_features.iloc[:,x], train_label], axis=1)

            # only for C.V.
            if isinstance(outerblind_set,str):
                scored_feature[feature_name] = abs(round(float(runML(algorithm, feature_name, temp_feature_set,'False', output_dir, target_label,'BestFit', n_cores,1,1,'False')['Training({})'.format(metric)][0]),3))

            else:
                # for both C.V. and blind dataset.
                diff_cutoff = cutoff
                train_blind_csv_metrics = runML(algorithm, feature_name, temp_feature_set, outerblind_set, output_dir, target_label,'BestFit', n_cores,1,1,'False')
                score_book.loc[feature_name,'training'] = abs(round(float(train_blind_csv_metrics['Training({})'.format(metric)][0]),3))
                score_book.loc[feature_name,'blind'] = abs(round(float(train_blind_csv_metrics['Blindtest({})'.format(metric)][0]),3))
                score_book.loc[feature_name,'diff'] = abs(round(abs(train_blind_csv_metrics['Training({})'.format(metric)][0] - float(train_blind_csv_metrics['Blindtest({})'.format(metric)][0])),3))
                score_book.sort_values(by=['training'],inplace=True, ascending=False)
                print("=====[1] Fixed features=====")
                if len(greedy_features.columns) < 1:
                    print("No Fixed Feature")
                else:
                    print("{}".format(greedy_features.columns))
                print("=====[2] Greedy test=====")
                print("{}".format(score_book))
                print("=====[3] Features satisfy the user given cutoff condition( {}, def=.1)=====".format(cutoff))

                if score_book.loc[feature_name,'diff'] <= diff_cutoff:
                    scored_feature[feature_name] = score_book.loc[feature_name,'training']
                    # print("The amount of features passed cutoff : ",len(scored_feature))
                if len(scored_feature.items()) < 1:
                    print("No Features selected yet")
                else:
                    print("{}".format(list(scored_feature.items())))
                    print("")
            ## time cost
            print("progress : ", str(running_counts)+"/" + str(int(((len(training_features_original.columns)*(len(training_features_original.columns)+1))*0.5))))
            print(feature_name)
            print(scored_feature)
            running_counts +=1

        if not isinstance(outerblind_set,str) and len(scored_feature) < 1:
            print("Your Job is Done")
            stop_signal = True
            pass

        else:

            if metric == 'RMSE' or metric == 'MSE':
                scored_feature = sorted(scored_feature.items(), key=lambda t: t[1], reverse=True)
            else:
                scored_feature = sorted(scored_feature.items(), key=lambda t: t[1])
            highest_feature_name = scored_feature[-1][0].strip()
            highest_feature_score = scored_feature[-1][1]
            if isinstance(outerblind_set,str):
                total_feature_list = total_feature_list.append({'feature_name':highest_feature_name,'training':highest_feature_score},ignore_index=True)
            else:
                total_feature_list = total_feature_list.append({'feature_name':highest_feature_name,'training':score_book.loc[highest_feature_name,'training'],'blind':score_book.loc[highest_feature_name,'blind']},ignore_index=True)

            greedy_features = pd.concat([greedy_features,training_features[highest_feature_name]],axis=1)
            training_features = training_features.drop(highest_feature_name, axis=1)
            scored_feature = collections.OrderedDict()
            scatter_plot = scatter_plot.append({'feature_id': int(greedy_feature_num), metric :highest_feature_score}, ignore_index=True)
            total_score_book[highest_feature_name] = highest_feature_score
            get_terminal_size()
            plot(scatter_plot['feature_id'].tolist(), scatter_plot[metric].tolist())

            greedy_feature_num +=1

            if len(training_features.columns) == 0:
                stop_signal = True

            score_book = pd.DataFrame()
            scatter_plot.feature_id = scatter_plot.feature_id.astype(int)
            scatter_plot = scatter_plot[['feature_id',metric]]
            #total_feature_list = total_feature_list[['feature_name','training''blind']]
            total_feature_list.index +=1
            scatter_plot.to_csv(timestr+'_greedy_fs_'+ input_filename + '_' + algorithm +'_scatter_plot.csv',index=False)
            total_feature_list.to_csv(timestr+'_greedy_fs_'+ input_filename + '_' + algorithm +'_total_feature.csv',index=True, columns=['feature_name','training','blind'])
        print("=====[4] Greedy-based selected features=====")
        print("{}".format(list(total_score_book)))
        print("============================================")
        # with open(timestr+'_greedy_fs_'+ input_filename  + '_'+ algorithm +'_result.csv', 'w') as feature_writer:
        #     wr = csv.writer(feature_writer)
        #     wr.writerow(total_feature_list)

    # score_book.to_csv(timestr+'_greedy_fs'+ input_filename + '_' + algorithm +'_score_book.csv',index=False)
    return True
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

    # overall_training.to_csv(os.path.join(output_dir,overall_train_result_filename), index=False)

    if not isinstance(blind_set,str):
        blind_kfold_result_filename = '{}_{}_{}_{}_kfold_blind_results.csv'.format(timestr,fname,algorithm,str(random_state))
        # blind_kfold_result.to_csv(os.path.join(output_dir,blind_kfold_result_filename), index=False)

        if num_of_shuffling == random_state:
            # blind_result_filename = '{}_{}_{}_{}_10CV_blind_result.csv'.format(timestr,fname,algorithm,str(random_state))
            blind_scatter_filename = '{}_{}_{}_10CV_blind_scatter_plot.csv'.format(timestr,fname,algorithm)
            blind_result, blind_scatter = run_blind(algorithm,fname,headers,target_label,regressor,blind_set,train_slope,train_intercept,error_type)
            # blind_result.to_csv(os.path.join(output_dir,blind_result_filename), index=False)

            result_ML = result_ML.append(
            {'filename': fname,
             'Training(Pearson)': abs(train_w_outliers_pearsons),
             'Training(Spearman)': abs(train_w_outliers_spearman),
             'Training(Kendall)': abs(train_w_outliers_kendalltau),
             'Training(MSE)': train_w_outliers_mse,
             'Training(RMSE)': train_w_outliers_rmse,
             'Training(Pearson_90%)': abs(train_wo_outliers_pearsons),
             'Training(Spearman_90%)': abs(train_wo_outliers_spearman),
             'Training(Kendall_90%)': abs(train_wo_outliers_kendalltau),
             'Training(MSE_90%)': train_wo_outliers_mse,
             'Training(RMSE_90%)': train_wo_outliers_rmse,
             'Blindtest(Pearson)': abs(blind_result.iloc[0]['Blindtest(Pearson)']),
             'Blindtest(Spearman)': abs(blind_result.iloc[0]['Blindtest(Spearman)']),
             'Blindtest(Kendall)': abs(blind_result.iloc[0]['Blindtest(Kendall)']),
             'Blindtest(MSE)': blind_result.iloc[0]['Blindtest(MSE)'],
             'Blindtest(RMSE)': blind_result.iloc[0]['Blindtest(RMSE)'],
             'Blindtest(Pearson_90%)': abs(blind_result.iloc[0]['Blindtest(Pearson_90%)']),
             'Blindtest(Spearman_90%)': abs(blind_result.iloc[0]['Blindtest(Spearman_90%)']),
             'Blindtest(Kendall_90%)': abs(blind_result.iloc[0]['Blindtest(Kendall_90%)']),
             'Blindtest(MSE_90%)': blind_result.iloc[0]['Blindtest(MSE_90%)'],
             'Blindtest(RMSE_90%)': blind_result.iloc[0]['Blindtest(RMSE_90%)']}, ignore_index=True) 
            # blind_scatter.to_csv(os.path.join(output_dir,blind_scatter_filename), index=False)
        else:
            result_ML = result_ML.append(
            {'filename': fname,
             'Training(Pearson)': abs(train_w_outliers_pearsons),
             'Training(Spearman)': abs(train_w_outliers_spearman),
             'Training(Kendall)': abs(train_w_outliers_kendalltau),
             'Training(MSE)': train_w_outliers_mse,
             'Training(RMSE)': train_w_outliers_rmse,
             'Training(Pearson_90%)': abs(train_wo_outliers_pearsons),
             'Training(Spearman_90%)': abs(train_wo_outliers_spearman),
             'Training(Kendall_90%)': abs(train_wo_outliers_kendalltau),
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
         'Training(Pearson)': abs(train_w_outliers_pearsons),
         'Training(Spearman)': abs(train_w_outliers_spearman),
         'Training(Kendall)': abs(train_w_outliers_kendalltau),
         'Training(MSE)': train_w_outliers_mse,
         'Training(RMSE)': train_w_outliers_rmse,
         'Training(Pearson_90%)': abs(train_wo_outliers_pearsons),
         'Training(Spearman_90%)': abs(train_wo_outliers_spearman),
         'Training(Kendall_90%)': abs(train_wo_outliers_kendalltau),
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

if __name__ == '__main__':
    output_dir = os.path.dirname(os.path.abspath(__file__))
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # REQUIRED    
    parser = argparse.ArgumentParser(description='ex) python greedy_regressor.py ExtraTrees ./test_files/training_regressor.csv dG')
    parser.add_argument("algorithm", help="Choose algorithm [GB,XGBOOST,RF,ExtraTrees,GAUSSIAN,ADABOOST,KNN,SVC,NEURAL]")
    parser.add_argument("train_csv", help="Choose input CSV(comma-separated values) format file")
    parser.add_argument("target_label", help="Type the name of label")
    # OPTIONAL
    parser.add_argument("-n_cores", help="Choose the number of cores to use", type=int, default=4)    
    parser.add_argument("-output_dir", help="Choose folder to save result(CSV)",default=output_dir)
    parser.add_argument("-metric", help="Choose one metric for greedy_feature selection ['Pearson','MSE','RMSE']",default="Pearson")
    parser.add_argument("-cutoff", help="Set cutoff value for the difference between training and blind performance, default=0.1", default=0.1, type=float)
    parser.add_argument("-blind_csv", help="Choose input CSV(comma-separated values) format file",default='False')
    
    args = parser.parse_args()

    # REQUIRED
    algorithm = args.algorithm
    training_csv = args.train_csv
    target_label = args.target_label
    
    # OPTIONAL
    n_cores = args.n_cores
    cutoff = args.cutoff
    blind_csv = args.blind_csv
    metric = args.metric
    output_dir = args.output_dir

    if blind_csv != 'False':
        outerblind_csv_set = pd.read_csv(blind_csv, header=0)

    else:
        outerblind_csv_set = "False"

    forwardGreedyFeature(training_csv, algorithm, output_dir, target_label, metric, n_cores, outerblind_csv_set, cutoff)
