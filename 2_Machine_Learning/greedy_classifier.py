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

from terminalplot import plot
from terminalplot import get_terminal_size
import collections
from sklearn import preprocessing
import tenfold_classifier

def forwardGreedyFeature(training_csv, algorithm, output_result_dir, label_name, scoring , n_cores, outerblind_set, cutoff):
    input_filename = os.path.split(training_csv)[1]
    
    ## DataFrame 
    training_set = pd.read_csv(training_csv, quotechar='\"', header=0)
    training_set_ID= training_set.pop('ID')
    train_label = training_set[label_name]

    training_features = training_set.drop(label_name, axis=1)
    training_features_original = training_features.copy()

    greedy_features = pd.DataFrame()
    total_score_book = collections.OrderedDict()
    scored_feature = collections.OrderedDict()
    score_book = pd.DataFrame(columns=['training','blind-test','diff'])
    score_book_cutoff = pd.DataFrame(columns=['training','blind-test','diff'])

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
                scored_feature[feature_name] = float(tenfold_classifier.runML(algorithm, feature_name, temp_feature_set, 'False', output_result_dir, label_name, n_cores, 1)[scoring].loc[0])
            else:
                # for both C.V. and blind-test dataset.
                diff_cutoff = cutoff
                train_blindtest_metrics = tenfold_classifier.runML(algorithm, feature_name, temp_feature_set, outerblind_set, output_result_dir, label_name, n_cores, 1)[scoring]
                temp_training_score = round(float(train_blindtest_metrics.loc[0]),3)
                temp_blindtest_score = round(float(train_blindtest_metrics.loc['blind-test']),3)
                temp_diff_score = abs(round(temp_training_score - temp_blindtest_score,3))
                score_book.loc[feature_name,'training'] = temp_training_score
                score_book.loc[feature_name,'blind-test'] = temp_blindtest_score
                score_book.loc[feature_name,'diff'] = temp_diff_score
                score_book.sort_values(by=['training'],inplace=True, ascending=False)    
                print("=====[1] Fixed features=====")
                if len(greedy_features.columns) < 1:
                    print("No Fixed Feature")
                else:
                    print("{}".format(greedy_features.columns))
                print("=====[2] Greedy test=====")
                print("{}".format(score_book))
                print("=====[3] Features satisfy the user given cutoff condition ({} ,def = 0.1) =====".format(cutoff))
                if temp_diff_score <= diff_cutoff:
                    scored_feature[feature_name] = temp_training_score
                if len(scored_feature.items()) < 1:
                    print("No Features selected yet")
                else:
                    print('{}'.format(list(scored_feature.items())))
                    print("")
            ## time cost
            print("progress : ", str(running_counts)+"/" + str(int(((len(training_features_original.columns)*(len(training_features_original.columns)+1))*0.5))))
            running_counts +=1

        if not isinstance(outerblind_set,str) and len(scored_feature) < 1:
            print("Your Job is Done")
            stop_signal = True
            pass

        else:

            scored_feature = sorted(scored_feature.items(), key=lambda t: t[1])
            highest_feature_name = scored_feature[-1][0].strip()
            highest_feature_score = scored_feature[-1][1]
            if isinstance(outerblind_set,str):
                total_feature_list = total_feature_list.append({'feature_name':highest_feature_name,'training':highest_feature_score},ignore_index=True)
            else:
                total_feature_list = total_feature_list.append({'feature_name':highest_feature_name,'training':score_book.loc[highest_feature_name,'training'],'blind-test':score_book.loc[highest_feature_name,'blind-test']},ignore_index=True)
            greedy_features = pd.concat([greedy_features,training_features[highest_feature_name]],axis=1) # Build input_data containing selected feature
            training_features = training_features.drop(highest_feature_name, axis=1) # Remove the selected feature for further runs
            scored_feature = collections.OrderedDict() # Reset
            scatter_plot = scatter_plot.append({'feature_id': int(greedy_feature_num), scoring :highest_feature_score}, ignore_index=True)
            total_score_book[highest_feature_name] = highest_feature_score
            get_terminal_size()
            plot(scatter_plot['feature_id'].tolist(), scatter_plot[scoring].tolist())

            greedy_feature_num +=1

            if len(training_features.columns) == 0:
                stop_signal = True

            score_book = pd.DataFrame()
            scatter_plot.feature_id = scatter_plot.feature_id.astype(int)
            scatter_plot = scatter_plot[['feature_id',scoring]]
            total_feature_list.index +=1
            scatter_plot.to_csv(timestr+'_greedy_fs_'+ input_filename + '_' + algorithm +'_scatter_plot.csv',index=False)
            total_feature_list.to_csv(timestr+'_greedy_fs_'+ input_filename + '_' + algorithm +'_total_feature.csv',index=True)
        print("=====[4] Greedy-based selected features=====")
        print("{}".format(list(total_score_book))) # This dataframe shows features that satisfy the user given cut-off condition (def = 0.1).
        print("============================================")
    return True


if __name__ == '__main__':
    output_dir = os.path.dirname(os.path.abspath(__file__))
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # REQUIRED    
    parser = argparse.ArgumentParser(description='ex) python greedy_classifier.py M5P ../test_files/training_classifier.csv dG')
    parser.add_argument("algorithm", help="Choose algorithm [GB,XGBOOST,RF,M5P,GAUSSIAN,ADABOOST,KNN,SVC,NEURAL]")
    parser.add_argument("train_csv", help="Choose input CSV(comma-separated values) format file")
    parser.add_argument("target_label", help="Type the name of label")

    # OPTIONAL    
    parser.add_argument("-n_cores", help="Choose the number of cores to use", type=int, default=4)
    parser.add_argument("-metric", help="Choose one metric for greedy_feature selection ['roc_auc','matthew','bacc','f1']", default="matthew")    
    parser.add_argument("-output_dir", help="Choose folder to save result(CSV)",default=output_dir)
    parser.add_argument("-cutoff", help="Set cutoff value for the difference between training and blind-test performance, default=0.1", default=0.1, type=float)
    parser.add_argument("-blind_csv", help="Choose input CSV(comma-separated values) format file",
                        default='False')
    args = parser.parse_args()
    
    # REQUIRED
    algorithm = args.algorithm
    training_csv = args.train_csv
    target_label = args.target_label

    # OPTIONAL
    n_cores = args.n_cores
    metric = args.metric    
    cutoff = args.cutoff
    blindtest = args.blind_csv
    output_dir = args.output_dir

    if blindtest != 'False':
        outerblindtest_set = pd.read_csv(blindtest, quotechar='\"', header=0)

    else:
        outerblindtest_set = "False"

    forwardGreedyFeature(training_csv, algorithm, output_dir, target_label, metric, n_cores, outerblindtest_set, cutoff)
