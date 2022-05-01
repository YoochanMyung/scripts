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
import argparse
import joblib

from math import sqrt
from scipy import stats

from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm # for SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report


timestr = time.strftime("%Y%m%d_%H%M%S")

def runML(algorithm, fname, training_set, outerblind_set, output_result_dir, label_name, n_cores, random_state):
    result_ML = pd.DataFrame()
    result_cm = pd.DataFrame() # Confusion Matrix
    blind_cm = pd.DataFrame() # Confusion Matrix
    cv_metrics = pd.DataFrame() 

    training_ID = pd.DataFrame(training_set['ID'])
    train_label = np.array(training_set[label_name])
    training_set = training_set.drop('ID', axis=1)
    training_features = training_set.drop(label_name, axis=1)

    # Label Encoding
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    num_of_class = len(le.classes_)
    train_label = le.transform(train_label)

    headers = list(training_features.columns.values)

    # 10-fold cross validation
    predicted_n_actual_pd = pd.DataFrame(columns=['ID', 'predicted', 'actual', 'fold'])
    outerblind_predicted_n_actual_pd = pd.DataFrame(columns=['ID', 'predicted', 'actual'])

    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    fold = 1 # for indexing

    for train, test in kf.split(training_features):
        # train and test number(row number) are based on training_features.
        # For example, if '1' from train or test, it would be '1' in training_features

        train_cv_features, test_cv_features, train_cv_label, test_cv_label = training_features.iloc[train], \
                                                                             training_features.iloc[test], train_label[
                                                                                 train], train_label[test]

        if algorithm == 'GB':
            temp_classifier = GradientBoostingClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'XGBOOST'):
            temp_classifier = XGBClassifier(n_estimators=300, random_state=1, use_label_encoder=False, n_jobs=n_cores)

        elif (algorithm == 'RF'):
            temp_classifier = RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'ExtraTrees'):
            temp_classifier = ExtraTreesClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'GAUSSIAN'):
            temp_classifier = GaussianProcessClassifier(random_state=1)

        elif (algorithm == 'ADABOOST'):
            temp_classifier = AdaBoostClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'KNN'):
            temp_classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=n_cores)

        elif (algorithm == 'SVC'):
            temp_classifier = svm.SVC(kernel='rbf')

        elif (algorithm == 'MLP'):
            temp_classifier = MLPClassifier(random_state=1)

        elif (algorithm == 'DecisionTree'):
            temp_classifier = DecisionTreeClassifier(random_state=1)

        elif (algorithm == 'LG'):
            temp_classifier = LogisticRegression(random_state=1,multi_class='ovr')

        temp_classifier.fit(train_cv_features, train_cv_label)
        temp_prediction = temp_classifier.predict(test_cv_features)
        temp_proba = temp_classifier.predict_proba(test_cv_features)
        predicted_n_actual_pd = pd.DataFrame()
        predicted_n_actual_pd = predicted_n_actual_pd.append(pd.DataFrame({'ID':test, 'actual':test_cv_label, 'predicted' : temp_prediction, 'fold':fold}),ignore_index=True, sort=True)
        fold += 1

    try :
        if num_of_class > 2:
            roc_auc = round(roc_auc_score(predicted_n_actual_pd['actual'].to_list(),temp_proba, multi_class='ovr'),3)
        else:
            roc_auc = round(roc_auc_score(predicted_n_actual_pd['actual'].to_list(),temp_proba),3)

    except ValueError:
        roc_auc = 0.0

    if num_of_class > 2:
        f1 = round(f1_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list(),average='micro'),3)
    else:
        f1 = round(f1_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)
    
    matthews = round(matthews_corrcoef(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)
    balanced_accuracy = round(balanced_accuracy_score(predicted_n_actual_pd['actual'].to_list(),predicted_n_actual_pd['predicted'].to_list()),3)
    
    try:
        if num_of_class > 2:
            mcm = multilabel_confusion_matrix(predicted_n_actual_pd['actual'].to_list(), predicted_n_actual_pd['predicted'].to_list())
            tn = mcm[:,0, 0]
            tp = mcm[:,1, 1]
            fn = mcm[:,1, 0]
            fp = mcm[:,0, 1]
            result_cm = pd.DataFrame(np.column_stack((tn,tp,fn,fp)),columns=['tn','tp','fn','fp'],index=le.classes_)
        else:
            tn, fp, fn, tp = confusion_matrix(predicted_n_actual_pd['actual'].to_list(), predicted_n_actual_pd['predicted'].to_list()).ravel()
            result_cm = pd.DataFrame(np.column_stack((tn,tp,fn,fp)),columns=['tn','tp','fn','fp'])

    except:
        tn, fp, fn, tp = 0

    cv_metrics = cv_metrics.append(pd.DataFrame(np.column_stack(['cv',roc_auc, matthews,balanced_accuracy, f1 ]), columns=['type','roc_auc','matthew','bacc','f1']), ignore_index=True, sort=True)

    # # To get ML model in SAV file.
    # model = classifier.fit(training_features,train_label)
    # model_filename = os.path.join(output_result_dir,fname + '_' + str(random_state) + '_' + timestr + '_model.sav')
    # joblib.dump(model, model_filename)

    if outerblind_set is not 'False':
        outerblind_cv_metrics = pd.DataFrame()
        blind_cm = pd.DataFrame()

        if algorithm == 'GB':
            classifier = GradientBoostingClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'XGBOOST'):
            classifier = XGBClassifier(n_estimators=300, random_state=1, use_label_encoder=False, n_jobs=n_cores)

        elif (algorithm == 'RF'):
            classifier = RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'ExtraTrees'):
            classifier = ExtraTreesClassifier(n_estimators=300, random_state=1, n_jobs=n_cores)

        elif (algorithm == 'GAUSSIAN'):
            classifier = GaussianProcessClassifier(random_state=1)

        elif (algorithm == 'ADABOOST'):
            classifier = AdaBoostClassifier(n_estimators=300, random_state=1)

        elif (algorithm == 'KNN'):
            classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=n_cores)

        elif (algorithm == 'SVC'):
            classifier = svm.SVC(kernel='rbf')

        elif (algorithm == 'MLP'):
            classifier = MLPClassifier(random_state=1)

        elif (algorithm == 'DecisionTree'):
            classifier = DecisionTreeClassifier(random_state=1)

        elif (algorithm == 'LG'):
            classifier = LogisticRegression(random_state=1,multi_class='ovr')

        else:
            print("Algorithm Selection ERROR!!")
            sys.exit()

        outerblind_set_ID = pd.DataFrame(outerblind_set['ID'])
        outerblind_label = np.array(outerblind_set[label_name])
        outerblind_features = outerblind_set[headers]
        outerblind_label = le.transform(outerblind_label)
        headers = list(training_features.columns.values)

        ## outerblind-Test
        classifier.fit(training_features,train_label)
        prediction = classifier.predict(outerblind_features)
        probability = classifier.predict_proba(outerblind_features)
        outerblind_predicted_n_actual_pd = outerblind_predicted_n_actual_pd.append(pd.DataFrame({'ID':outerblind_set_ID['ID'].to_list(), 'actual':outerblind_label, 'predicted' : prediction}),ignore_index=True, sort=True)

        outerblind_matthews_corrcoef = round(matthews_corrcoef(outerblind_label, prediction),3)
        outerblind_balanced_accuracy_score = round(balanced_accuracy_score(outerblind_label, prediction),3)

        try :
            if num_of_class > 2:            
                outerblind_roc_auc_score = round(roc_auc_score(outerblind_label, probability,multi_class='ovr'),3)
            else:
                outerblind_roc_auc_score = round(roc_auc_score(outerblind_label, probability),3)

        except ValueError:
            outerblind_roc_auc_score = 0.0

        if num_of_class > 2:
            outerblind_f1_score = round(f1_score(outerblind_label, prediction,average='micro'),3)
        else:
            outerblind_f1_score = round(f1_score(outerblind_label, prediction),3)

        try:
            if num_of_class > 2:
                outerblind_mcm = multilabel_confusion_matrix(outerblind_label, prediction)
                outerblind_tn = mcm[:,0, 0]
                outerblind_tp = mcm[:,1, 1]
                outerblind_fn = mcm[:,1, 0]
                outerblind_fp = mcm[:,0, 1]
                blind_cm = pd.DataFrame(np.column_stack((outerblind_tn,outerblind_tp,outerblind_fn,outerblind_fp)),columns=['tn','tp','fn','fp'],index=le.classes_)

            else:
                outerblind_tn, outerblind_fp, outerblind_fn, outerblind_tp = confusion_matrix(outerblind_label, prediction).ravel()
                blind_cm = pd.DataFrame(np.column_stack((outerblind_tn,outerblind_tp,outerblind_fn,outerblind_fp)),columns=['tn','tp','fn','fp'])

        except:
            outerblind_tn,outerblind_fp,outerblind_fn,outerblind_tp = 0
            blind_cm = pd.DataFrame(np.column_stack((0,0,0,0)),columns=['tn','tp','fn','fp'])

        outerblind_cv_metrics = outerblind_cv_metrics.append(pd.DataFrame(np.column_stack(['blind-test',outerblind_roc_auc_score, outerblind_matthews_corrcoef,
            outerblind_balanced_accuracy_score, outerblind_f1_score]),
             columns=['type','roc_auc','matthew','bacc','f1']), ignore_index=True, sort=True)
        outerblind_cv_metrics.set_index([['blind-test']*len(outerblind_cv_metrics)], inplace=True)
        cv_metrics = pd.concat([cv_metrics, outerblind_cv_metrics], sort=True)

    cv_metrics = cv_metrics.round(3)
    predicted_n_actual_pd['predicted'] = le.inverse_transform(predicted_n_actual_pd['predicted'].to_list())
    predicted_n_actual_pd['actual'] = le.inverse_transform(predicted_n_actual_pd['actual'].to_list())
    fname_predicted_n_actual_pd = os.path.join(output_result_dir,'cv_{}_{}_predicted_data.csv'.format(algorithm,label_name))
    fname_result_cm = os.path.join(output_result_dir,'cv_{}_{}_confusion_matrix.csv'.format(algorithm,label_name))
    predicted_n_actual_pd['ID'] = predicted_n_actual_pd['ID'] + 1
    predicted_n_actual_pd = predicted_n_actual_pd.sort_values(by=['ID'])
    predicted_n_actual_pd.to_csv(fname_predicted_n_actual_pd,index=False)
    result_cm.to_csv(fname_result_cm)

    outerblind_predicted_n_actual_pd['predicted'] = le.inverse_transform(outerblind_predicted_n_actual_pd['predicted'].to_list())
    outerblind_predicted_n_actual_pd['actual'] = le.inverse_transform(outerblind_predicted_n_actual_pd['actual'].to_list())

    if outerblind_set is not 'False':
        fname_outerblind_predicted_n_actual_pd = os.path.join(output_result_dir,'blindtest_{}_{}_predicted_data.csv'.format(algorithm,label_name))
        fname_blind_cm = os.path.join(output_result_dir,'blindtest_{}_{}_confusion_matrix.csv'.format(algorithm,label_name))
        outerblind_predicted_n_actual_pd.to_csv(fname_outerblind_predicted_n_actual_pd,index=False)
        blind_cm.to_csv(fname_blind_cm)
        print("Label: {}".format(le.classes_))
        print("CV_confusion_matrix")
        print(result_cm)
        print("Blind-test_confusion_matrix")        
        print(blind_cm)
        return cv_metrics
    else:
        print("Label: {}".format(le.classes_))
        print("CV_confusion_matrix")
        print(result_cm)
        return cv_metrics


def main(algorithm, input_csv, outerblindtest_csv, output_result_dir, label_name, n_cores, num_shuffle):
    fname = os.path.split(input_csv.name)[1]
    original_dataset = pd.read_csv(input_csv, sep=',', quotechar='\"', header=0)

    print("filename :", fname)

    result_ML = pd.DataFrame()
    result_of_blinds = pd.DataFrame()

    if outerblindtest_csv != 'False':
        outerblindtest_set = pd.read_csv(outerblindtest_csv, header=0)

    else:
        outerblindtest_set = "False"

    if original_dataset.columns[0] != 'ID':
        print("'ID' column should be given as 1st column.")
        sys.exit()

    for each in range(1, int(num_shuffle) + 1):

        each_result_ML = runML(algorithm, fname, original_dataset, outerblindtest_set, output_result_dir, label_name,
                               n_cores, each)

        result_ML = result_ML.append([each_result_ML], ignore_index=False)  # for general results

    result_ML = result_ML.reset_index(drop=True)
    result_ML.index += 1

    fname_result_ML = os.path.join(output_result_dir,'10CV_{}_result.csv'.format(algorithm))
    result_ML.to_csv(fname_result_ML,index=False)

    return result_ML


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # REQUIRED    
    parser = argparse.ArgumentParser(description='ex) python tenfold_classifier.py ExtraTrees ../test_files/training_classifier.csv age')
    parser.add_argument("algorithm", help="Choose algorithm between RF,GB,XGBOOST and ExtraTrees")
    parser.add_argument("input_csv", help="Choose input CSV(comma-separated values) format file",
                        type=argparse.FileType('rt'))
    parser.add_argument("target_label", help="Type the name of label")
    
    # OPTIONAL
    parser.add_argument("-num_shuffle", help="Choose the number of shuffling", type=int, default=10)    
    parser.add_argument("-output_dir", help="Choose folder to save result(CSV)",default=output_dir)
    parser.add_argument("-n_cores", help="Choose the number of cores to use", type=int, default=4)
    parser.add_argument("-blind_csv", help="Choose input CSV(comma-separated values) format file",
                        default='False')    # optional

    args = parser.parse_args()

    # REQUIRED
    algorithm = args.algorithm
    input_csv = args.input_csv
    target_label = args.target_label
    
    # OPTIONAL
    blind_csv = args.blind_csv
    n_cores = args.n_cores
    num_shuffle = args.num_shuffle
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(main(algorithm, input_csv, blind_csv, output_dir, target_label, n_cores, num_shuffle))
