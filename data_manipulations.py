import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, mean_squared_error
from sklearn.metrics import confusion_matrix


def check_template(c, t):
    c_split = c.split('//')
    return c_split[0] == t


def get_entries(data, patient, arztbrief_columns, behandlungsplan_columns, ct_columns, kai_columns, laborparameter_columns, baseline=True):
    patient_dict = {'PatientID': patient}

    # Get relevant Arztbrief
    tmp_data = data[data['PatientID'] == patient]
    rel_index = np.argmin(list(tmp_data[tmp_data['Template'] == 'Arztbrief/KIS Angaben']
                          ['Arztbrief/KIS Angaben//Arztbrief/KIS Angaben::Tage seit Aufnahme']))
    patient_dict.update(tmp_data[tmp_data['Template'] == 'Arztbrief/KIS Angaben'].reset_index(
        drop=True).iloc[rel_index][arztbrief_columns])

    # Get relevant CT
    tmp_data = data[data['PatientID'] == patient]
    rel_index = np.argmin(
        list(tmp_data[tmp_data['Template'] == 'CT']['CT//StudyDate']))
    patient_dict.update(tmp_data[tmp_data['Template'] == 'CT'].reset_index(
        drop=True).iloc[rel_index][ct_columns])

    # Get relevant Klinisch-anamnestische Information
    tmp_data = data[data['PatientID'] == patient]
    if baseline:
        tmp_data = tmp_data[(tmp_data['Template'] == 'Klinisch-anamnestische Information') & (tmp_data['Klinisch-anamnestische Information//Komorbiditäten aus Arztbrief::Untersuchungstyp'].isin(
            ['Baseline', 'Baseline, CT', 'Baseline, Zwischenwert', 'Baseline, CT, Zwischenwert', 'CT, Zwischenwert']))].reset_index(drop=True)
    else:
        tmp_data = tmp_data[tmp_data['Template'] ==
                            'Klinisch-anamnestische Information'].reset_index(drop=True)
    patient_dict.update(tmp_data.iloc[0][kai_columns].to_dict())

    # Get relevant Behandlungsplan
    tmp_data = data[(data['PatientID'] == patient) & (
        data['Template'] == 'Behandlungsplan')]
    tmp_data = tmp_data.sort_values(
        'Behandlungsplan//Behandlungsplan::Behandlungsplan: Erhebungsdatum')
    for column in behandlungsplan_columns:
        for i, row in tmp_data.iterrows():
            if column not in patient_dict:
                try:
                    if np.isnan(row[column]):
                        continue
                    else:
                        patient_dict.update({column: row[column]})
                except TypeError:
                    patient_dict.update({column: row[column]})
                except AttributeError:
                    patient_dict.update({column: row[column]})

    # Get relevant Laborparameter
    tmp_data = data[(data['PatientID'] == patient) &
                    (data['Template'] == 'Laborparameter')]
    tmp_data = tmp_data.sort_values(
        'Laborparameter//Laborparameter::Labordaten: Erhebungsdatum')
    for column in laborparameter_columns:
        for i, row in tmp_data.iterrows():
            if column not in patient_dict:
                try:
                    if np.isnan(row[column]):
                        continue
                    else:
                        patient_dict.update({column: row[column]})
                except TypeError:
                    patient_dict.update({column: row[column]})
                except AttributeError:
                    patient_dict.update({column: row[column]})
    return patient_dict


def lower_value(v):
    try:
        return v.lower()
    except AttributeError:
        return v


def encode(v, enc):
    v = lower_value(v)
    if v in enc.keys():
        return enc[v]
    else:
        return v


def get_highest_value(data, pid, col):
    tmp_data = data[data['PatientID'] == pid]
    # target_list = tmp_data[col].unique()
    # return np.max(target_list)
    return tmp_data[col].max()

def is_severe(row):
    if (row['Target_0'] == 5) & (row['Target_2'] >= 3):
        return 1
    else:
        return 0


def plot_feature_importances(model, model_features, n_features=20):
    importances = model.feature_importances_
    selected_features = model_features
    indices = np.argsort(importances)[::-1][:n_features][::-1]

    plt.title('Feature Importances')
    plt.barh(range(len(indices)),
             importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [selected_features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    return selected_features, indices


def has_baseline(data, patient):
    study_types = data[data['PatientID'] ==
                       patient]['Klinisch-anamnestische Information//Komorbiditäten aus Arztbrief::Untersuchungstyp'].unique()
    if len(study_types) > 0:
        for t in study_types:
            try:
                if 'Baseline' in t:
                    return True
            except TypeError:
                continue
    return False


def list_diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


def count_affected_lobes(lobes):
    if lobes == 'Undefiniert':
        return 0
    return len(lobes.split(','))


def count_comorbs(row, comorbs):
    sum_ = 0
    for c in comorbs:
        sum_ += row[f"Occurrence_{c.split('::')[1]}"]
    return sum_


def get_target_predictions(x, y, print_conf_matrix=False, print_dist=False, plot_imp=False):
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)

    print('RandomForest\n--------------------')
    print(f'Accuracy: {accuracy_score(y_test, rf_predictions)}')
    print(
        f"F1 score (weighted average): {f1_score(y_test, rf_predictions, average='weighted')}")
    if print_dist:
        print(
            f'Distribution y_test: {sorted(collections.Counter(y_test).items())}')
        print(
            f'Distribution xgb_predictions: {sorted(collections.Counter(rf_predictions).items())}')
        print(f'# test items: {len(y_test)}')
    if print_conf_matrix:
        print('Confusion matrix:')
        print(confusion_matrix(y_test, rf_predictions))
    if plot_imp:
        features, _ = plot_feature_importances(
            rf, x.columns, n_features=plot_imp)

    xgb = GradientBoostingClassifier(random_state=42)
    xgb.fit(X_train, y_train)
    xgb_predictions = xgb.predict(X_test)

    print('====================')

    print('\nGradientBoostingClassifier\n--------------------')
    print(f'Accuracy: {accuracy_score(y_test, xgb_predictions)}')
    print(
        f"F1 score (weighted average): {f1_score(y_test, xgb_predictions, average='weighted')}")
    if print_dist:
        print(
            f'Distribution y_test: {sorted(collections.Counter(y_test).items())}')
        print(
            f'Distribution xgb_predictions: {sorted(collections.Counter(xgb_predictions).items())}')
        print(f'# test items: {len(y_test)}')
    if print_conf_matrix:
        print('Confusion matrix:')
        print(confusion_matrix(y_test, xgb_predictions))
    if plot_imp:
        features, _ = plot_feature_importances(
            xgb, x.columns, n_features=plot_imp)


def get_predictions(data, columns=None, print_conf_matrix=False, print_dist=False, plot_imp=False):
    df = data.copy()
    df.fillna(-1, inplace=True)

    ids = df.pop('PatientID')
    target_0 = df.pop('Target_0')
    target_1 = df.pop('Target_1')
    target_2 = df.pop('Target_2')
    target_3 = df.pop('Target_3')
    target_4 = df.pop('Target_4')

    targets = [target_0, target_1, target_2, target_3, target_4]

    if columns is not None:
        X = df[[col for col in columns if col in df.columns]]
    else:
        X = df[df.columns]

    for i, target in enumerate(targets):
        print(f'Predictions target {i}\n====================\n')
        get_target_predictions(
            X, target, print_conf_matrix=print_conf_matrix, print_dist=print_dist, plot_imp=plot_imp)
        print('\n\n')


def get_agg_severity(row, patho, agg='sum'):
    sum_ = 0
    for lobe in ['Oberlappen rechts', 'Mittellappen rechts', 'Unterlappen rechts', 'Unterlappen links', 'Lingula', 'Unterlappen rechts']:
        severity = row[f'CT//{patho}::Schweregrad ({lobe})']
        if not pd.isnull(severity):
            sum_ += severity
    if agg == 'sum':
        return sum_
    elif agg == 'avg':
        return sum_ / 6


def select_significants(df, features, target, alpha=0.05):
    significants = []
    for col in features:
        if col not in df:
            continue
        cont = pd.crosstab(df[col], df[target])
        p_value = chi2_contingency(cont)[1]
        if p_value < alpha:
            significants.append(col)
    return significants


def predict_missing_values(df, column, pred_type='classifier'):
    try:
        d = df.copy()
        
        # ids_train = d.pop('PatientID')
        # target_0 = d.pop('Target_0')
        # target_2 = d.pop('Target_2')
        # target_3 = d.pop('Target_3')

        missing_indices = d[d[column].isnull()].index
        train_dev = d.drop(missing_indices)

        y = train_dev.pop(column)
        X = train_dev[list(train_dev.columns[~train_dev.isnull().any()])]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if pred_type == 'classifier':
            rf = RandomForestClassifier(random_state=0)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            print(f'Accuracy for {column}: {accuracy}')
        else:
            rf = RandomForestRegressor(random_state=0)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print(f'MSE for {column}: {mse}')
        
        all_test = d.iloc[missing_indices]
        all_train = d.drop(missing_indices)
        
        prediction_columns = list(d.columns[~d.isnull().any()])
        X_train = all_train[prediction_columns]
        X_test = all_test[prediction_columns]
        y_train = all_train[column]
        y_test = all_train[column]
        
        if pred_type == 'classifier':
            rf = RandomForestClassifier(random_state=0)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
        else:
            rf = RandomForestRegressor(random_state=0)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
        
        return rf, missing_indices, predictions
    except Exception as e:
        print(column)
        print(e)
        return None, [], []


def total_severity_score(df):
    total_col = 'CT//Severity Scores::Lungen-CT Gesamtscore'
    sev_columns = [f'CT//Severity Scores::Lunge {loc} {direction}' for loc in ['Oberfeld', 'Mittelfeld', 'Unterfeld'] for direction in ['rechts', 'links']]
    
    missing_indices = df[df[total_col].isnull()].index
    total_sums = []
    for missing_index in missing_indices:
        sev_sum = 0
        for col in sev_columns:
            sev_sum += df[col][missing_index]
        total_sums.append(sev_sum)
    return missing_indices, total_sums