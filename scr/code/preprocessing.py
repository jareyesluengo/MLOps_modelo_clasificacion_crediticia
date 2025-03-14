import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
    
install('datasist')
install('imblearn')

# Libraries
import os
import boto3
import joblib
import tarfile
import argparse

import numpy as np
import pandas as pd

from datasist.structdata import detect_outliers

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer, StandardScaler

#############
# Functions #
#############
def parse_args():
    parser = argparse.ArgumentParser()
    
    # bucket variables
    parser.add_argument("--bucket-input", type=str, required=True)
    parser.add_argument("--bucket-prefix", type=str, required=True)

    return parser.parse_known_args()

def limpieza_cambio(value):
    value_str = str(value).strip('_')
    if value_str == '':
        return np.nan
    try:
        return float(value_str)
    except ValueError:
        return np.nan

#############
# Variables #
#############
# variables numericas
clean_num = ['Age',
             'Annual_Income',
             'Num_of_Loan',
             'Changed_Credit_Limit',
             'Num_of_Delayed_Payment',
             'Outstanding_Debt']
# # listado organizado de codificación
credit_mix_order = ['good', 'standard', 'poor', np.nan]
credit_score_order = ['good', 'standard', 'poor']

# definiciones
s3 = boto3.client('s3')
base_dir = "/opt/ml/processing"

if __name__ == "__main__":
    args, _ = parse_args()

    get_last_modified = lambda obj: int(obj['LastModified'].strftime('%s'))
    objs = s3.list_objects_v2(Bucket=args.bucket_input, Prefix=args.bucket_prefix)['Contents']
    last_added = [obj['Key'] for obj in sorted(objs, key=get_last_modified, reverse=True)][0].split('/')[-1]
    print(f'Ultimo archivo agregado: {last_added}')
    train_dataset= 'train.csv'
    dataset = pd.read_csv(f'{base_dir}/input/{train_dataset}') # last_added
    ## eliminación de columnas 
    # se eliminan datos unicos de identificación
    dataset.drop(['ID','Customer_ID','Name','SSN'],axis=1 , inplace =True)
    # se eliminan las variables que no aportan y la variable objetivo
    dataset.drop(['Month', 'Occupation', 'Type_of_Loan', 'Credit_Utilization_Ratio', 'Payment_Behaviour'], axis = 1, inplace = True)
    ## corrección datos cuantitativos
    # reemplazo de valores
    dataset.Amount_invested_monthly = dataset.Amount_invested_monthly.replace('__10000__', np.nan).astype(float)
    dataset.Monthly_Balance = dataset.Monthly_Balance.replace('__-333333333333333333333333333__', np.nan).astype(float)
    for var in clean_num:
        dataset[var] = dataset[var].apply(limpieza_cambio)
    ## corrección datos cualitativos
    # reemplazo '_' por missing
    dataset['Credit_Mix'] = dataset['Credit_Mix'].replace(['_', 'Bad'], [np.nan, 'Poor'])
    # conversión a float
    dataset.Credit_History_Age = dataset.Credit_History_Age.str.split(' and ',expand=True)[0].str.replace(' Years', '').astype(float)\
                                 + dataset.Credit_History_Age.str.split(' and ',expand=True)[1].str.replace(' Months', '').astype(float) / 12
    ## normalización de variables
    vars_str = dataset.select_dtypes(exclude=['int', 'float']).columns.tolist()
    print(f'Listado de columnas categoricas: {vars_str}')
    dataset[vars_str] = dataset[vars_str].apply(lambda x: x.str.strip())
    dataset[vars_str] = dataset[vars_str].apply(lambda x: x.str.lower())
    ## codificacion
    # label encoder
    lbl = LabelEncoder()
    dataset['Payment_of_Min_Amount'] = lbl.fit_transform(dataset['Payment_of_Min_Amount'])
    # ordinal encoder
    ordinal_cols = ['Credit_Mix', 'Credit_Score']
    cat_list = [credit_mix_order, credit_score_order]
    for cat, col in zip(cat_list, ordinal_cols):
        ordinal = OrdinalEncoder(categories=[cat])
        dataset[col] = ordinal.fit_transform(dataset[[col]])
    ## manejo de outliers
    outlieres_identificados = detect_outliers(dataset, 0, dataset.columns)
    print(f'Número de outliers detectados {len(outlieres_identificados)}')
    dataset.drop(outlieres_identificados, inplace = True)
    ## imputacion de perdidos
    imputer = KNNImputer(n_neighbors=8)
    imputer_datatset = imputer.fit_transform(dataset)
    imputer_datatset = pd.DataFrame(imputer_datatset, columns=dataset.columns)
    ## separamos el dataset en train, test y val
    y = imputer_datatset.Credit_Score
    X = imputer_datatset.drop('Credit_Score', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=42, test_size = 0.2, stratify = y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test , random_state=42, test_size = 0.25, stratify = y_test)
    ## escalado de variables
    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    X_val   = scaler.transform(X_val)

    print(f'Size X_train: {X_train.shape}, X_test: {X_test.shape}, X_val: {X_val.shape}')

    # gestión de clases desbalanceadas
    # creamos objetos
    sm = SMOTE(random_state=42)
    # retransformamos
    X_res, y_res = sm.fit_resample(X_train, y_train)

    train_dataset = pd.concat([pd.DataFrame(X_res), y_res.reset_index(drop=True)], axis=1)
    test_dataset  = pd.concat([pd.DataFrame(X_test), y_test.reset_index(drop=True)], axis=1)
    vali_dataset  = pd.concat([pd.DataFrame(X_val), y_val.reset_index(drop=True)], axis=1)
    
    train_dataset.columns = dataset.columns
    test_dataset.columns  = dataset.columns
    vali_dataset.columns  = dataset.columns

    train_dataset.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    test_dataset.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
    vali_dataset.to_csv(f"{base_dir}/validation/validation.csv", header=True, index=False)

    # guardar scaler model
    joblib.dump(scaler, "model.joblib")
    with tarfile.open(f"{base_dir}/scaler_model/model.tar.gz", "w:gz") as tar_handle:
        tar_handle.add("model.joblib")
