# -*- coding: utf-8 -*-
"""
APPLICATION : Projet de Python 
@author: Daya
         Chris 
         Ali 
"""

######################################
##        Liste des variables       ##

# VarA = Incident_r_r :(qualitative)
# VarE = Motif_pret : (qualitative)
# VarF = Profession : (qualitative)
# VarB = Montant_pret
# VarC = Montant_hypotheque
# VarD = Val_propriete
# VarG = Nb_annees_travail
# VarH = Nb_report_pret
# VarI = Nb_litiges
# VarJ = Age_cred
# Vark = Nb_demandes_cred
# VarL = Ratio_dette_revenu
######################################

import os
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

from os import chdir
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data():
    """
    Load the dataset and set the working directory.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    chdir(r"C:\Users\carrel\Downloads\Projet")   # work directory
    pd.set_option('display.max_column', 12)
    dfp = pd.read_sas(r"C:\Users\carrel\Downloads\Projet\Tab2.sas7bdat")
    return dfp

def rename_variables(dfp):
    """
    Rename the columns of the dataframe.

    Args:
        dfp (pd.DataFrame): The original dataframe.

    Returns:
        pd.DataFrame: The dataframe with renamed columns.
    """
    dfp.rename(columns={
        'varA': 'Incident_r', 'varB': 'Montant_pret', 'varC': 'Montant_hypotheque',
        'varD': 'Val_propriete', 'varE': 'Motif_pret', 'varF': 'Profession',
        'varG': 'Nb_annees_travail', 'varH': 'Nb_report_pret', 'varI': 'Nb_litiges',
        'varJ': 'Age_cred', 'varK': 'Nb_demandes_cred', 'varL': 'Ratio_dette_revenu'
    }, inplace=True)
    return dfp

def data_overview(dfp):
    """
    Display basic information and descriptive statistics of the dataframe.

    Args:
        dfp (pd.DataFrame): The dataframe to be described.
    """
    print(dfp)
    print(dfp.shape)  # (Nb_ligne, Nb_col)
    print(dfp.dtypes)  # (Types des variables)
    print(dfp["Incident_r"].value_counts(dropna=False))  # Occurrence of Incident_r
    print(dfp["Motif_pret"].value_counts(dropna=False))  # Occurrence of Motif_pret
    print(dfp["Profession"].value_counts(dropna=False))  # Occurrence of Profession
    dfp['Age_cred'] = round(dfp['Age_cred'] / 12, 2)  # Convert Age_cred to years
    print(dfp.describe(include="all"))  # Descriptive statistics

def plot_qualitative_distributions(dfp):
    """
    Plot pie charts for qualitative variables.

    Args:
        dfp (pd.DataFrame): The dataframe with qualitative variables.
    """
    for col in dfp.select_dtypes('object'):
        plt.figure()
        dfp[col].value_counts().plot.pie()
        plt.title(col)
        plt.show()

def plot_countplots(dfp):
    """
    Plot countplots for specified variables.

    Args:
        dfp (pd.DataFrame): The dataframe to plot.
    """
    variables = ['Montant_pret', 'Ratio_dette_revenu', 'Val_propriete', 'Nb_report_pret',
                 'Nb_litiges', 'Age_cred', 'Nb_demandes_cred', 'Montant_hypotheque']
    for var in variables:
        sns.countplot(x=var, hue='Incident_r', data=dfp)
        plt.title(var)
        plt.show()

def missing_values_analysis(dfp):
    """
    Analyze missing values in the dataframe.

    Args:
        dfp (pd.DataFrame): The dataframe to analyze.

    Returns:
        pd.DataFrame: A dataframe with the percentage of missing values.
    """
    print(dfp.isna().any())  # NA yes or no
    print(dfp.isna().sum())  # count of NAs
    dfp_Na = pd.DataFrame({"Pourcentage_Na": round(dfp.isnull().sum() / (dfp.shape[0]) * 100, 2)})
    print(dfp_Na)
    return dfp_Na

def plot_missing_values_histogram():
    """
    Plot histogram of missing values for specified variables.
    """
    List_var = ('Incident_r', 'Montant_pret', 'Montant_hypotheque', 'Val_propriete', 'Motif_pret',
                'Profession', 'Nb_annees_travail', 'Nb_report_pret', 'Nb_litiges', 'Age_cred',
                'Nb_demandes_cred', 'Ratio_dette_revenu')

    List_value = [0, 0, 8.65, 1.90, 4.56, 4.70, 8.56, 12.06, 9.90, 5.12, 8.65, 21.74]
    y_pos = np.arange(len(List_var))

    plt.bar(y_pos, List_value)
    plt.xticks(y_pos, List_var, rotation=90)
    plt.ylabel('val. manquantes (%)')
    plt.subplots_adjust(bottom=0.4, top=0.99)
    plt.show()

def plot_outlier_distributions(dfp):
    """
    Plot histograms and boxplots for continuous variables.

    Args:
        dfp (pd.DataFrame): The dataframe with continuous variables.
    """
    for col in dfp.select_dtypes('float'):
        plt.figure()
        sns.distplot(dfp[col])
        plt.title(col)
        plt.show()

    for col in dfp.select_dtypes('float'):
        dfp.boxplot(column=col)
        plt.title(col)
        plt.show()

def impute_missing_values(dfp):
    """
    Impute missing values for categorical and continuous variables.

    Args:
        dfp (pd.DataFrame): The dataframe with missing values.

    Returns:
        pd.DataFrame: The dataframe with imputed values.
    """
    # Imputation for categorical variables
    cat_var = dfp[['Motif_pret', 'Profession']]
    cat_var['Motif_pret'] = cat_var['Motif_pret'].fillna(cat_var['Motif_pret'].mode()[0])
    cat_var['Profession'] = cat_var['Profession'].fillna(cat_var['Profession'].mode()[0])
    dfp['Motif_pret'] = cat_var['Motif_pret']
    dfp['Profession'] = cat_var['Profession']

    # Imputation for continuous variables using KNN
    quant_var = dfp[['Incident_r', 'Montant_pret', 'Montant_hypotheque', 'Val_propriete',
                     'Nb_annees_travail', 'Nb_report_pret', 'Nb_litiges', 'Age_cred',
                     'Nb_demandes_cred', 'Ratio_dette_revenu']]

    # Transformation of outliers to missing values
    u = quant_var.Ratio_dette_revenu
    for i in range(len(u)): 
        if u[i] > 75 or u[i] < 17:
            u[i] = np.nan

    u = quant_var.Val_propriete
    for i in range(len(u)): 
        if u[i] > 400000:
            u[i] = np.nan

    u = quant_var.Age_cred
    for i in range(len(u)): 
        if u[i] > 50:
            u[i] = np.nan

    u = quant_var.Montant_pret
    for i in range(len(u)): 
        if u[i] > 55000:
            u[i] = np.nan

    u = quant_var.Montant_hypotheque
    for i in range(len(u)): 
        if u[i] > 270000:
            u[i] = np.nan

    imputer = KNNImputer(n_neighbors=16)
    quant_var = pd.DataFrame(imputer.fit_transform(quant_var), columns=quant_var.columns)

    # Combining the imputed data
    dfp = pd.concat([quant_var, cat_var], axis=1)
    return dfp

def transform_categorical_to_dummies(dfp):
    """
    Transform categorical variables into dummy variables.

    Args:
        dfp (pd.DataFrame): The dataframe with categorical variables.

    Returns:
        pd.DataFrame: The dataframe with dummy variables.
    """
    cat_variables = dfp[['Motif_pret', 'Profession']]
    cat_dummies = pd.get_dummies(cat_variables, drop_first=True)
    dfp2 = dfp.drop(['Motif_pret', 'Profession'], axis=1)
    dfp2 = pd.concat([dfp2, cat_dummies], axis=1)
    return dfp2

def correlation_matrix(dfp2):
    """
    Compute and display the correlation matrix.

    Args:
        dfp2 (pd.DataFrame): The dataframe for which to compute the correlation matrix.
    """
    corr = dfp2.corr()
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)

def drop_column(dfp2, column):
    """
    Drop a specified column from the dataframe.

    Args:
        dfp2 (pd.DataFrame): The dataframe from which to drop the column.
        column (str): The column to drop.

    Returns:
        pd.DataFrame: The dataframe with the column dropped.
    """
    dfp2.drop([column], axis='columns', inplace=True)
    return dfp2

def prepare_modeling_data(dfp2):
    """
    Prepare the data for modeling by separating the dependent and explanatory variables.

    Args:
        dfp2 (pd.DataFrame): The dataframe to prepare.

    Returns:
        tuple: Tuple containing the matrices X (explanatory) and Y (dependent).
    """
    X = dfp2.iloc[:, 1:14]
    Y = dfp2.iloc[:, 0]
    return X, Y

def split_data(X, Y):
    """
    Split the data into training and testing sets.

    Args:
        X (pd.DataFrame): The explanatory variables.
        Y (pd.Series): The dependent variable.

    Returns:
        tuple: Tuple containing the training and testing sets (X_app, X_test, Y_app, Y_test).
    """
    return train_test_split(X, Y, test_size=0.2, random_state=10)

def logistic_regression_model(X_app, Y_app, X_test, Y_test):
    """
    Train and evaluate a logistic regression model.

    Args:
        X_app (pd.DataFrame): Training set of explanatory variables.
        Y_app (pd.Series): Training set of the dependent variable.
        X_test (pd.DataFrame): Testing set of explanatory variables.
        Y_test (pd.Series): Testing set of the dependent variable.

    Returns:
        float: The accuracy of the model.
    """
    logit = LogisticRegression()
    modele = logit.fit(X_app, Y_app)
    Y_pred = modele.predict(X_test)
    succes = metrics.accuracy_score(Y_test, Y_pred)
    return succes

def knn_model(X_app, Y_app, X_test, Y_test):
    """
    Train and evaluate a k-NN model.

    Args:
        X_app (pd.DataFrame): Training set of explanatory variables.
        Y_app (pd.Series): Training set of the dependent variable.
        X_test (pd.DataFrame): Testing set of explanatory variables.
        Y_test (pd.Series): Testing set of the dependent variable.

    Returns:
        float: The accuracy of the model.
    """
    param_grid = {'n_neighbors': list(range(1, 16))}
    score = 'accuracy'
    knn = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, cv=5, scoring=score)
    digit_knn = knn.fit(X_app, Y_app)
    best_k = digit_knn.best_params_["n_neighbors"]
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=best_k)
    digit_knn = knn.fit(X_app, Y_app)
    error = 1 - digit_knn.score(X_test, Y_test)
    return error

def decision_tree_model(X_app, Y_app, X_test, Y_test):
    """
    Train and evaluate a decision tree model.

    Args:
        X_app (pd.DataFrame): Training set of explanatory variables.
        Y_app (pd.Series): Training set of the dependent variable.
        X_test (pd.DataFrame): Testing set of explanatory variables.
        Y_test (pd.Series): Testing set of the dependent variable.

    Returns:
        float: The accuracy of the model.
    """
    dtree = DecisionTreeClassifier(min_samples_split=5)
    tit_tree = dtree.fit(X_app, Y_app)
    error = 1 - tit_tree.score(X_test, Y_test)
    return error

def random_forest_model(X_app, Y_app, X_test, Y_test):
    """
    Train and evaluate a random forest model.

    Args:
        X_app (pd.DataFrame): Training set of explanatory variables.
        Y_app (pd.Series): Training set of the dependent variable.
        X_test (pd.DataFrame): Testing set of explanatory variables.
        Y_test (pd.Series): Testing set of the dependent variable.

    Returns:
        float: The accuracy of the model.
    """
    forest = RandomForestClassifier(n_estimators=500, min_samples_split=5, oob_score=True)
    forest = forest.fit(X_app, Y_app)
    error = 1 - forest.score(X_test, Y_test)
    return error

def main():
    """
    Main function to run the data analysis and modeling pipeline.
    """
    dfp = load_data()
    dfp = rename_variables(dfp)
    data_overview(dfp)
    plot_qualitative_distributions(dfp)
    plot_countplots(dfp)
    missing_values_analysis(dfp)
    plot_missing_values_histogram()
    plot_outlier_distributions(dfp)
    dfp = impute_missing_values(dfp)
    dfp2 = transform_categorical_to_dummies(dfp)
    correlation_matrix(dfp2)
    dfp2 = drop_column(dfp2, 'Montant_hypotheque')
    X, Y = prepare_modeling_data(dfp2)
    X_app, X_test, Y_app, Y_test = split_data(X, Y)

    log_reg_accuracy = logistic_regression_model(X_app, Y_app, X_test, Y_test)
    print(f"Logistic Regression Accuracy: {log_reg_accuracy}")

    knn_error = knn_model(X_app, Y_app, X_test, Y_test)
    print(f"k-NN Error: {knn_error}")

    dtree_error = decision_tree_model(X_app, Y_app, X_test, Y_test)
    print(f"Decision Tree Error: {dtree_error}")

    forest_error = random_forest_model(X_app, Y_app, X_test, Y_test)
    print(f"Random Forest Error: {forest_error}")

if __name__ == "__main__":
    main()
