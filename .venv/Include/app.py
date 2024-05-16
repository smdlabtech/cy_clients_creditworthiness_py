import streamlit as st
import os
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

# from streamlit.components import AgGrid, GridOptionsBuilder  ## Manage grid output

from os import chdir
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# LOAD DATASET
def load_data():
    chdir(r"C:\Users\DASY\Downloads\cy_clients_creditworthiness_py")   # work directory
    
    # Size of a app windows
    # pd.set_option('display.max_column', 12)
    pd.set_option('display.max_column', 18)
    
    
    dfp = pd.read_sas(r"C:\Users\DASY\Downloads\cy_clients_creditworthiness_py\data\Tab2.sas7bdat")
    return dfp

# MAIN FUNCTION
def main():
    st.title("Predicting the creditworthiness of a bank's customers")
    dfp = load_data()

    #----------------------#
    # 1. Data Explorations
    #----------------------#
    
    if st.checkbox('Show dataframe before columns renamed'):
        st.write(dfp)

    st.subheader('Renommage des variables : ')
    dfp.rename(columns={'varA': 'Incident_r', 'varB': 'Montant_pret', 'varC': 'Montant_hypotheque',
    'varD': 'Val_propriete','varE': 'Motif_pret','varF': 'Profession',
    'varG': 'Nb_annees_travail','varH': 'Nb_report_pret', 'varI': 'Nb_litiges',
    'varJ': 'Age_cred','varK': 'Nb_demandes_cred','varL': 'Ratio_dette_revenu'}, inplace=True)
    
    st.subheader('Descriptive Statistics : ')
    if st.checkbox('Show dataframe after columns renamed'):
        st.write(dfp)
    
    if st.checkbox('Show shape'):
        st.write(dfp.shape)

    if st.checkbox('Show dtypes'):
        st.write(dfp.dtypes)

    if st.checkbox('Show value counts'):
        st.write(dfp["Incident_r"].value_counts(dropna = False))
        st.write(dfp["Motif_pret"].value_counts(dropna = False))
        st.write(dfp["Profession"].value_counts(dropna = False))

    dfp['Age_cred'] = round(dfp['Age_cred'] /12 ,2)

    st.subheader('Data Exploration (DE) : ')
    if st.checkbox('Show descriptions'):
        st.write(dfp.describe(include="all"))

    if st.checkbox('Show pie charts'):
        for col in dfp.select_dtypes('object'):
            st.write(f'{col :-<30} {dfp[col].unique()}')
            plt.figure()
            dfp[col].value_counts().plot.pie()
            st.pyplot()

    if st.checkbox('Show count plots'):
        sns.countplot(x='Montant_pret', hue='Incident_r', data=dfp)
        st.pyplot()

    ## ---
    # if st.checkbox('Show missing values'):
    #     st.write(dfp.isna().any())
    #     st.write(dfp.isna().sum())

    if st.checkbox('Show missing values'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Columns with missing values')
            st.write(dfp.isna().any())
            
        with col2:
            st.subheader('Number of missing values')
            st.write(dfp.isna().sum())



    if st.checkbox('Show missing values percentage'):
        dfp_Na=pd.DataFrame({"Pourcentage_Na" : round(dfp.isnull().sum()/(dfp.shape[0])*100,2)})
        st.write(dfp_Na)

    if st.checkbox('Show box plots'):
        for col in dfp.select_dtypes('float'):
            plt.figure()
            sns.boxplot(dfp[col])
            st.pyplot()

    if st.checkbox('Show unique values'):
        for col in dfp.select_dtypes('object'):
            st.write(f'{col :-<30} {dfp[col].unique()}')

    if st.checkbox('Show descriptive statistics before imputation'):
        cat_var_avant=dfp[['Motif_pret', 'Profession']]
        st.write(cat_var_avant.describe())
        st.write(dfp.describe())
        

    # Detection of outliers
    if st.checkbox('Show box plots for outlier detection'):
        st.subheader('Box Plots for Outlier Detection')
        for col in dfp.select_dtypes(include=np.number):
            st.write(f'Box plot for {col}')
            fig, ax = plt.subplots()
            ax.boxplot(dfp[col].dropna())
            st.pyplot()

    if st.checkbox('Show missing values frequency'):
        st.subheader('Missing Values Frequency')
        missing_values_freq = dfp.isnull().sum() / len(dfp)
        st.write(missing_values_freq)
        
    # if st.checkbox('Show missing values frequency'):
    #     col1, col2 = st.beta_columns(2)

    #     with col1:
    #         st.subheader('Missing Values Frequency')

    #     with col2:
    #         # Ajoutez ici le code pour l'autre colonne
    #         missing_values_freq = dfp.isnull().sum() / len(dfp)
    #         st.write(missing_values_freq)



    # Catching missing variables containing missing values
    if st.checkbox('Impute missing values'):
        st.subheader('Imputation of Missing Values')
        st.write('Before imputation:')
        st.write(dfp.isnull().sum())

        # Imputation of missing values for numerical columns
        imputer = KNNImputer(n_neighbors=5)
        dfp[dfp.select_dtypes(include=np.number).columns] = imputer.fit_transform(dfp.select_dtypes(include=np.number))

        # Imputation of missing values for categorical columns
        for col in dfp.select_dtypes(include='object').columns:
            dfp[col].fillna(dfp[col].mode()[0], inplace=True)

        st.write('After imputation:')
        st.write(dfp.isnull().sum())


    st.subheader('Variable Correlation Analysis : ') 
    #  Transformation of Categorical Variables into Numeric variables to prepare the CORRELATION MATRIX (CM)
    if st.checkbox('Transform categorical variables'):
        st.subheader('Transformation of Categorical Variables into Numeric variables')
        dfp = pd.get_dummies(dfp, drop_first=True)
        dfp.rename(columns=lambda x: x.replace("'", ""), inplace=True)
        st.write(dfp)

    if st.checkbox('Show correlation matrix'):
        st.subheader('Correlation Matrix')
        corr = dfp.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap='coolwarm')
        st.pyplot()

    if st.checkbox('Analyze correlations'):
        st.subheader('Correlation Analysis')
        correlations = dfp.corr()['Incident_r'].sort_values(ascending=False)
        st.write(correlations)
    
    #-----------------------------------------------------------------------------#
    # Separate date into Train and Test datasets : prepraring data for predictions
    #-----------------------------------------------------------------------------#
    st.subheader('Dataset preparation: testing and training : ') 
    if st.checkbox('Split data into training and test sets'):
        st.subheader('Splitting Data')
        X = dfp.drop('Incident_r', axis=1)
        y = dfp['Incident_r']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write('Training set:')
        st.write(X_train)
        st.write(y_train)
        st.write('Test set:')
        st.write(X_test)
        st.write(y_test)
        
        
        #---------------------------------------------#
        # MODELS EXPLANATIONS
        #----------------------------------------------#
        # Add a markdown area for the model explanations
        st.subheader('Reminder of models definitions : ') 
        # st.subheader('A few reminders : ') 
    
        model_explanations = """
        K-NN (K-Nearest Neighbors) :
        Avantages :
        - Simple à comprendre et à mettre en œuvre.
        - Pas besoin de faire des hypothèses sur la distribution des données, ce qui le rend utile pour les données non linéaires.
        - Le modèle ne nécessite pas d'entraînement, ce qui peut rendre l'apprentissage rapide.
        Inconvénients :
        - Le temps de prédiction peut être lent pour les grands ensembles de données.
        - Sensible aux variables non pertinentes et à l'échelle des données.
        - Il ne fournit pas d'informations sur l'importance des caractéristiques.
        ...
        """
        st.markdown(model_explanations)
    
    
    #------------------------------------------#
    # MODELIZATION : Train and evaluate models
     #-----------------------------------------#
    
    if st.checkbox('Train and evaluate models'):
        st.subheader('Training and Evaluating Models')
        
        # Variable to predict (Or for SCORING ANALYSIS)
        # Separations between predictor and predicted variables.
        X = dfp.drop('Incident_r', axis=1)
        y = dfp['Incident_r']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            
            ### We need to fillna for missing values for (K-NN) and (Logistic Regression)   
            # (2 MODELS : 2 models left to test)
            # 'K-NN': KNeighborsClassifier(),
            # 'Logistic Regression': LogisticRegression(),
            
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write(f'Model: {name}')
            st.write('Confusion matrix:')
            st.write(confusion_matrix(y_test, y_pred))
            st.write('Error rate:', 1 - accuracy_score(y_test, y_pred))
            st.write('Precision:', precision_score(y_test, y_pred))
            st.write('Recall:', recall_score(y_test, y_pred))
            st.write('F1 score:', f1_score(y_test, y_pred))
            st.write('Accuracy:', accuracy_score(y_test, y_pred))
        
        ## Features importances  ##
        if name in ['Decision Tree', 'Random Forest']:
            st.write('Feature importances:')
            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
            st.write(feature_importances.sort_values(ascending=False))
    
        #-----------------------------#
        # REMAINS TO DO
        
        # Models comparisons
        # Explain models features importances
        # Improve model performance by using only those variables that best explain the models
        # 
        #-----------------------------#
        
        
    #------------------------------------------#
    # SCORING : (Predictions)
     #-----------------------------------------#
        


if __name__ == "__main__":
    main()
    
    