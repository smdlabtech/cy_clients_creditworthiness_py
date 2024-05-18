import streamlit as st
import os
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

# from streamlit.components import AgGrid, GridOptionsBuilder  ## Manage grid output
import styles_app  ## Module local créer pour le style de l'application (local module)
from styles_app import load_image

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
    # Chemin du répertoire courant
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Chemin du dossier "_data" relatif au répertoire courant
    data_dir = os.path.join(current_dir, "_data")
    
    # Chemin complet du fichier de données
    data_file = os.path.join(data_dir, "Tab2.sas7bdat")
    
    # Définir les options d'affichage de Pandas
    pd.set_option('display.max_column', 18)
    
    # Charger les données
    dfp = pd.read_sas(data_file)
    return dfp


#--------------#
# MAIN FUNCTION
#--------------#
def main():
    
    #---------------------------------#
    # Set PAGE configuration
    #---------------------------------#
    
    # Tuto :
    # https://www.youtube.com/watch?v=nnmBdpvN6u8
    

    #------------------#
    ##  Page settings ##
    st.set_page_config(
        page_title="Creditworthiness",
        # page_icon=":rocket:",  # Peut être un emoji ou un chemin vers un fichier d'image
        page_icon="🏦",  # Emoji représentant une banque
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.example.com/help',
            'Report a bug': 'https://www.example.com/bug',
            'About': "# This is a header. This is an *extremely* cool app!"
        }
        
    )

    
    # App's Title (title, font)
    #st.title("Predicting the creditworthiness of a bank's customers")
    # st.sidebar.header("Sidebar Content")
    st.title("Welcome !")
    st.markdown("<h1 style='text-align: center; color: grey;'>Predicting the Creditworthiness of Bank Customers</h1>", unsafe_allow_html=True)
    
    
    ################################################ 
    # Initialisation des paramètres de mise en forme :
    sidebar = st.sidebar
    block = st.container()



    # Ajout de logo dans le sidebar
    with st.sidebar:
        path_senlab_ia_gen = load_image("senlab_ia_gen_rmv_bgrd.png")  ## load_image : retourne le chemin de l'image
    
        # Afficher l'image avec des paramètres personnalisés
        st.image(path_senlab_ia_gen, 
                caption="SenLab IA",   # Légende de l'image
                width=200,
                use_column_width=True,   # Ignorer la largeur de la colonne et utiliser la largeur spécifiée
                output_format='PNG'   # Format de l'image (par exemple 'JPEG', 'PNG')
                )   

            
    ##st.set_page_config(page_icon="🚀")
    # sidebar.title("Sidebar Panel : ")
    sidebar.markdown("<h1 style='text-align: left; color: grey;'>Sidebar Panel : </h1>", unsafe_allow_html=True)
 

 
    
    #------------------#
    ## 0. Load Data ---#
    # with st.spinner("Loading data"): ## (Indent code)
    dfp = load_data()
    
    # with st.echo('below') : ## Affichage du code
    st.subheader('Load Data : ')
    st.sidebar.subheader('Load Data')  # Modification ici pour utiliser st.sidebar.subheader
    with st.expander("**Preview [Load Data]**"):
        with st.container():
            with st.sidebar.expander("**(Options)**", expanded=True):  # Modification ici pour utiliser st.sidebar.expander
                
                # Créer un conteneur pour les cases à cocher
                show_raw_data = st.checkbox('Raw data')
                data_after_columns_renamed = st.checkbox('Ranemmed columns')
                            
            # Afficher les données brutes et calculer le nombre de variables
            if show_raw_data:
                # Afficher les données brutes et calculer le nombre de variables en une seule ligne
                st.write("**Raw data** :", f"{dfp.shape[1]}", "variables")

                st.write(dfp)

            # Renommer les colonnes
            dfp.rename(columns={
                'varA': 'Incident_r', 'varB': 'Montant_pret', 'varC': 'Montant_hypotheque',
                'varD': 'Val_propriete','varE': 'Motif_pret','varF': 'Profession',
                'varG': 'Nb_annees_travail','varH': 'Nb_report_pret', 'varI': 'Nb_litiges',
                'varJ': 'Age_cred','varK': 'Nb_demandes_cred','varL': 'Ratio_dette_revenu'
            }, inplace=True)
            
            # Afficher les données après le renommage des colonnes
            if data_after_columns_renamed:
                st.write("**Data [renamed columns]** :", f"{dfp.shape[1]}", "variables")
                st.write(dfp)
                st.success("Success !")  # Message de succès
                
            
           
    #---------------------------#
    # 1. Descriptive Statistics
     
    # Section for Descriptive Statistics
    st.subheader('Descriptive Statistics : ')
    with st.expander("**Preview [Descriptive Statistics]**"):
        sidebar.subheader('Descriptive Statistics')
        with st.sidebar.expander("**(Options)**", expanded=True):
            shape = st.checkbox('Shape')
            dtypes = st.checkbox('Dtypes')
            value_counts = st.checkbox('Value counts')

        col1, col2, col3 = st.columns(3)
            
        with col1:
            # Display shape of the data
            if shape:
                st.write("**Load data** (Rows, Columns) :", dfp.shape)

                
            # Display data types of the columns
            if dtypes:
                st.write("**Data Types (dtypes)** : Variables")
                dtypes_df = dfp.dtypes.reset_index()
                dtypes_df.index = dtypes_df.index + 1  # Commencer l'index par 1
                st.write(dtypes_df)


        with col2:
            # Display value counts for specific columns
            # if value_counts:
            #     st.write("**Value Counts** : Categorical variables")
            #     st.write(dfp["Incident_r"].value_counts(dropna=False))
            #     st.write(dfp["Motif_pret"].value_counts(dropna=False))
            #     st.write(dfp["Profession"].value_counts(dropna=False))
            
            if value_counts:
                st.write("**Value Counts** : Categorical variables")
                
                st.write("**frequency** : ")
                st.write(dfp["Incident_r"].value_counts(dropna=False))
                st.write(dfp["Motif_pret"].value_counts(dropna=False))
                st.write(dfp["Profession"].value_counts(dropna=False))
            
                #----
                # Variables catégorielles à afficher
                # categorical_vars = ["Incident_r", "Motif_pret", "Profession"]
                categorical_vars = ["Incident_r"]

                st.write("**distributions** : ")
                for var in categorical_vars:
                    st.write(f"**var** : {var} ")
                    value_counts_data = dfp[var].value_counts(dropna=False)

                    # Créer un countplot avec Seaborn
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(x=var, data=dfp, ax=ax)
                    ax.set_xlabel(var)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {var}')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)




        with col3:
            # Adjust the 'Age_cred' column
            st.write("**Age_cred** : Proportion of customers by credit age")
            st.write("**frequency** : ")
            dfp['Age_cred'] = round(dfp['Age_cred'] / 12, 2)
            dfp_sorted = dfp.sort_values(by='Age_cred', ascending=False)
            st.write(dfp_sorted['Age_cred'])
            
            #--
            # Plotting the distribution of credit age
            st.write("**distributions** : ")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(dfp_sorted['Age_cred'], bins=20, kde=True, ax=ax)
            ax.set_xlabel('Age_cred (years)')
            ax.set_ylabel('Frequency (Customers)')
            ax.set_title('Distribution of Age_cred')
            st.pyplot(fig)
            #--

            
            # Représente les classes d'âge de crédit avec le nombre de clients en intervals
            # Utilise la variable "Age_cred" pour créer des classe d'age d'intervalle 5.
            
            #--
            bins = np.arange(0, dfp_sorted['Age_cred'].max() + 5, 5)
            age_intervals = pd.cut(dfp_sorted['Age_cred'], bins=bins)
            age_counts = age_intervals.value_counts().sort_index()

            # Créer un DataFrame pour les données
            age_data = pd.DataFrame({'Age Interval': age_counts.index.astype(str),
                                    'Number of Customers': age_counts.values})

            # Représenter les classes d'âge de crédit avec le nombre de clients en intervalles
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Age Interval', y='Number of Customers', data=age_data, ax=ax)
            ax.set_xlabel('Credit Age_cred (Intervals : 5 years)')
            ax.set_ylabel('Number of Customers')
            ax.set_title('Distribution of Age_cred (Intervals : 5 years)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            #--
            



    #-----------------------------#
    # Section for Data Explorations
    st.subheader('Data Explorations : ')
    with st.expander("**Preview [Data Explorations]**"):
        with st.container():
            sidebar.subheader('Data Explorations')

            # Create adjustable columns for checkboxes
            with sidebar.expander("**(Options)**", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    descriptions = st.checkbox('Descriptions')
                    pie_charts = st.checkbox('Pie charts')
                    count_plots = st.checkbox('Count plots')

                # with col2:
                    missing_values = st.checkbox('Missing values')
                    missing_values_percentage = st.checkbox('Missing values (pctg)')
                    box_plots = st.checkbox('Box plots')
                    unique_values = st.checkbox('Unique values')
                    descriptive_statistics_before_imputation = st.checkbox('Descriptive statistics before imputation')

        # Show data descriptions
        if descriptions:
            st.write(dfp.describe(include="all"))

        # Show pie charts for categorical variables
        if pie_charts:
            for col in dfp.select_dtypes('object'):
                st.write(f'{col :-<30} {dfp[col].unique()}')
                plt.figure()
                dfp[col].value_counts().plot.pie()
                st.pyplot()

        # Show count plots
        if count_plots:
            sns.countplot(x='Montant_pret', hue='Incident_r', data=dfp)
            st.pyplot()

        # Show missing values information
        if missing_values:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Cols with missing values')
                st.write(dfp.isna().any())
                
            with col2:
                st.subheader('Number of missing values')
                st.write(dfp.isna().sum())

        # Show missing values percentage
        if missing_values_percentage:
            dfp_Na = pd.DataFrame({"Pourcentage_Na": round(dfp.isnull().sum() / (dfp.shape[0]) * 100, 2)})
            st.write(dfp_Na)

        # Show box plots for numerical variables
        if box_plots:
            for col in dfp.select_dtypes('float'):
                plt.figure()
                sns.boxplot(dfp[col])
                st.pyplot()

        # Show unique values for categorical variables
        if unique_values:
            for col in dfp.select_dtypes('object'):
                st.write(f'{col :-<30} {dfp[col].unique()}')

        # Show descriptive statistics before imputation
        if descriptive_statistics_before_imputation:
            cat_var_avant = dfp[['Motif_pret', 'Profession']]
            st.write(cat_var_avant.describe())
            st.write(dfp.describe())
    
    
#---------------------------------------------------------------------------------#

    # Detection of outliers
    st.subheader('Detection of outliers:')
    sidebar.subheader('Detection of outliers')

    with st.expander("**Preview [Detection of outliers]**"):
        
        with st.container():
            # Create adjustable columns for checkboxes
            col1, col2 = st.columns(2)

            #--------#
            with col1:
                box_plots_checkbox = st.sidebar.checkbox('Box plots for outlier detection')

                if box_plots_checkbox:
                    # List of outlier's variables 
                    outliers_info = []

                    for col in dfp.select_dtypes(include=np.number):
                        st.write(f'Box plot for: {col}')
                        fig, ax = plt.subplots()
                        ax.boxplot(dfp[col].dropna())
                        st.pyplot(fig)
                        
                        # Counting outliers
                        outliers_count = len(dfp[col][dfp[col] > dfp[col].quantile(0.75) + 1.5 * (dfp[col].quantile(0.75) - dfp[col].quantile(0.25))] +
                                        dfp[col][dfp[col] < dfp[col].quantile(0.25) - 1.5 * (dfp[col].quantile(0.75) - dfp[col].quantile(0.25))])
                        
                        # Calculate total number of elements per variable
                        total_elements = dfp[col].count()
                        
                        # Calculate percentage of outliers over total elements
                        outliers_percentage = outliers_count / total_elements * 100
                        
                        # Append variable name, number of outliers, total elements, and outlier percentage to the list
                        outliers_info.append({'Variable': col, 
                                            'Outliers count': outliers_count,
                                            'Total elements': total_elements,
                                            'Outliers percentage (%)': outliers_percentage})
                    # Convert the list of dictionaries to DataFrame
                    outliers_df = pd.DataFrame(outliers_info)
                    
                    # Sort DataFrame by Outliers percentage (# Reset DataFrame index)
                    outliers_df_sorted = outliers_df.sort_values(by='Outliers percentage (%)', ascending=False)
                    outliers_df_sorted.reset_index(drop=True, inplace=True)
                    outliers_df_sorted.index = np.arange(1, len(outliers_df_sorted) + 1)





            #--------#
            with col2:
                outliers_checkbox = st.sidebar.checkbox('Outliers (check above first)')
                                
                ## Displaying Outliers
                if outliers_checkbox:
                    # Showing DataFrame with variables, number of outliers, total elements, and outlier percentage detected
                    st.subheader("Outliers proportions (percentage):")
                    st.table(outliers_df_sorted)
                    st.success("Success !")  # Message de succès




    #---------------------------------------------------------------#
    
    # Catching missing variables containing missing values
    sidebar.subheader('Missing values')
    if st.sidebar.checkbox('Missing values frequency'):
        st.subheader('Missing Values Frequency')
        missing_values_freq = dfp.isnull().sum() / len(dfp)
        st.write(missing_values_freq)
    
    if st.sidebar.checkbox('Impute missing values'):
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




    sidebar.subheader('Correlation Analysis')
    st.subheader('Variable Correlation Analysis : ') 
    #  Transformation of Categorical Variables into Numeric variables to prepare the CORRELATION MATRIX (CM)
    if st.sidebar.checkbox('Transform categorical variables'):
        st.subheader('Transformation of Categorical Variables into Numeric variables')
        dfp = pd.get_dummies(dfp, drop_first=True)
        dfp.rename(columns=lambda x: x.replace("'", ""), inplace=True)
        st.write(dfp)

    if st.sidebar.checkbox('correlation matrix'):
        st.subheader('Correlation Matrix')
        corr = dfp.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap='coolwarm')
        st.pyplot()

    if st.sidebar.checkbox('Analyze correlations'):
        st.subheader('Correlation Analysis')
        correlations = dfp.corr()['Incident_r'].sort_values(ascending=False)
        st.write(correlations)
    
    #-----------------------------------------------------------------------------#
    # Separate date into Train and Test datasets : prepraring data for predictions
    #-----------------------------------------------------------------------------#
    sidebar.subheader('Train and Test datasets')
    st.subheader('Dataset preparation: testing and training : ') 
    if st.sidebar.checkbox('Split data into training and test sets'):
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
        
        
    #------------------------------------------#
    # MODELIZATION : Train and evaluate models
    #-----------------------------------------#
    sidebar.subheader('Modelization : ')
    
    
    # MODELS EXPLANATIONS
    
    # Add a markdown area for the model explanations
    st.subheader('Reminder of models definitions') 
    
    #------------------------------------------------------------------------------------------------------------------------
    css_remind_def = styles_app.input_css("style.css")
    st.markdown(f"<style>{css_remind_def}</style>", unsafe_allow_html=True)
    st.markdown("""
    <div class="container">
        <div class="header">K-NN (K-Nearest Neighbors) : </div>
        <div class="content">
            <br>• Easy to understand and implement.
            <br>• No need to make assumptions about the distribution of data, making it useful for nonlinear data.
            <br>• The model does not require training, which can make learning fast.
            <br>Inconveniences:
            <br>• Prediction time may be slow for large datasets.
            <br>• Sensitive to irrelevant variables and the scale of data.
            <br>• It does not provide information on the importance of features.
        </div>
    </div>
    """, unsafe_allow_html=True)

    #------------------------------------------------------------------------------------------------------------------------
    
    

    ## TRAINING AND EVALUATING MODELS
    
    st.subheader('Training and Evaluating Models : ')
    if st.sidebar.checkbox('Train and evaluate models'):
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
            
            #--------------------------------------------------------------------------------#
            # # st.write(f'Model: {name}')
            # st.write(f'**Model:** {name}')  # Utilisation de ** pour rendre le texte en gras
            # st.write('Confusion matrix:')
            # st.write(confusion_matrix(y_test, y_pred))
            # st.write('Error rate:', 1 - accuracy_score(y_test, y_pred))
            # st.write('Precision:', precision_score(y_test, y_pred))
            # st.write('Recall:', recall_score(y_test, y_pred))
            # st.write('F1 score:', f1_score(y_test, y_pred))
            # st.write('Accuracy:', accuracy_score(y_test, y_pred))
            #--------------------------------------------------------------------------------#
            
            # Utilisation de st.markdown pour insérer du HTML avec le style CSS
            st.markdown(f"""
            <div class="container">
                <div class="header">{name}</div>
                <div class="content">
                    Error rate: {1 - accuracy_score(y_test, y_pred)}<br>
                    Precision: {precision_score(y_test, y_pred)}<br>
                    Recall: {recall_score(y_test, y_pred)}<br>
                    F1 score: {f1_score(y_test, y_pred)}<br>
                    Accuracy: {accuracy_score(y_test, y_pred)}
                </div>
            </div>
            """, unsafe_allow_html=True)
                
            st.write('Confusion matrix:')
            st.write(confusion_matrix(y_test, y_pred))
        

            # # Create a DataFrame for the model results
            # model_results = pd.DataFrame({
            #     'Model': [name],
            #     'Confusion matrix': [confusion_matrix(y_test, y_pred)],
            #     'Error rate': [1 - accuracy_score(y_test, y_pred)],
            #     'Precision': [precision_score(y_test, y_pred)],
            #     'Recall': [recall_score(y_test, y_pred)],
            #     'F1 score': [f1_score(y_test, y_pred)],
            #     'Accuracy': [accuracy_score(y_test, y_pred)]
            # })

            # # Display the DataFrame as a table in Streamlit
            # st.table(model_results)
        
        
        ## Features importances  ##
        sidebar.subheader('Features importances : ')
        st.subheader('Features importances : ')
        if name in ['Decision Tree', 'Random Forest']:
            st.write('Feature importances:')
            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
            st.write(feature_importances.sort_values(ascending=False))
    
        
    #------------------------------------------#
    # MODELS : (Comparisons)
    #-----------------------------------------#
    # Select the best model
    # Explain models features importances
    
    
    #-------------------------------------------------------------------------------------------------#
    # TUNINGS : (Improve model performance by using only those variables that best explain the models)
    #-------------------------------------------------------------------------------------------------#
    # Improve model performance by using only those variables that best explain the models

    #------------------------------------------#
    # SCORING : (Predictions)
    #-----------------------------------------#
        
        
        
        

    # #-----------------------------------------#
    # ## Ajout de logo pour les bas de pages : 
    # # Ajout de logo dans le sidebar
    # with st.sidebar:
    #     path_senlab_ia_gen = load_image("senlab_ia_gen_rmv_bgrd.png")  ## load_image : retourne le chemin de l'image
    
    #     # Afficher l'image avec des paramètres personnalisés
    #     st.image(path_senlab_ia_gen, 
    #             caption="SenLab IA",   # Légende de l'image
    #             width=20,
    #             use_column_width=True,   # Ignorer la largeur de la colonne et utiliser la largeur spécifiée
    #             output_format='PNG'   # Format de l'image (par exemple 'JPEG', 'PNG')
    #             )

    # #-----------------------------------------#
    
    

if __name__ == "__main__":
    # st.set_page_config()
    main()
    
