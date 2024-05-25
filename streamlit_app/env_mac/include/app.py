import streamlit as st
import streamlit.components.v1 as components
# from streamlit_carousel import carousel

import os
import time
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from itertools import combinations
from scipy.stats import chi2_contingency

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


## From streamlit.components import AgGrid, GridSelect optionsBuilder  ## Manage grid output
## Module local créer pour le style de l'application (local module)
import styles_app


###1. Add html component (file)
def display_html(file_name):
    try:
        html_content = styles_app.load_html(file_name)
        st.markdown(html_content, unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.error(f"Erreur: {e}")

        

###2. Display JavaScript
def display_js(file_name):
    try:
        js_content = styles_app.load_js(file_name)
        st.markdown(js_content, unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.error(f"Erreur: {e}")
        

def apply_js(file_name):
    try:
        js_content = styles_app.load_js(file_name)
        # Utiliser st.write pour appliquer le code JavaScript
        st.write(js_content, unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.error(f"Erreur: {e}")




###3. LOAD DATASET
@st.cache_data     ## Docorateur qui mets les données en cache.
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "_data")
    data_file = os.path.join(data_dir, "Tab2.sas7bdat")
    pd.set_option('display.max_column', 18)
    dfp = pd.read_sas(data_file)
    return dfp


# Tuto :
# https://www.youtube.com/watch?v=nnmBdpvN6u8

#--------------#
# MAIN FUNCTION
#--------------#
###4. main
def main():

    ### Set PAGE configuration
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
    # st.title("Welcome !")
    st.markdown("<h1 style='text-align: center; color: grey;'>Predicting the Creditworthiness of Bank Customers</h1>", unsafe_allow_html=True)
    
    
    ### Initialisation du style css dans l'application
    css_styles = styles_app.load_css("style.css")
    
    
    ################################################ 
    # Initialisation des paramètres de mise en forme :
    sidebar = st.sidebar
    block = st.container()


    # Ajout de logo dans le sidebar
    with st.sidebar:
        styles_app.load_img("senlab_ia_gen_rmv_bgrd.png", caption="SenLab IA", width=100, use_column_width=True, output_format='PNG')

            
    ##st.set_page_config(page_icon="🚀")
    # sidebar.title("Sidebar Panel : ")
    sidebar.markdown("<h1 style='text-align: left; color: grey;'>Sidebar Panel : </h1>", unsafe_allow_html=True)
    ## st.button("do stuff then close expander",on_click=toggle_closed)
    
    #######################


    
    
    
    #######################    


    

    #------------------#
    ## 0. Load Data ---#
    # with st.spinner("Loading data"): ## (Indent code)
    dfp = load_data()
    
    # with st.echo('below') : ## Affichage du code
    st.subheader('Load Data : ')
    st.sidebar.subheader('1. Load Data')  # Modification ici pour utiliser st.sidebar.subheader    
    with st.expander("**Preview [Load Data]**"):
        with st.container():
            with st.sidebar.expander("**(Select options)**", expanded=True):  # Modification ici pour utiliser st.sidebar.expander
                
                # Créer un conteneur pour les cases à cocher
                show_raw_data = st.checkbox('Raw data', value=True)
                data_after_columns_renamed = st.checkbox('Ranemmed columns', value=True)
                            
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
        sidebar.subheader('2. Descriptive Statistics')
        
        # Create adjustable columns for checkboxes => sidebar widgets (checkboxes)
        with st.sidebar.expander("**(Select options)**", expanded=True):
            shape = st.checkbox('Shape', value=True)
            dtypes = st.checkbox('Dtypes', value=True)
            value_counts = st.checkbox('Value counts', value=True)

        # Format outputs results (visuals)
        col1, col2, col3 = st.columns(3)
        
        #--------#
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
                
                ## Statistic Test : Chix-2 test                
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Statistic Test : Chix-2 test 💡</div>
                    <div class="content">
                        <ul>
                            <li>
                                The chi-square test is used to identify whether there is a statistically significant relationship between <strong> two categorical (qualitative) variables</strong>. 
                                A contingency table is constructed by crossing the modalities of the two variables.<br><br> 
                            </li>
                            <li>
                                The test compares the observed numbers in each cell of the table with the theoretical numbers that would be expected if the two variables were independent (unrelated). 
                                If the differences between observed and theoretical numbers are significant, this suggests a <strong>dependency</strong> between the variables.<br><br>
                            </li>
                            <li>
                                The chi-square statistic quantifies this overall difference. 
                                A p-value associated with the chi-square is then calculated. 
                                If the p-value is below the chosen significance level <strong>(usually 5%)</strong>, the hypothesis of independence is rejected, and it is concluded that there is a statistical link between the two variables.<br><br>
                            </li>
                            <li>
                                In short, the chi-square test evaluates whether the observed distribution of the numbers in the contingency table can reasonably be explained by simple chance (independence), or whether a relationship between the variables must be concluded.
                            </li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # (Chix 2 Test)  Catégorisation en variables qualitatives
                qualitative_columns = dfp.select_dtypes(include=['object', 'category']).columns.tolist()
                st.write(f"**[Qualitatives variables]** : {qualitative_columns}")
                    
                row_var = dfp[qualitative_columns[0]]
                col_var = dfp[qualitative_columns[1]]
                contingency_table = pd.crosstab(row_var, col_var)
                st.dataframe(contingency_table)
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                st.write(f"**[Statistic Test Results]** : {chi2_stat},P-value : {p_value}")
                if p_value > 0.05 : st.write(f"The 2 variables are **independent**.")
                else : st.write(f"We conclude that there is a **statistical link** between the two variables. So these 2 variables are not independent.", color = "blue")
                                

        #--------#
        with col2:
            if value_counts:
                st.write("**Value Counts** : Categorical variables")
                
                st.write("**frequency** : ")
                st.write(dfp["Incident_r"].value_counts(dropna=False))
                st.write(dfp["Motif_pret"].value_counts(dropna=False))
                st.write(dfp["Profession"].value_counts(dropna=False))
            
                #---
                # Variables catégorielles à afficher (améliorer cette partie)
                # categorical_vars = ["Incident_r", "Motif_pret", "Profession"]
                categorical_vars = ["Incident_r"]
                #---

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



        #--------#
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
            sidebar.subheader('a. data explorations')

            # Create adjustable columns for checkboxes  => Sidebar widgets (checkboxes)
            with sidebar.expander("**(Select options)**", expanded=True):
                descriptions = st.checkbox('Descriptions', value=True)
                pie_charts = st.checkbox('Pie charts', value=True)
                count_plots = st.checkbox('Count plots', value=True)
                missing_values = st.checkbox('Missing values',value=True)
                missing_values_percentage = st.checkbox('Missing values (pctg)', value=True)
                box_plots = st.checkbox('Box plots', value=True)
                unique_values = st.checkbox('Unique values', value=True)
                descriptive_statistics_before_imputation = st.checkbox('Descriptive statistics before imputation', value=True)
                chix_test_qualitative_var = st.checkbox('Chix-2 test', value=True)

        # Show data descriptions
        if descriptions:
            st.write("**Descriptions** :")
            st.write(dfp.describe(include="all"))

        # Show pie charts for categorical variables
        if pie_charts:
            st.write("**Pie charts** :")
            for col in dfp.select_dtypes('object'):
                st.write(f'{col :-<30} {dfp[col].unique()}')
                fig, ax = plt.subplots()
                dfp[col].value_counts().plot.pie(ax=ax)
                st.pyplot(fig)

        # Show count plots
        if count_plots:
            st.write("**Count plots** :")
            fig, ax = plt.subplots()
            sns.countplot(x='Montant_pret', hue='Incident_r', data=dfp, ax=ax)
            st.pyplot(fig)
            
            #-------------------------------------------------------------------
            # sns.countplot(x='Montant_pret', hue='Incident_r', data=dfp)
            # sns.countplot(x='Ratio_dette_revenu', hue='Incident_r', data=dfp)
            # sns.countplot(x='Val_propriete', hue='Incident_r', data=dfp)
            # sns.countplot(x='Nb_report_pret', hue='Incident_r', data=dfp)
            # sns.countplot(x='Nb_litiges', hue='Incident_r', data=dfp)
            # sns.countplot(x='Age_cred', hue='Incident_r', data=dfp)
            # sns.countplot(x='Nb_demandes_cred', hue='Incident_r', data=dfp)
            # sns.countplot(x='Ratio_dette_revenu', hue='Incident_r', data=dfp)
            # sns.countplot(x='Montant_hypotheque', hue='Incident_r', data=dfp)
            #-------------------------------------------------------------------------

        # Show missing values information
        if missing_values:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Missing values** :")
                st.subheader('Variables with missing values')
                st.write(dfp.isna().any())
                

                List_var=('Incident_r','Montant_pret','Montant_hypotheque','Val_propriete','Motif_pret',
                'Profession','Nb_annees_travail','Nb_report_pret','Nb_litiges','Age_cred',
                'Nb_demandes_cred','Ratio_dette_revenu')
                List_value=[0,0,8.65,1.90,4.56,4.70,8.56,12.06,9.90,5.12,8.65,21.74]
                y_pos=np.arange(len(List_var))
                
                # Figure and axis creation in the same workspace
                fig, ax = plt.subplots(figsize=(10, 6))  # Ajustez la taille de la figure si nécessaire
                ax.barh(y_pos, List_value)  # Utilisez ax pour placer le graphique horizontalement
                ax.set_yticks(y_pos)
                ax.set_yticklabels(List_var, rotation=0)  # Rotation ajustée pour mieux voir les labels
                ax.set_xlabel('Number of missing values (%)')
                ax.set_title('Distribution of missing values')
                plt.subplots_adjust(left=0.15, bottom=0.25, right=0.95, top=0.85, wspace=0.35, hspace=0.35)
                
                # Affichage de la figure avec Streamlit
                st.pyplot(fig)
            
            #--------#        
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
                fig, ax = plt.subplots()
                sns.boxplot(dfp[col], ax=ax)
                st.pyplot(fig)

        # Show unique values for categorical variables
        if unique_values:
            for col in dfp.select_dtypes('object'):
                st.write(f'{col :-<30} {dfp[col].unique()}')

        # Show descriptive statistics before imputation
        if descriptive_statistics_before_imputation:
            cat_var_avant = dfp[['Motif_pret', 'Profession']]
            st.write(cat_var_avant.describe())
            st.write(dfp.describe())
            
        # chix_test_qualitative_var = st.checkbox('Chix-2 test', value=True)
        # if chix_test_qualitative_var:
            

    
    #----------------------#
    # Detection of outliers
    st.subheader('Detection of outliers:')
    with st.expander("**Preview [Detection of outliers]**"):
        with st.container():
            sidebar.subheader('b. detection of outliers')
            
            # Create adjustable columns for checkboxes  => Sidebar widgets (checkboxes)
            with sidebar.expander("**(Select options)**", expanded=True):
                box_plots_checkbox = st.checkbox('Box plots for outlier detection', value=True)
                outliers_checkbox = st.checkbox('Outliers (check above first)', value=True)


            # Create adjustable columns for checkboxes
            col1, col2 = st.columns(2)

            #--------#
            with col1:
                if box_plots_checkbox:
                    outliers_info = []      # List of outlier's variables 
 
                    for col in dfp.select_dtypes(include=np.number):
                        st.write(f'**Box plot** : {col}')
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
                if outliers_checkbox:
                    st.subheader("Outliers proportions (percentage):")
                    st.table(outliers_df_sorted)
                    st.success("Success !")  # Message de succès





    # Catching missing variables containing missing values
    st.subheader('Missing values : Processing')
    with st.expander("**Preview [Missing values]**"):
        with st.container():
            sidebar.subheader('c. Missing values (processing)')

            # Create adjustable columns for checkboxes  => Sidebar widgets (checkboxes)
            with sidebar.expander("**(Select options)**", expanded=True):
                missing_values_freq = st.checkbox('Missing values (review)', value=True, key='missing_values_freq')
                impute_missing_values = st.checkbox('Impute missing values', value=True, key='impute_missing_values')

            # Create adjustable columns for checkboxes
            col1, col2 = st.columns(2)
            
            #--------#
            with col1:
                if missing_values_freq:
                    # Display missing values frequency
                    st.write('Missing Values (review):')
                    missing_values_freq = dfp.isnull().sum() / len(dfp)
                    st.write(missing_values_freq)
            
            
                    # Visualization of missing values frequency
                    fig, ax = plt.subplots()
                    sns.barplot(x=missing_values_freq.values, y=missing_values_freq.index, orient='h', ax=ax)
                    ax.set_title('Missing Values (review)')
                    ax.set_xlabel('Frequency')
                    ax.set_ylabel('Variables')
                    st.pyplot(fig)   
                    
                    
                    ## Definition of mode
                    st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class="container">
                        <div class="header">Handling Missing Data : 💡</div>
                        <div class="content">
                            The <strong>mode imputation</strong> involves replacing missing values of a <strong>categorical (qualitative) variable</strong> with the most frequent value, that is, the mode, among the present values of this variable. 
                            <br> This is a simple and intuitive method for handling missing data in qualitative variables.
                            <br> <br>For a <strong>quantitative (numerical) variable</strong>, a similar approach would be to use the <strong>mean</strong> or <strong>median</strong> of the present values to impute the missing values. However, mean imputation can be sensitive to outliers and significantly alter the distribution of the data. Therefore, the median is generally preferred because it is more robust to extreme values.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            
            #--------#
            with col2:
                if impute_missing_values:
                    # Display missing values before imputation
                    st.write('Before imputation: count')
                    st.write(dfp.isnull().sum())

                    # Imputation of missing values for numerical columns
                    imputer = KNNImputer(n_neighbors=5)
                    dfp[dfp.select_dtypes(include=np.number).columns] = imputer.fit_transform(dfp.select_dtypes(include=np.number))

                    # Imputation of missing values for categorical columns
                    for col in dfp.select_dtypes(include='object').columns:
                        dfp[col].fillna(dfp[col].mode()[0], inplace=True)

                    # Display missing values after imputation
                    st.write('After imputation: count ')
                    st.write(dfp.isnull().sum())



    #------------------------------------#
    # Subsection for Correlation Analysis
    st.subheader('Correlation Analysis')
    with st.expander("Preview [Correlation Analysis]"):
        with st.container():
            sidebar.subheader('3. Correlation Analysis')

            # Create adjustable columns for checkboxes => Sidebar widgets (checkboxes)
            with sidebar.expander("**(Select options)**", expanded=True):
                transform_categorical_variables = st.checkbox('Transform categorical variables', value=True)
                pca_analysis = st.checkbox('PCA Analysis', value=True)
                correlation_matrix = st.checkbox('Correlation matrix', value=True)
                analyze_correlations = st.checkbox('Analyze correlations', value=True)

            # Option to Transform Categorical Variables
            if transform_categorical_variables:
                st.subheader('Transformation of Categorical Variables into Numeric variables [0, 1] : One-hot-dummies')
                # st.write('Transformation of Categorical Variables into Numeric variables [0, 1] : One-hot-dummies')
                st.write("**Transformed dataframe**(Rows, Cols) :", f"{dfp.shape}")
                dfp = pd.get_dummies(dfp, drop_first=True)
                dfp.rename(columns=lambda x: x.replace("'", ""), inplace=True)
                st.write(dfp)

            # Create adjustable columns for checkboxes
            col1, col2 = st.columns(2)

            #--------#
            with col1:
                
                if correlation_matrix:
                    
                    ## Model: Principal Component Analysis
                    st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class="container">
                        <div class="header">Correlation Matrix (CM): 💡</div>
                        <div class="content">
                            <p>A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The value is between -1 and 1.</p>
                            <p>Here is a more detailed definition of a correlation matrix:</p>
                            <ul>
                                <li><strong>Correlation Coefficient:</strong> The correlation coefficient is a measure of the linear relationship between two variables. A value of 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation.</li>
                                <li><strong>Symmetric Matrix:</strong> The correlation matrix is symmetric because the correlation between variable A and variable B is the same as the correlation between variable B and variable A.</li>
                                <li><strong>Diagonal Values:</strong> The diagonal values of a correlation matrix are always 1 because each variable is perfectly correlated with itself.</li>
                                <li><strong>Usage:</strong> Correlation matrices are used in various statistical analyses to understand the relationships between variables, to detect multicollinearity in regression analyses, and to serve as input for other analyses like PCA.</li>
                            </ul>
                            <p>In summary, a correlation matrix provides a concise summary of the relationships between multiple variables, helping to identify patterns and dependencies in the data.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # st.write("**(Raw data)** :", f"{dfp.shape[1]}", "variables")
                    st.subheader('Correlation Matrix (CM) :')
                    corr = dfp.corr()
                    fig, ax = plt.subplots(figsize=(10, 10))
                    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap='coolwarm')
                    st.pyplot(fig)  # Pass the figure to st.pyplot()

                
                
                if analyze_correlations:
                    # Correlations var
                    st.subheader('Correlation (corr. with predictive var) :')
                    correlations = dfp.corr()['Incident_r'].sort_values(ascending=False)
                    st.write(correlations)
                    
                    # Highly Correlated Variable Pairs
                    st.subheader('Highly Correlated Variable Pairs')
                    correlations = dfp.corr()
                    all_pairs = list(combinations(correlations.index, 2))
                    high_corr_pairs = [(pair[0], pair[1], correlations.loc[pair[0], pair[1]]) for pair in all_pairs if abs(correlations.loc[pair[0], pair[1]]) > 0.5]
                    high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Variable1', 'Variable2', 'Correlation'])
                    high_corr_df = high_corr_df.sort_values(by='Correlation', ascending=False)
                    st.write(high_corr_df)

            
            
            
            #------------------------------------#
            # Option to Display Correlation Matrix
            with col2:
                
                if pca_analysis:
                    ## Model: Principal Component Analysis
                    st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class="container">
                        <div class="header">Principal Component Analysis (PCA) : 💡</div>
                        <div class="content">
                            <p>Principal Component Analysis (PCA) is a statistical method used to transform multivariate data into a set of linearly uncorrelated variables, called principal components. The goal of PCA is to reduce the dimensionality of the data while preserving as much information as possible.</p>
                            <p>Here is a more detailed definition of PCA:</p>
                            <ul>
                                <li><strong>Data Transformation:</strong> PCA takes a set of multivariate data and transforms it into a new set of variables, called principal components, which are linear combinations of the original variables.</li>
                                <li><strong>Dimensionality Reduction:</strong> PCA reduces the dimensionality of the data by projecting the observations into a new space of variables that captures the maximum variance of the data.</li>
                                <li><strong>Linear Independence:</strong> The principal components are uncorrelated with each other, meaning they capture different and independent aspects of the data variation.</li>
                                <li><strong>PCA Usage:</strong> PCA is widely used for data visualization, dimensionality reduction, pattern detection, data compression, and preparing data for other analysis techniques.</li>
                            </ul>
                            <p>In summary, PCA is a powerful technique for exploring and analyzing multivariate data by reducing its complexity while preserving as much information as possible.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                



                #------------------------------------
                # Subsection for Correlation Analysis
                st.subheader('Principal Component Analysis (PCA)')
                # L'ACP, ou l'Analyse en Composantes Principales, 
                # est une méthode statistique utilisée pour réduire la dimensionnalité d'un ensemble de données 
                # tout en conservant autant que possible la variabilité présente dans ces données. 
                # Voici ses principaux objectifs et applications :
                
                #-------------------#
                # # Critrere du Coude
                # plot(res.pca$eig[,1], type="o", main='Eboulis de valeurs propres',
                #     xlab = 'dimensions', ylab = 'valeurs propres')


                # #-------------------#
                # # Critrere de Kaiser
                # res.pca$eig[,1:3]


                # #-----------------------#
                # # GRAPHIQUE DES INDIVIDUS
                # plot.PCA(res.pca, axes = c(1,2),
                #         choix = 'ind',
                #         label = 'var',
                #         new.plot = TRUE
                # )


                # #-----------------------#
                # # GRAPHIQUE DES VARIABLES
                # plot.PCA(res.pca, axes = c(1,2),
                #         choix = 'var',
                #         new.plot = TRUE,
                #         col.var = 'black',
                #         label = 'var'
                # )
                #-----------------------------------------------------------------------------------------------



    
    #-----------------------------------------------------------------------------#
    # MODELIZATION
    #-----------------------------------------------------------------------------#
    st.subheader('Modelization :')
    with st.expander("**Preview [Split Data]**"):
        with st.container():
            sidebar.subheader('4. Modelization')
            
            with sidebar.expander("**(Select options)**", expanded=True):
                split_data = st.checkbox('Train and Test datasets', value=True)
                train_models = st.checkbox('Train and evaluate models', value=True)
                the_best_model = st.checkbox('Choose the best model', value=True)
                
        if split_data:
            # st.subheader('Train and Test datasets :')
            X = dfp.drop('Incident_r', axis=1)
            y = dfp['Incident_r']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create adjustable columns for checkboxes
            st.subheader('Train and Test datasets')
            col1, col2 = st.columns(2)
            
            with col1:
                st.write('**Training dataset :** *Explanatory variables*')
                st.write("(Rows, Cols) :", f"{X_train.shape}")
                st.write(X_train)
                st.write("**Training dataset** : *Explained variable*")
                st.write("(Rows, Cols) :", f"{y_train.shape}")
                st.write(y_train)
                
            with col2:
                st.write('**Test dataset :** *Explanatory variables*')
                st.write("(Rows, Cols) :", f"{X_test.shape}")
                st.write(X_test)
                st.write("**Test dataset** : *Explained variable*")
                st.write("(Rows, Cols) :", f"{y_test.shape}")
                st.write(y_test)



        #-----------------------------------------#
        # TRAINING AND EVALUATING MODELS in Col1
        #-----------------------------------------#
        
        st.subheader('Training and Evaluating Models : 🚀 ')
        col1, col2 = st.columns(2)
        
        #---------#
        with col1 : 
            if train_models:
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
                                    
                    # Utilisation de st.markdown pour insérer du HTML avec le style CSS
                    ## ML metrics
                    st.markdown(f"""
                    <div class="container">
                        <div class="header">{name} : metrics 📊</div>
                        <div class="content">
                            Error rate: {1 - accuracy_score(y_test, y_pred)}<br>
                            Precision: {precision_score(y_test, y_pred)}<br>
                            Recall: {recall_score(y_test, y_pred)}<br>
                            F1 score: {f1_score(y_test, y_pred)}<br>
                            Accuracy: {accuracy_score(y_test, y_pred)}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    #------------------------------------------------------------#
                    # ## Matrice de confusion : (La formule marche très bien !!!)
                    
                    # st.write(f"**{name}** : ", 'Confusion matrix')
                    CM = confusion_matrix(y_test, y_pred)
                    st.write(CM)
                    

                    #------------------------------------------------------------#
                    # from sklearn.datasets import load_digits
                    # from sklearn.model_selection import train_test_split, GridSearchCV
                    # from sklearn.neighbors import KNeighborsClassifier
                    # from sklearn.metrics import confusion_matrix, accuracy_score

                    # # Charger l'ensemble de données
                    # # digits = load_digits()
                    # # X = digits.data
                    # # y = digits.target

                    # # # Diviser l'ensemble de données en jeux d'entraînement et de test
                    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # #-----------------------------------------------#
                    # # Fixer les valeurs des hyperparamètres à tester
                    # param_grid = {'n_neighbors': list(range(1, 16))}

                    # # Choisir un score à optimiser, ici l'accuracy
                    # score = 'accuracy'

                    # # Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
                    # knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring=score)

                    # # Optimiser le classifieur sur le jeu d'entraînement
                    # knn.fit(X_train, y_train)

                    # # Afficher le(s) hyperparamètre(s) optimal(s)
                    # st.write(f"Meilleur(s) hyperparamètre(s) trouvé(s) : {knn.best_params_}")

                    # # Estimation de l’erreur de prévision
                    # error_rate = 1 - knn.score(X_test, y_test)
                    # st.write(f"Taux d'erreur de prévision : {error_rate}")

                    # # Prévision
                    # y_pred = knn.predict(X_test)

                    # # Matrice de confusion
                    # mat_conf = confusion_matrix(y_test, y_pred)
                    # st.write(f"Matrice de confusion :\n{mat_conf}")

                    # # Afficher la matrice de confusion avec matplotlib
                    # fig, ax = plt.subplots(figsize=(10, 7))
                    # ax.matshow(mat_conf)
                    # st.pyplot(fig)
                    # #-----------------------------------------------#
                        
    
    
    
            #------------------------------------------#
            # MODELS & DEFINITIONS : in Col2
            #------------------------------------------#            
            with col2 : 
                ## Model : K-NN (K-Nearest Neighbors)
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Statistical Model : 💡</div>
                    <div class="content">
                        <p>A statistical model is a mathematical representation of observed data. It describes the relationships between different variables in the data using statistical concepts and techniques.</p>
                        <p>Here is a more detailed definition of a statistical model:</p>
                        <ul>
                            <li><strong>Mathematical Representation:</strong> A statistical model uses mathematical equations to represent the relationships between variables. These equations are based on statistical principles.</li>
                            <li><strong>Parameters:</strong> The model includes parameters that quantify the strength and nature of the relationships between variables. These parameters are estimated from the data.</li>
                            <li><strong>Assumptions:</strong> Statistical models are based on certain assumptions about the data, such as normality, independence, and linearity. These assumptions must be checked to ensure the validity of the model.</li>
                            <li><strong>Usage:</strong> Statistical models are used for various purposes, including prediction, inference, hypothesis testing, and data exploration. They help in understanding the underlying patterns and relationships in the data.</li>
                        </ul>
                        <p>In summary, a statistical model provides a structured and formalized way to analyze data, making it possible to draw meaningful conclusions and make informed decisions based on the data.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                
                ## Model : K-NN (K-Nearest Neighbors)
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">K-NN (K-Nearest Neighbors) : 💡</div>
                    <div class="content">
                        The KNN (K-nearest neighbor) model is a supervised machine learning method that can be used for classification and regression. 
                        It works by identifying the “k” data points closest to a given point, and assigning to that point the class or average class value of those k points. 
                        This process predicts the class or value of a data point based on its nearest neighbors in the data space.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                ## Model : Decision Tree                
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Decision Tree : 💡</div>
                    <div class="content">
                        <i class="fa-solid fa-square"></i>
                        <img src="./assets/img/stars.png" alt='Decision Tree Icon' style='width: 50px; height: 50px;'>
                        A decision tree is a supervised learning technique that builds a prediction model by forming a hierarchical diagram of decisions based on tests performed on input variables. 
                        Given a set of labeled data, the decision tree learns to make decisions by dividing into branches based on specific criteria. 
                        Each division (or node) in the tree represents a test on a variable, and each leaf of the tree represents a final conclusion or prediction. 
                        Decision trees are popular for their ability to handle complex, non-linear problems, offering a clear visualization of decision-making processes.
                    </div>
                </div>
                """, unsafe_allow_html=True)        
            
            
                ## Model : Logistic Regression
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Logistic Regression : 💡</div>
                    <div class="content">
                        The Logistic regression is a statistical method used to predict the probability of an observation belonging to a certain category. 
                        Unlike linear regression, which predicts a continuous value, logistic regression is used to predict a binary probability, such as yes or no, based on input variables. 
                        It is based on the logistic function, which transforms continuous values into probabilities between 0 and 1, making logistic regression ideal for binary classification problems.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                ## Model : Random Forest
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Random Forest : 💡</div>
                    <div class="content">
                        The Random Forest model is a machine learning method that generates a set of numerous decision trees. 
                        These trees are combined to avoid overlearning and produce more accurate predictions. 
                        It is a supervised algorithm that can be used for classification and regression tasks.
                        Random Forest constructs several decision trees and merges them to obtain a more accurate and stable prediction.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                
                # st.subheader("Model metrics", color=css_styles)
                st.subheader("Metrics")
                
                
                ## 1. Confusion Matrix : Definition
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Confusion Matrix : 💡</div>
                    <div class="content">
                        The confusion matrix is a tabular representation where :
                        <ul>
                            <li>Each row represents the instances of a real class.</li>
                            <li>Each column represents the instances of a class predicted by the model.</li>
                            <li>The cells of the matrix indicate the number of instances classified in each combination of real class and predicted class.</li>
                        </ul>
                        There are generally four categories :
                        <ul>
                            <li>True Positives (TP): Correctly predicted positive instances.</li>
                            <li>False Positives (FP): Negative instances incorrectly predicted as positive.</li>
                            <li>True Negatives (TN): Correctly predicted negative instances.</li>
                            <li>False Negatives (FN): Positive instances incorrectly predicted as negative.</li><br>
                            The confusion matrix can thus be used to visualize a model's classification errors and to derive various evaluation metrics such as precision, recall, F1-score, or accuracy.
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.latex(r'''
                            {Confusion matrix} =
                                \begin{bmatrix}
                                \text{{True Negatives}} & \text{{False Positives}} \\
                                \text{{False Negatives}} & \text{{True Positives}}
                                \end{bmatrix}
                ''')
                
                st.latex(r'''
                CM = \begin{pmatrix}
                TN & FP \\
                FN & TP
                \end{pmatrix}
                ''')
                st.write("CM : Confusion Matrix")
                
                # Affichage des métriques calculées en utilisant st.latex pour les formules
                st.latex(r"""
                \begin{align*}
                \text{Accuracy} &= \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} \\
                \text{Error Rate} &= 1 - \text{Accuracy} \\
                \text{Precision} &= \frac{\text{TP}}{\text{TP} + \text{FP}} \\
                \text{Recall} &= \frac{\text{TP}}{\text{TP} + \text{FN}} \\
                \text{F1 Score} &= 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
                \end{align*}
                """)
                
                # Explication du Total Number of Examples
                st.write("Where :")
                st.markdown("""
                - True Positives (TP)
                - True Negatives (TN)
                - False Positives (FP)
                - False Negatives (FN)
                """)
                
                ## 2. Error Rate : Definition
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Error Rate : 🎯</div>
                    <div class="content">
                        Error : Error Rate indicates the proportion of examples misclassified by the model. 
                        It is calculated as the complement of accuracy (1 - accuracy).
                        A lower error rate means better model performance.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.latex(r'\text{{Error rate}} = 1 - \text{{Accuracy}}')
                
                
                ## 3. Precision : Definition
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Precision : 🎯</div>
                    <div class="content">
                        Measures the model's ability to correctly predict the positive class. 
                        It is calculated as the ratio of true positives (TP) to the sum of true positives (TP) and false positives (FP). 
                        High accuracy indicates that the model is good at identifying positive cases without making false positives.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Latex formula
                st.latex(r'\text{{Precision}} = \frac{\text{{True Positives}}} {\text{{True Positives}} + \text{{False Positives}}}')

                
                
                ## 4. Recall (Sensitivity) : Definition
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Recall (Sensitivity) : 🎯</div>
                    <div class="content">
                        Assesses the model's ability to detect all positive examples. 
                        It is calculated as the ratio of true positives (TP) to the sum of true positives (TP) and false negatives (FN).
                        A high recall indicates that the model is effective in finding all positive cases, even though it may have false negatives.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Latex formula
                st.latex(r'\text{{Recall (Sensitivity)}} = \frac{\text{{True Positives}}} {\text{{True Positives}} + \text{{False Negatives}}}')

                    
                ## 5. F1 Score : Definition
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">F1 Score : 🎯</div>
                    <div class="content">
                        Combines precision and recall in a single measure, being the harmonic mean of the two. 
                        It gives an indication of the balance between these two aspects of model performance. 
                        A high F1 score indicates good overall model performance in classification.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Latex formula
                st.latex(r'\text{{F1 Score}} = 2 \times \frac{\text{{Precision}} \times \text{{Recall}}} {\text{{Precision}} + \text{{Recall}}}')

                
                    
                ## 6. F1 Score : Definition
                with col2 : 
                    st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class="container">
                        <div class="header">Accuracy : 🎯</div>
                        <div class="content">
                            Estimates the proportion of examples correctly classified by the model. 
                            It is calculated as the ratio of true positives (TP) and true negatives (TN) to the total number of examples. 
                            A high accuracy indicates that the model performs well overall.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Latex formula : Formule de calcul de l'Accuracy
                    st.latex(r'\text{{Accuracy}} = \frac{\text{{True Positives}} + \text{{True Negatives}}} {\text{{Total Number of Examples}}}')
                    st.write("Where **Total Number of Examples** represents the sum of all examples in the dataset.")
                        
                    #### Choose de the best model ####
                        


    
    
    #------------------------------------------#
    # MODELS : (Tuning de model)
    #-----------------------------------------#
    # sidebar.subheader('5. Model comparison')
    sidebar.subheader("5. Model's Tuning")
    # Select the best model
    # Explain models features importances
    # TUNINGS : (Improve model performance by using only those variables that best explain the models)
    # Improve model performance by using only those variables that best explain the models
    
    
    
    #-------------------------#
    # ## Features importances  ##
    # with col2 : 
    #     st.subheader('Features importances : ')
    #     st.checkbox('Features importances : ', value=True)
    #     if name in ['Decision Tree', 'Random Forest']:
    #         st.write('Feature importances:')
    #         feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    #         st.write(feature_importances.sort_values(ascending=False))
    #--------------------------------------------------------------------------------------#



    #------------------------------------------#
    # SCORING : (Predictions)
    #-----------------------------------------#
    sidebar.subheader('6. Scoring')        
        
        
        

    #-----------------------------------------#
    # FOOTER : Bas de page (html)
    #-----------------------------------------#
    display_html("footer.html")
    
    

if __name__ == "__main__":
    # st.set_page_config(page_title="Mon Application", layout="wide")
    main()
    
