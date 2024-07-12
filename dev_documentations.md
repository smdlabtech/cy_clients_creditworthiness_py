# File .DS_Store ()

# File dev_generate_docs.py (.py)
 - **extract_info_from_file** : Extrait les informations pertinentes (docstrings, fonctions, classes) d'un fichier Python,
et le contenu pour les fichiers CSS et JavaScript, tout en ignorant les types de fichiers spécifiés.

Args:
    file_path (str): Le chemin complet vers le fichier.
    
Returns:
    dict: Un dictionnaire contenant les informations extraites.
 - **generate_documentation_md** : Génère un fichier Markdown contenant la documentation de tous les fichiers dans un répertoire,
en ignorant les répertoires spécifiés et les types de fichiers spécifiques.

Args:
    directory_path (str): Le chemin du répertoire contenant les fichiers.
    output_file (str): Le chemin du fichier de sortie Markdown.

# File LICENSE ()

# File Scrypt_projet with docs tring.py (.py)
 - **load_data** : Load the dataset and set the working directory.

Returns:
    pd.DataFrame: The loaded dataset.
 - **rename_variables** : Rename the columns of the dataframe.

Args:
    dfp (pd.DataFrame): The original dataframe.

Returns:
    pd.DataFrame: The dataframe with renamed columns.
 - **data_overview** : Display basic information and descriptive statistics of the dataframe.

Args:
    dfp (pd.DataFrame): The dataframe to be described.
 - **plot_qualitative_distributions** : Plot pie charts for qualitative variables.

Args:
    dfp (pd.DataFrame): The dataframe with qualitative variables.
 - **plot_countplots** : Plot countplots for specified variables.

Args:
    dfp (pd.DataFrame): The dataframe to plot.
 - **missing_values_analysis** : Analyze missing values in the dataframe.

Args:
    dfp (pd.DataFrame): The dataframe to analyze.

Returns:
    pd.DataFrame: A dataframe with the percentage of missing values.
 - **plot_missing_values_histogram** : Plot histogram of missing values for specified variables.
 - **plot_outlier_distributions** : Plot histograms and boxplots for continuous variables.

Args:
    dfp (pd.DataFrame): The dataframe with continuous variables.
 - **impute_missing_values** : Impute missing values for categorical and continuous variables.

Args:
    dfp (pd.DataFrame): The dataframe with missing values.

Returns:
    pd.DataFrame: The dataframe with imputed values.
 - **transform_categorical_to_dummies** : Transform categorical variables into dummy variables.

Args:
    dfp (pd.DataFrame): The dataframe with categorical variables.

Returns:
    pd.DataFrame: The dataframe with dummy variables.
 - **correlation_matrix** : Compute and display the correlation matrix.

Args:
    dfp2 (pd.DataFrame): The dataframe for which to compute the correlation matrix.
 - **drop_column** : Drop a specified column from the dataframe.

Args:
    dfp2 (pd.DataFrame): The dataframe from which to drop the column.
    column (str): The column to drop.

Returns:
    pd.DataFrame: The dataframe with the column dropped.
 - **prepare_modeling_data** : Prepare the data for modeling by separating the dependent and explanatory variables.

Args:
    dfp2 (pd.DataFrame): The dataframe to prepare.

Returns:
    tuple: Tuple containing the matrices X (explanatory) and Y (dependent).
 - **split_data** : Split the data into training and testing sets.

Args:
    X (pd.DataFrame): The explanatory variables.
    Y (pd.Series): The dependent variable.

Returns:
    tuple: Tuple containing the training and testing sets (X_app, X_test, Y_app, Y_test).
 - **logistic_regression_model** : Train and evaluate a logistic regression model.

Args:
    X_app (pd.DataFrame): Training set of explanatory variables.
    Y_app (pd.Series): Training set of the dependent variable.
    X_test (pd.DataFrame): Testing set of explanatory variables.
    Y_test (pd.Series): Testing set of the dependent variable.

Returns:
    float: The accuracy of the model.
 - **knn_model** : Train and evaluate a k-NN model.

Args:
    X_app (pd.DataFrame): Training set of explanatory variables.
    Y_app (pd.Series): Training set of the dependent variable.
    X_test (pd.DataFrame): Testing set of explanatory variables.
    Y_test (pd.Series): Testing set of the dependent variable.

Returns:
    float: The accuracy of the model.
 - **decision_tree_model** : Train and evaluate a decision tree model.

Args:
    X_app (pd.DataFrame): Training set of explanatory variables.
    Y_app (pd.Series): Training set of the dependent variable.
    X_test (pd.DataFrame): Testing set of explanatory variables.
    Y_test (pd.Series): Testing set of the dependent variable.

Returns:
    float: The accuracy of the model.
 - **random_forest_model** : Train and evaluate a random forest model.

Args:
    X_app (pd.DataFrame): Training set of explanatory variables.
    Y_app (pd.Series): Training set of the dependent variable.
    X_test (pd.DataFrame): Testing set of explanatory variables.
    Y_test (pd.Series): Testing set of the dependent variable.

Returns:
    float: The accuracy of the model.
 - **main** : Main function to run the data analysis and modeling pipeline.

# File Scrypt_projet.ipynb (.ipynb)

# File Scrypt_projet.py (.py)
 - **optimize_k** : No docstring found.
 - **my_custom_loss_func** : No docstring found.

