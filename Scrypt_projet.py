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

### Chargement de la base de données ######
chdir(r"C:\Users\carrel\Downloads\Projet")   # work directory
pd.set_option('display.max_column', 12)
dfp = pd.read_sas(r"C:\Users\carrel\Downloads\Projet/Tab2.sas7bdat")

####
# 1)
# Renommage des variables
dfp.rename(columns={'varA': 'Incident_r', 'varB': 'Montant_pret', 'varC': 'Montant_hypotheque',
'varD': 'Val_propriete','varE': 'Motif_pret','varF': 'Profession',
'varG': 'Nb_annees_travail','varH': 'Nb_report_pret', 'varI': 'Nb_litiges',
'varJ': 'Age_cred','varK': 'Nb_demandes_cred','varL': 'Ratio_dette_revenu'}, inplace=True)
print(dfp)

dfp.shape       #(Nb_ligne, Nb_col)
dfp.dtypes      #(Types des variables)

#compter le nombre d'occurence de la variable Incident_r à prédire
dfp["Incident_r"].value_counts(dropna = False)

#compter le nombre d'occurence des variables qualitatives
dfp["Motif_pret"].value_counts(dropna = False)
dfp["Profession"].value_counts(dropna = False)

## Conversion de l'age_cred en annees
dfp['Age_cred'] = round(dfp['Age_cred'] /12 ,2)
dfp.describe(include="all")

# variables qualitatives (Calcul des fréquences des modalités)
for col in dfp.select_dtypes('object'):
    print(f'{col :-<30} {dfp[col].unique()}')

for col in dfp.select_dtypes('object'):
    plt.figure()
    dfp[col].value_counts().plot.pie()

######################################################
### Mise en forme des variables des variables avant le 
### Lancement de la matrice de correlation

sns.countplot(x='Montant_pret', hue='Incident_r', data=dfp)
sns.countplot(x='Ratio_dette_revenu', hue='Incident_r', data=dfp)
sns.countplot(x='Val_propriete', hue='Incident_r', data=dfp)
sns.countplot(x='Nb_report_pret', hue='Incident_r', data=dfp)
sns.countplot(x='Nb_litiges', hue='Incident_r', data=dfp)
sns.countplot(x='Age_cred', hue='Incident_r', data=dfp)
sns.countplot(x='Nb_demandes_cred', hue='Incident_r', data=dfp)
sns.countplot(x='Ratio_dette_revenu', hue='Incident_r', data=dfp)
sns.countplot(x='Montant_hypotheque', hue='Incident_r', data=dfp)

##### Recherche des valeurs manquantes #####
dfp.isna().any() # NA oui ou non
dfp.isna().sum() # comptage des NA

# Calcul en pourcentage des valeurs manquantes de chaque variable
dfp_Na=pd.DataFrame({"Pourcentage_Na" : round(dfp.isnull().sum()/(dfp.shape[0])*100,2)})
dfp_Na


####  Histogramme des valeurs manquantes ####
List_var=('Incident_r','Montant_pret','Montant_hypotheque','Val_propriete','Motif_pret',
            'Profession','Nb_annees_travail','Nb_report_pret','Nb_litiges','Age_cred',
            'Nb_demandes_cred','Ratio_dette_revenu')

List_value=[0,0,8.65,1.90,4.56,4.70,8.56,12.06,9.90,5.12,8.65,21.74]
y_pos=np.arange(len(List_var))

plt.bar(y_pos,List_value)
plt.xticks(y_pos,List_var,rotation=90)
plt.ylabel('val. manquantes (%)')
plt.subplots_adjust(bottom=0.4,top=0.99)
plt.show()

###################################
# DETECTION DES VALEURS ABERRANTES
# Histogramme des variables continues 
for col in dfp.select_dtypes('float'):
   plt.figure()
   sns.distplot(dfp[col])
# variables qualitatives
for col in dfp.select_dtypes('object'):
    print(f'{col :-<30} {dfp[col].unique()}') #systeme de marge


# Diagrammes en moustache des 9 variables quantitatives
dfp.boxplot(column='Montant_pret')
dfp.boxplot(column='Ratio_dette_revenu') # <17 et >75
dfp.boxplot(column='Age_cred') # >50
dfp.boxplot(column='Nb_litiges')
dfp.boxplot(column='Nb_demandes_cred')
dfp.boxplot(column='Nb_report_pret')
dfp.boxplot(column='Nb_annees_travail')
dfp.boxplot(column='Val_propriete') # >400000
dfp.boxplot(column='Montant_pret') # >55000
dfp.boxplot(column='Montant_hypotheque') # 270000


###############################################
#TRAITEMENT : VALEURS MANQUANTES ET ABBERANTES

##Statistiques desc avant imputation des var qualitatives
cat_var_avant=dfp[['Motif_pret', 'Profession']]
print(cat_var_avant.describe())   #variables catégorielle avant imputation
## On voit clairement que nous avons des manquantes manquantes 
## au niveau de ces variables, car nous avons un total de 3750 individus

##Statistiques desc avant imputation des var quantitives
print(dfp.describe())



######### Imputations #######
### VARIABLES CATEGORIELLES : imputation par le mode
cat_var = dfp[['Motif_pret', 'Profession']]
cat_var['Motif_pret'] = cat_var['Motif_pret'].fillna(cat_var['Motif_pret'].mode()[0])
cat_var['Profession'] = cat_var['Profession'].fillna(cat_var['Profession'].mode()[0])
print(cat_var.describe()) #variables catégorielle après imputation

## Autrement en application
dfp['Motif_pret']= cat_var['Motif_pret'].fillna(cat_var['Motif_pret'].mode()[0])
dfp['Profession']=cat_var['Profession'].fillna(cat_var['Profession'].mode()[0])
print(dfp['Profession'])


# reVérifier s'il y a des 
cat_var.isna().any()
## Nous voyons clairement que toutes les variables ont été imputé avec succès


### VARIABLES CONTINUES : imputation par le kNN
quant_var = dfp[['Incident_r', 'Montant_pret', 'Montant_hypotheque', 'Val_propriete',
                 'Nb_annees_travail', 'Nb_report_pret', 'Nb_litiges', 'Age_cred', 
                 'Nb_demandes_cred', 'Ratio_dette_revenu']]


# Transformation des valeurs abberantes en manquantes 
u=quant_var.Ratio_dette_revenu
for i in range(len(u)): 
 if u[i] > 75 or u[i] < 17 : u[i]= 'NaN'
print(u)

#variable Val_propriete
u=quant_var.Val_propriete
for i in range(len(u)): 
 if u[i] > 400000 : u[i]= 'NaN'
 
#variable Age_cred
u=quant_var.Age_cred
for i in range(len(u)): 
 if u[i] > 50 : u[i]= 'NaN'
 
#variable Montant_pret
u=quant_var.Montant_pret
for i in range(len(u)): 
 if u[i] > 55000 : u[i]= 'NaN'
 
 #variable Montant_hypotheque
u=quant_var.Montant_hypotheque
for i in range(len(u)): 
 if u[i] > 270000 : u[i]= 'NaN'
 
 
# IDéfinition d'une fonction pour choisir le k-optimal 
rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))

def optimize_k (data, target):
    errors = [] # liste vide qui va contenir les erreurs calculées
    for k in range(1, 20, 1):
        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(data)
        quant_var_imputed = pd.DataFrame(imputed, columns=quant_var.columns)
    
        X = quant_var_imputed.drop(target, axis=1)
        y = quant_var_imputed[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        error = rmse(y_test, preds)
        errors.append({'K': k, 'RMSE': error})
    return errors

# le k-optimal est égal est à 16
k_errors = optimize_k(data=quant_var, target='Incident_r')
k_errors

# Imputation proprement dite 
imputer = KNNImputer(n_neighbors=16)
quant_var = pd.DataFrame(imputer.fit_transform(quant_var),columns = quant_var.columns)
# reVérifier s'il y a des 
quant_var.isna().any()


# Base de données totalement imputée
dfp = pd.concat([quant_var, cat_var], axis=1)
dfp.isna().any()

 
# Transformation des variables catégorielles en dummies 
cat_variables = dfp[['Motif_pret', 'Profession']]
cat_dummies = pd.get_dummies(cat_variables, drop_first=True)


# Ajout des dummies à la base initiale 
dfp2 = dfp
dfp2 = dfp.drop(['Motif_pret', 'Profession'], axis=1)
dfp2 = pd.concat([dfp2, cat_dummies], axis=1)


# ############# Matrice de correlation ###########  
# Matrice de corrélation 
corr = dfp2.corr()
corr.style.background_gradient(cmap='coolwarm')
corr.style.background_gradient(cmap='coolwarm').set_precision(2) 

# ### Commentaire :
# # La matrice de coorelation montre une forte correlation entre 
# # la valeur_propriété et  montant_hypothèque s 87,92%. 
# # Donc nous allons supprimer le montant de l'hypothèque
#########################################################


# Supression de la colonne du Montant_hypotheque
dfp2.drop(['Montant_hypotheque'], axis='columns', inplace=True)
dfp2.head()


################################################################
# 		     MODELISATION  
################################################################

# SEPARATION DE la variable DEPENDANTE (Y) DES EXPLICATIVES (X)
dfp2=dfp

# Matrice des explicatives (X)
X = dfp2.iloc[:,1:14]
print(X)

# Matrice de la dépendante (Incident_r)
Y = dfp2.iloc[:, 0]
print(Y)


# CONSTRUCTION DES DEUX ECHANTILLONS (Apprentissage et test)
from sklearn import model_selection 
X_app, X_test, Y_app, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state=10)
print(X_app.shape, X_test.shape, Y_app.shape, Y_test.shape)
## Echantillon Aprentissage: 80 %
## Echantillon Test: 20 %


################################################
# 	1 : MODELE DE REGRESSION LOGISTIQUE 
################################################
# On importe LogisticRegression depuis sklearn
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
modele = logit.fit(X_app,Y_app) # construction du modele sur l'´echantillon d'apprentissage
print(modele.coef_,modele.intercept_) # param`etres du mod`ele

# prediction sur l'échantilllon test 
Y_pred = modele.predict(X_test) # prediction sur l'´echantillon test

# taux de bonne prédiction
from sklearn import metrics
succes = metrics.accuracy_score(Y_test,Y_pred)
print(succes) 

# taux d'erreur
err = 1.0 - succes
print(err)
## taux_erre=0.206


##########################################
#          2 : MODELE DE k-NN
##########################################
# Importation du package depuis sklearn
import sklearn
from sklearn import neighbors, metrics

# Fixer les valeurs des hyperparamètres à tester
param_grid = {'n_neighbors':list(range(1,16))}

# Choisir un score à optimiser, ici l'accuracy (proportion de prédictions correctes)
score = 'accuracy'

# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
knn = model_selection.GridSearchCV(
    neighbors.KNeighborsClassifier(), # un classifieur kNN
    param_grid,     # hyperparamètres à tester
    cv=5,           # nombre de folds de validation croisée
    scoring=score   # score à optimiser
    )

# Optimiser le classifieur sur le jeu d'entraînement 
digit_knn=knn.fit(X_app, Y_app)

# # Afficher le(s) hyperparamètre(s) optimaux
digit_knn.best_params_["n_neighbors"]
##le paramètre vaut 11

#Ensuite on procède à l'estimation du modèle avec la valeur "optimale" de notre paramètre qui vaut 6
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=digit_knn.best_params_["n_neighbors"])
digit_knn=knn.fit(X_app, Y_app)

# Estimation de l’erreur de prévision
1-digit_knn.score(X_test,Y_test)

# Prévision
Y_chap = digit_knn.predict(X_test)

# Matrice de confusion
mat_conf=pd.crosstab(Y_test, Y_chap)
print(mat_conf)
plt.matshow(mat_conf)
## taux_erre=0.206


#####################################################
#          3 : MODELE D'ARBRE DE DECISION 
#####################################################
# Importation du package depuis sklearn
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(min_samples_split=5) 

# définition du modèle de l'arbre de décision
tit_tree = dtree.fit(X_app, Y_app)

# Estimation de l'erreur de prévision
1-tit_tree.score(X_test, Y_test) 
## taux_erre=0.166


#Visualisation de l’arbre.
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'

#installer (vian Anaconda Prompt) la librairie pydotplus : pip install pydotplus
import pydotplus
from sklearn.tree import export_graphviz

# Représentation de l'arbre
dot_data = export_graphviz(tit_tree, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("Incident.pdf")
graph.write_png("Incident.png")
#L’arbre est généré dans un fichier image ainsi qu'un pdf à visualiser pour se rende compte 
 

######################################################
# 	       4 : RANDOM FOREST
######################################################
# Importation du package depuis sklearn
from sklearn.ensemble import RandomForestClassifier

#définition des paramètres 
forest = RandomForestClassifier(n_estimators=500, min_samples_split=5, oob_score=True)

# apprentissage
forest = forest.fit(X_app, Y_app)
print(1-forest.oob_score_) # =0.1252

# erreur de prévision sur le test
1-forest.score(X_test,Y_test) 
## taux_erre=0.116

# prévision
Y_chap = forest.predict(X_test)

# Matrice de confusion
mat_conf=pd.crosstab(Y_test, Y_chap)
print(mat_conf)



###############################################
# 	             SCORING 
###############################################
# Création des scores 

modele = logit.fit(X_app,Y_app) 
probas = logit.predict_proba(X_test) #calcul des probabilités d'affectation sur l'´echantillon test

#score de "Incident"
score = probas[:,1]
print(score)

score2 = sorted(score, reverse=True)
Grille100 = round(100*(score[0]+max(score[0]))/(min(score[0])+max(score[0])),2)

max(score)

####################################
# prediction sur l'échantilllon test 
Y_pred = modele.predict(X_test) # prediction sur l'´echantillon test

def my_custom_loss_func(y_true, y_pred):
    diff = np.abs(y_true - y_pred).max()
    return np.log1p(diff)

from sklearn.metrics import fbeta_score, make_scorer
my_custom_loss_func(Y_test, Y_pred)
score3 = make_scorer(my_custom_loss_func, greater_is_better=False)
score3(modele, X_app, Y_app)


from custom_scorer_module import custom_scoring_function 
cross_val_score(model,
                X_app,
                y_train,
                scoring=make_scorer(custom_scoring_function, greater_is_better=False),
                cv=5,
                n_jobs=-1)



from sklearn import svm
import sklearn.metrics as metrics
mvs=svm.SVC()
modele2=mvs.fit(X_app,Y_app)
z_pred2=modele2.predict(X_test)
print(metrics.confusion_matrix(Y_test,Y_pred))
print(metrics.accuracy_score(Y_test, Y_pred))


from sklearn import model_selection
parametres=[{'C':[0.1,1,10],'kernel':['rbf','linear']}]
grid=model_selection.GridSearchCV(estimator=mvs,param_grid=parametres,scoring='accuracy')
grille=grid.fit(X_app,Y_app)
print(pandas.Dataframe.from_dict(grille.cv_results_).loc[:,["params","mean_test_score"]])
print(grille.best_params_)
print(grille.best_score_)
y_pred3=grille.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred3))




