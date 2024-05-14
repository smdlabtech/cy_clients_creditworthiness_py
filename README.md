# Prédiction de la solvabilité des clients d'une banque
Les points clés de cette étude se façon suivantes :  
- Classification des clients en fonction de leur solvabilité
- Distinction des clients en fonction des risques de crédit
- Construction d'un nouveau score de risque pour améliorer sa stratégie d'attribution de crédit
- Amélioration de la stratégie de fidélisation des clients
- Création d'un modèle de prédiction de remboursement de prêts des clients (fidèles et nouveaux arrivants)

# Objectifs pratiques
L’objectif principal ici est de prédire la solvabilité des clients d’une banque. Il s’agira
donc de distinguer la population de recherche (clients) en fonction du risque de crédit, c’est-à-
dire de classer les clients solvables et les clients peu fiables.  
Il faudra développer un score de risque à attribuer aux nouveaux clients et aux demandeurs de prêt, ce qui permettra
à la banque de leur octroyer des prêts sur la base de leurs ***scores*** et des ***ressources*** disponibles de la banque.


## Création de l'application Streamlit ML

Pour lancer une application streamlit en local, il faudra appliquer les étapes suivantes :


- Ouvrir un terminal dans l'endroit contenant le script *"Script_projet.py"*  
- Puis saisir depuis le terminal la commande suivante :

```python
streamlit run Scrypt_projet.py
```


Voici le schéma du repo :  

```
/tests_data
|-- /.venv
	|-- /Include
	|   |-- Script_projet.py
	|-- /Scripts
	|-- /Lib
|-- /data
|-- /docs
|-- /archives
|-- LICENCE
|-- README.md
```

## Installations et dépendances
### Démarches à suivre
Pour constituer notre repo :   

**1**- Nous avons d'abord definie notre architechture de repo dont nous avons besoin comme le montre le schéma ci-dessus.

**2**- Ensuite, nous avons pointé ver le dossier principal du repo   
```cd /chemin/vers/votre/projet```

**3**- Puis créer notre environnement virtuel qui va accueillir notre programme python et ses dépendances.  
```python -m venv /chemin/vers/nouvel/environnement```

**4**- Par exemple, si vous voulez créer l'environnement virtuel dans le répertoire de votre projet, la commande serait :  
```python -m venv.venv```

**NB** : Normalement lors de la création de l'environnement virtuel, il y a eu un fichier .gitignore (txt) qui a été créé. Il faudra rajouter la commande suivante dans de fichier :  

Exemple de contenu pour.gitignore  
```venv/```  
Donc le fait d'ajouter dans le gitignore ce bout de code précédent permet de pouvoir intégré le .venv dans le repo.

**5**- Activez l'environnement virtuel : Activez l'environnement virtuel en utilisant la commande appropriée selon votre système d'exploitation.
```.\.venv\Scripts\activate``` et pour désactiver l'environnement :    
```.\.venv\Scripts\Deactivate```    

Et enfin, quand la MAJ du code python est terminée, vous pourrez créez le fichier de ```requirements.txt``` avec les commandes suivantes :  

```pip freeze > requirements.txt```    
```git add requirements.txt```    
```git commit -m "Add requirements.txt"```    
```git push```      


Rques : Si vous ne souhaitez pas inclure tous les paquets installés dans votre projet, mais seulement ceux qui sont effectivement utilisés, vous pouvez utiliser pipreqs. pipreqs analyse votre code pour identifier les paquets importés et génère un fichier requirements.txt plus concis. Pour l'utiliser, installez pipreqs et exécutez.

```pip install pipreqs```  
```pipreqs /chemin/vers/votre/projet```  

Ensuite :  
```pip install -r requirements.txt```  

