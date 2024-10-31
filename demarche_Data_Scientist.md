# Démarche de travail Data Scientist

## 1. Définir un objectif mesurable

- **Objectif** :
- **Métrique** : https://scikit-learn.org/stable/modules/model_evaluation.htm

### Classification
**Précision (Precision)** : $$ \text{Précision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$
  - Objectif : Réduire le taux de faux positifs (FP).

**Rappel (Recall)** : $$ \text{Rappel} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$
  - Objectif : Réduire le taux de faux négatifs (FN).

- **Déséquilibre des classes** : générer samples (ex. : `imbalanced-learn`) ou score F1, recall, precision

## 2. EDA (Exploratory Data Analysis)

### 2.1 Analyse de la forme

- **Identification de la target** : 
- **Dimensions du dataset** : df.shape
- **Types de variables** :
- **Valeurs manquantes** :

### 2.2 Analyse du fond

- **Visualisation de la target** : 
- **Compréhension des variables** : 
- **Visualisation des relations features-target** : 
- **Identification des outliers** : 

## 3. Pré-processing
- **Séparation Train/Test** : Diviser les données en ensembles d'entraînement et de test.
- **Gestion des NaN** : Éliminer ou imputer les valeurs manquantes.
- **Encodage** : Convertir les variables catégoriques en numériques (ex. : One-Hot Encoding).
- **Suppression des outliers néfastes** : Éliminer les outliers qui dégradent la performance du modèle.
- **Sélection de features** : Utiliser des techniques de sélection de features (ex. : `pycaret`).
- **Engineering de features** : Créer de nouvelles features à partir des données existantes.
- **Normalisation/Standardisation** : Appliquer un scaling approprié (standardisation, normalisation).


## 4. Modélisation

- **Définir une fonction d'évaluation** : Choisir une fonction d'évaluation pertinente pour le modèle.
- **Entraînement des modèles** : Entraîner plusieurs modèles et comparer leurs performances (ex. : `pycaret`).
- **Optimisation des hyperparamètres** : Utiliser GridSearchCV, Optuna ou d'autres techniques pour optimiser les hyperparamètres.
- **Analyse des erreurs** : Identifier les sources d'erreurs et revenir au pré-processing si nécessaire.
- **Learning Curve** : Utiliser les courbes d'apprentissage pour diagnostiquer la performance des modèles.

**Workflow**
Utiliser la learning curve

* underfitting:
  - polynomial features
  - features engineering
  - modèles plus complexes

* overfitting:
  - plus de données
  - imputation/ fillna
  - selectFromModel
  - modèle régularisé (random forest)
  - data leakage

## 5. Debug
- Explicabilité : SHAP
- Visualisation décisions: supertree, Graphviz