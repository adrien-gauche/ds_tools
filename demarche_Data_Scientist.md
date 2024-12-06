# Démarche de travail Data Scientist

## 1. Définir un objectif mesurable

- **Objectif** :
- **Métrique** : https://scikit-learn.org/stable/modules/model_evaluation.html

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
- **Valeurs aberrantes** :

### 2.2 Analyse du fond

- **Visualisation de la target** : 
- **Compréhension des variables** : 
- **Visualisation des relations features-target** : 
- **Identification des outliers** : 

## 3. Pré-processing
- **Séparation Train/Test** : KFold
- **Gestion des NaN** : Éliminer ou imputer les valeurs manquantes.
- **Encodage** : Convertir les variables catégoriques
- **Suppression des outliers néfastes** :
- **Sélection de features** : 
- **Engineering de features** : Créer de nouvelles features à partir des données existantes.
- **Normalisation/Standardisation** :


## 4. Modélisation

- **Définir une fonction d'évaluation** : Choisir une fonction d'évaluation pertinente pour le modèle.
- **Entraînement des modèles** : 
- **Optimisation des hyperparamètres** : GridSearchCV, Optuna
- **Analyse des erreurs** : Identifier les sources d'erreurs et revenir au pré-processing si nécessaire.
- **Learning Curve** : courbes d'apprentissage pour diagnostiquer performance

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