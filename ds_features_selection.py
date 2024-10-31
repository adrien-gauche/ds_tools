#https://scikit-learn.org/stable/modules/feature_selection.html
from sklearn.feature_selection import VarianceThreshold #transformers with variance
from sklearn.feature_selection import SelectKBest #transformers with statistical test
from sklearn.feature_selection import SelectFromModel #transformers with model

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif # dependency test for categorical
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression # dependency test for numerical

def remove_low_variance_features(X, threshold=0.0):
    """
    Remove features with low variance.

    Parameters:
    X (DataFrame): The feature matrix.
    threshold (float): The threshold below which a feature's variance is considered low.

    Returns:
    DataFrame: A DataFrame containing the selected features.
    """
    
        # Check if k is valid
    if threshold < 0:
        raise ValueError("threshold must be greater than 0")
        
    # Initialize the selector
    selector = VarianceThreshold(threshold=threshold)

    # Fit the selector to the data
    X_new = selector.fit_transform(X)

    # Get the boolean index of selected columns
    selected = selector.get_support()

    # Get the names of selected columns
    selected_columns = X.columns[selected]

    # Print the names of selected columns
    print("Selected columns:", selected_columns)

    # Return the selected columns as a DataFrame
    return pd.DataFrame(X_new, columns=selected_columns)

def select_kbest_features(X, y, test, k=1):
    """
    Select k features with a statistical test.

    Parameters:
    X (DataFrame): The feature matrix.
    y (Series): The target variable.
    test (callable): The statistical test to use for feature selection.
    k (int): The number of features to select.

    Returns:
    DataFrame: A DataFrame containing the selected features.
    """
    # Check if k is valid
    if k <= 0 or k > X.shape[1]:
        raise ValueError("k must be a positive integer less than or equal to the number of features.")

    # Initialize the selector
    selector = SelectKBest(test, k=k)

    # Fit the selector to the data
    X_new = selector.fit_transform(X, y)

    # Get the boolean index of selected columns
    selected = selector.get_support()

    # Create a DataFrame to display the scores and p-values
    scores_pvalues = pd.DataFrame({'scores': selector.scores_, 'pvalues': selector.pvalues_}, index=X.columns)

    # Display the scores and p-values
    print(scores_pvalues)

    # Get the names of selected columns
    selected_columns = X.columns[selected]

    # Print the names of selected columns
    print("Selected columns:", selected_columns)

    # Return the selected columns as a DataFrame
    return pd.DataFrame(X_new, columns=selected_columns)

def select_with_model(X, y, model, threshold=0.05):
    
    from sklearn.feature_selection import SelectFromModel

    # Initialize SelectFromModel with the instance of SGDClassifier
    selector = SelectFromModel(model, threshold='mean')

    # Fit the selector on the data
    X_transformed = selector.fit_transform(X, y.values.ravel())

    # Get the support mask
    support = selector.get_support()
    print(support)
    print(selector.estimator_)
    
    from sklearn.feature_selection import RFECV

    #Feature ranking with recursive feature elimination and cross-validation
    selector = RFECV(model, 
                    step=1,
                    min_features_to_select=2,
                    cv=5,
                    )
    selector.fit(X, y.values.ravel())
    print(selector.ranking_)


### STATISTICAL TESTS ###
#https://scikit-learn.org/stable/api/sklearn.feature_selection.html

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif # dependency test for categorical
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression # dependency test for numerical

def test_y_quali_X_quanti(df: pd.DataFrame, test_stat, target_col: str, alpha=0.05):
    """Calculate the T-test for the means of two independent samples of scores.
    #https://www.bibmath.net/dico/index.php?action=affiche&quoi=./s/studenttest.html
    #https://docs.scipy.org/doc/scipy/reference/stats.html

    This is a test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default.

    Args:
        df (pd.DataFrame): _description_
        target_col (str): _description_
        alpha (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    # Séparation des données en fonction de la colonne cible
    positive_df = df[df[target_col] == True]
    negative_df = df[df[target_col] == False]

    # Échantillonnage équilibré du groupe négatif pour avoir la même taille que le groupe positif
    balanced_neg = negative_df.sample(positive_df.shape[0])

    # DataFrame pour stocker les résultats
    df_result = pd.DataFrame()
    n_nan = df.isna().sum() / df.shape[0] * 100
    # df_result['nan'] = n_nan

    # Parcourir chaque colonne du DataFrame (sauf la colonne cible)
    for col in df.select_dtypes(include=["number"]).columns:
        if col != target_col:  # On ignore la colonne cible
            # Effectuer le test t pour chaque colonne
            # stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
            (
                statistic,
                pvalue,
            ) = test_stat(balanced_neg[col], positive_df[col], nan_policy="omit")

            # Vérifier si l'hypothèse nulle est rejetée
            result_str = (
                f"H0 Rejetée avec un risque d'erreur <{alpha*100}%"
                if pvalue < alpha
                else "H0 Accepté, moyennes égales"
            )

            # Création d'un DataFrame temporaire pour la colonne actuelle
            df_current = pd.DataFrame(
                {
                    "missing": [n_nan[col]],
                    "stat": [statistic],
                    "pvalue": [pvalue],
                    "result": [result_str],
                },
                index=[col],
            )

            # Concatenation des résultats pour chaque colonne
            df_result = pd.concat([df_result, df_current])

    # Format the 'missing %' column as a percentage
    df_result["missing"] = df_result["missing"].map(lambda x: f"{x:.2f}%")

    # Retourner les résultats triés par la p-valeur
    return df_result.sort_values(by="pvalue")


## Exemple d'appel de la fonction, en filtrant les colonnes numériques
#df_y_quali_X_quanti = test_y_quali_X_quanti(
#    df,
#    stats.ttest_ind,
#    target_col="EA FPI\n>ou= à 1 EA 0/1",
#)

def test_y_quali_X_quali(df: pd.DataFrame, test_stat, target_col: str, alpha=0.05):
    """Calculate the T-test for the means of two independent samples of scores.
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

    This is a test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default.

    Args:
        df (pd.DataFrame): _description_
        target_col (str): _description_
        alpha (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
    """

    n_nan = df.isna().sum() / df.shape[0] * 100

    # DataFrame pour stocker les résultats
    df_result = pd.DataFrame()

    # Parcourir chaque colonne du DataFrame (sauf la colonne cible)
    for col in df.columns:
        if col != target_col:  # On ignore la colonne cible
            obs = pd.crosstab(df[target_col], df[col])
            # Effectuer le test pour chaque colonne
            statistic, pvalue, dof, expected_freq = stats.chi2_contingency(
                obs, correction=False
            )

            # Vérifier si l'hypothèse nulle est rejetée
            result_str = (
                f"H0 Rejetée avec un risque d'erreur <{alpha*100}%"
                if pvalue < alpha
                else "H0 Accepté, variables indépendantes"
            )

            # Création d'un DataFrame temporaire pour la colonne actuelle
            df_current = pd.DataFrame(
                {
                    "missing": [n_nan[col]],
                    "stat": [statistic],
                    "pvalue": [pvalue],
                    "result": [result_str],
                },
                index=[col],
            )

            # Concatenation des résultats pour chaque colonne
            df_result = pd.concat([df_result, df_current])

    # Format the 'missing %' column as a percentage
    df_result["missing"] = df_result["missing"].map(lambda x: f"{x:.2f}%")

    # Retourner les résultats triés par la p-valeur
    return df_result.sort_values(by="pvalue")


## Exemple d'appel de la fonction, en filtrant les colonnes numériques
#df_y_quali_X_quali = test_y_quali_X_quali(
#    df.select_dtypes(include=["boolean"]),
#    stats.chi2,
#    target_col="EA FPI\n>ou= à 1 EA 0/1",
#)