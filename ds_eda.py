#https://scikit-learn.org/stable/modules/feature_selection.html

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np

### CORRELATION ###
#Correlation between two categorical variables: chi2_contingency
#Correlation between two continuous variables: Pearson correlation, default .corr()
#Correlation between a categorical and a continuous variable: ANOVA, t-test

### DATA EXPLORATION ###
#see: https://seaborn.pydata.org/tutorial
def analyze_dataframe(df, missing_percent_threshold=1):
    pd.set_option("display.max_row", df.shape[0])
    pd.set_option("display.max_columns", df.shape[1])

    # Display the first 5 rows of the DataFrame
    print("First 5 rows of the DataFrame:")
    display(df.head())

    # Display information about the DataFrame
    print("=" * 30)
    print("\nInformation about the DataFrame:")
    print(df.info())

    df.dtypes.value_counts().plot.pie()

    # Display statistical details about the DataFrame
    print("=" * 30)
    # print("\nStatistical details of the DataFrame:")
    # display(df.describe(include='all'))
    print("\nStatistical details for numbers:")
    display(df.describe(include=[np.number]))
    print("\nStatistical details for categorical data:")
    display(df.describe(exclude=[np.number]))

    # Display the number of missing values in each column
    print("=" * 30)
    print("\nPercent of missing values in each column:")
    missing_values = df.isna().sum()
    missing_percent = missing_values / df.shape[0]
    total = df.shape[0]
    missing_stats = pd.DataFrame(
        {
            "Missing Values": missing_values,
            "Total Values": total,
            "Missing Percent": missing_percent,
        },
        index=df.columns,
    )

    missing_stats = missing_stats.sort_values(by="Missing Percent")

    display(missing_stats[missing_stats["Missing Percent"] < missing_percent_threshold])

    plt.figure(figsize=(15, 8))
    # plt.figure()
    sns.heatmap(df.isna(), cbar=False)
    plt.title("Missing Values in DataFrame")

    for col in df.select_dtypes(["object", "category"]):
        print(f"{col :-<50} {df[col].unique()}")
    

### CATEGORICAL DATA ###
#categorical target variable function of categorical features

def print_unique_values(df):
    """Print the unique values of each column in the DataFrame.

    Args:
        df (_type_): _description_
    """
    for col in df.select_dtypes(exclude=["number", "datetime"]):
        print(f"{col :-<50} {df[col].unique()}")

def plot_crosstab(df, target):
    for col in df.select_dtypes(exclude=["number", "datetime"]).columns:
        plt.figure(figsize=(5, 1))  # Create a new figure for each column
        sns.heatmap(
            pd.crosstab(df[target], df[col], dropna=False),
            annot=True,
            fmt="d",
        )
        plt.xticks(rotation=45, ha="right")
        plt.show()
        print("nan sum", col, df[col].isna().sum() / df.shape[0], "% \n")
        # plt.savefig(f"fig\\crosstab_{col}.png")
    
def plot_qualitative_data(df):
    for col in df.select_dtypes(exclude=["number", "datetime"]):
        plt.figure(figsize=(5, 1))  # Create a new figure for each column

        # df[col].value_counts().plot.pie(autopct="%1.1f%%")
        sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts())

        plt.xticks(rotation=45, ha="right")
        plt.title(col)
        # plt.legend()
        plt.show()
        print("nan sum", col, df[col].isna().sum(), "\n")

def plot_categorical_feature(data, feature, target):
    """
    Generate countplot and heatmap to visualize the relationship between a categorical feature and a target variable.

    Parameters:
    - data: DataFrame containing the data.
    - feature: The categorical feature variable to visualize.
    - target: The target categorical variable to visualize.
    
    https://seaborn.pydata.org/tutorial/categorical.html
    """

    plt.figure(figsize=(10, 6))

    # Countplot: Shows the count distribution of the target variable within each category of the feature
    sns.countplot(x=feature, hue=target, data=data)
    plt.title(f'Countplot of {feature} by {target}')
    plt.show()

    # Crosstab and Heatmap: Creates a contingency table and visualizes it using a heatmap
    crosstab = pd.crosstab(data[feature], data[target])
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f'Heatmap of {feature} by {target}')
    plt.show()

    
def plot_categorical(df, x, y, hue, palette="muted"):
    # Create a categorical plot of y in function of x, with hue as the category
    g = sns.catplot(x=x, y=y, hue=hue, data=df, kind="bar", palette=palette)

    # Despine the left side of the plot
    g.despine(left=True)

    # Set the y-label of the plot
    g = g.set_ylabels("survival probability")

    # Add text labels to the bars
    myaxis = g.axes.flatten()
    for ax in myaxis:
        for patch in ax.patches:
            label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle
            label_y = patch.get_y() + patch.get_height()/2
            ax.text(label_x, label_y,
                    '{:.3%}'.format(patch.get_height()),
                    horizontalalignment='center', verticalalignment='center')

    # Show the plot
    plt.show()
    
#plot_categorical(train_data, "Pclass", "Survived", "Sex")

def plot_categorical_relation(df, category_column):
    for col in df.select_dtypes(exclude=["number", "datetime"]).columns:

        plt.figure(figsize=(5, 1))  # Create a new figure for each column
        sns.heatmap(
            pd.crosstab(df[category_column], df[col]), annot=True, fmt="d"
        )
        plt.xticks(rotation=45, ha="right")
        plt.show()
        print("nan sum", col, df[col].isna().sum(), "\n")

### NUMERICAL DATA ###
#    https://seaborn.pydata.org/tutorial/distributions.html
def plot_numerical_data(df, category_column, standard_deviation=None):
    """
    Trace la distribution des colonnes numériques d'un DataFrame par catégorie.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        category_column (str): La colonne catégorielle par laquelle les données seront séparées.
        standard_deviation (float): Le seuil d'écart-type pour filtrer les valeurs aberrantes.
    """
    numeric_columns = df.select_dtypes(include=["number"]).columns
    
    # Supprimer category_column des colonnes numériques si elle est présente
    numeric_columns = [col for col in numeric_columns if col != category_column]
    
    for col in numeric_columns:
        fig, ax = plt.subplots(figsize=(8, 4))  # Nouveau graphique pour chaque colonne

        # Boucle sur chaque catégorie dans la colonne spécifiée
        for category in df[category_column].unique():
            category_df = df[df[category_column] == category]
            
            # Filtrer les valeurs en dehors des seuils de déviation standard
            if standard_deviation is None:
                category_clean = category_df
            else:
                category_clean = category_df[
                    (
                        np.abs(stats.zscore(category_df[col], nan_policy="omit"))
                        < standard_deviation
                    )
                ]

            # Tracer la distribution de la catégorie courante
            sns.histplot(
                category_clean[col],
                label=f"Category {category}",
                kde=True,
                stat="percent",
                common_norm=True,
                alpha=0.25,
                ax=ax,
            )

        ax.set_title(f"Distribution of {col} by Category")  # Titre avec le nom de la colonne
        ax.legend(title=category_column)  # Ajouter une légende avec le nom de la colonne de catégories
        plt.tight_layout()  # Ajuster la disposition pour une meilleure lisibilité
        plt.show()  # Afficher le graphique
        # plt.savefig(f"fig\\histplot_{col}_category_{category}.png")  # Enregistrer si besoin

#sns.pairplot(df[columns_sang].select_dtypes(include=["number"]))

#sns.clustermap(
#    df[columns_efr].select_dtypes(include=["number"]).dropna().corr(),
#    annot=True,
#    fmt=".1f",
#)


### OUTLIER DETECTION ###
def detect_outliers(dataframe, method='zscore', threshold=3, columns=None):
    """
    This function detects outliers in a given dataset using the IQR or Z-score methods.

    Parameters:
    dataframe (pandas.DataFrame): The dataset to detect outliers in.
    method (str, optional): The method to use for outlier detection. Can be 'iqr' or 'zscore'. Defaults to 'zscore'.
    threshold (int, optional): The Z-score threshold. Defaults to 3.

    Returns:
    pandas.DataFrame: The dataset without the outliers.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    # If columns is None, set it to all column names
    if columns is None:
        columns = dataframe.columns

    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame()

    # Analyze each column
    for column in columns:

        if method == 'iqr':
            # Calculate Q1, Q3, and IQR
            Q1 = dataframe[column].quantile(0.25)
            Q3 = dataframe[column].quantile(0.75)
            IQR = Q3 - Q1

            # Identify outliers (any value that is below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]

            # Print summary
            print(f"\nNumber of outliers in {column} (IQR method): {len(outliers)}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")

        elif method == 'zscore':
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(dataframe[column]))

            # Identify outliers (any value with a Z-score above the threshold)
            outliers = dataframe[z_scores > threshold]

            # Print summary
            print(f"\nNumber of outliers in {column} (Z-score method): {len(outliers)}")

        # Store the results in the results DataFrame
        results = results.append(outliers)

    # Remove the outliers from the original DataFrame and return the result
    data_no_outliers = dataframe.drop(results.index)
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dataframe[column])
    plt.title(f'Boxplot for {column}')
    plt.show()

    return data_no_outliers