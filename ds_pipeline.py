#https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats

#pip install pipreqs
#pipreqs /path/to/project
#pip install -r requirements.txt

# Feature Scaling
# One-hot Encoding
# Dimensionality Reduction
# Handling Duplicate Data
# Data Normalization

### CLEAN DATA ###
# better to use pipeline for all these functions, see below

### BUILDING A PIPELINE ###
#cleaning imputer, scaling, encoding, model

def remove_duplicates(df):
    """
    This function removes duplicate rows from a given DataFrame.
    """
    # remove duplicates
    df_cleaned = df.drop_duplicates()

    return df_cleaned


### MISSING

def process_missing_data(df, method='indicator'):
    """
    This function processes missing data in a DataFrame based on the specified method.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    method (str): The method to use for processing missing data.
        'indicator': Adds a binary indicator column for each column, denoting whether the value in that row is missing (NaN) or not.
        'drop_rows': Removes all rows with missing data from the DataFrame.
        'drop_columns': Removes all columns with missing data from the DataFrame.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """
    if method == 'indicator':
        df_indicator = pd.DataFrame(df.isnull(), columns=df.columns + "_missing_indicator")
        df_processed = pd.concat([df, df_indicator], axis=1)
    elif method == 'drop_rows':
        df_processed = df.dropna()
    elif method == 'drop_columns':
        cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
        df_processed = df.drop(cols_with_missing, axis=1)
        print("Columns in original dataset: %d \n" % df.shape[1])
        print("Columns with na's dropped: %d" % df_processed.shape[1])
    else:
        raise ValueError("Invalid method. Choose from 'indicator', 'drop_rows', 'drop_columns'.")

    return df_processed



### CATEGORY ENCODING ###
def get_column_types(df, threshold=10):
    # Categorical columns in the training data
    object_cols = [col for col in df.columns if df[col].dtype == "object"]

    # Get number of unique entries in each column with categorical data
    object_nunique = list(map(lambda col: df[col].nunique(), object_cols))

    # Create a dictionary from the two lists
    d = dict(zip(object_cols, object_nunique))

    # Sort the dictionary by value (unique entry count)
    sorted_d = dict(sorted(d.items(), key=lambda x: x[1]))

    print(sorted_d)

    # Columns that will be one-hot encoded
    low_cardinality_cols = [
        col
        for col in object_cols
        if df[col].nunique() < threshold and df[col].dtype == "object"
    ]

    # Columns that will be dropped or label encoded
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

    # Select numerical columns
    numerical_cols = [
        cname for cname in df.columns if df[cname].dtype in ["int64", "float64"]
    ]

    return low_cardinality_cols, high_cardinality_cols, numerical_cols


def drop_categorical_columns(df):
    # Drop the categorical columns from the DataFrame
    drop_df = df.select_dtypes(exclude=["object"])

    return drop_df


def ordinal_encode(df):
    from sklearn.preprocessing import OrdinalEncoder

    # Categorical columns in the training data
    object_cols = [col for col in df.columns if df[col].dtype == "object"]

    # Make copy to avoid changing original data
    label_df = df.copy()

    # Apply ordinal encoder to each column with categorical data
    ordinal_encoder = OrdinalEncoder()
    label_df[object_cols] = ordinal_encoder.fit_transform(df[object_cols])

    return label_df


def one_hot_encode_low_cardinality(df):
    # one column per category with a 1 or 0 value for each label, for low cardinality columns

    from sklearn.preprocessing import OneHotEncoder

    # Categorical columns in the training data
    object_cols = [col for col in df.columns if df[col].dtype == "object"]

    # Columns that will be one-hot encoded
    low_cardinality_cols = [
        col
        for col in object_cols
        if df[col].nunique() < 10 and df[col].dtype == "object"
    ]

    # PARAMETERS
    # handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data, and
    # setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
    # Apply one-hot encoder to each low cardinality column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(df[low_cardinality_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = df.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_df = df.drop(low_cardinality_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_df = pd.concat([num_df, OH_cols_train], axis=1)

    # Ensure all columns have string type
    OH_df.columns = OH_df.columns.astype(str)

    return OH_df


### SCALING AND NORMALIZATION ##
def scale_data(df, method="standard"):
    """
    This function scales the data in a given DataFrame using different approaches.

    Parameters:
    df (pandas.DataFrame): The DataFrame to scale.
    method (str, optional): The scaling method to use. Defaults to 'standard'.
        'standard': Standardize features by removing the mean and scaling to unit variance.
        'minmax': Scale features to lie between a given minimum and maximum value, often between 0 and 1.
        'maxabs': Scale each feature by its maximum absolute value.
        'robust': Scale features using statistics that are robust to outliers.

    Returns:
    pandas.DataFrame: The scaled DataFrame.
    """
    if method == "standard":
        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    elif method == "minmax":
        # Scale features to lie between a given minimum and maximum value, often between 0 and 1
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    elif method == "maxabs":
        # Scale each feature by its maximum absolute value
        from sklearn.preprocessing import MaxAbsScaler

        scaler = MaxAbsScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    elif method == "robust":
        # Scale features using statistics that are robust to outliers
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    else:
        raise ValueError(
            "Invalid scaling method. Please choose from 'standard', 'minmax', 'maxabs', 'robust'."
        )

    # return the scaled DataFrame
    return df_scaled


### FEATURE ENGINEERING ###
# Tips on Creating Features
# Consider the strengths and weaknesses of your model when creating features.
#
# # create new features by combining existing features, such as physical equations, surface...
# Linear models can easily learn sums and differences, but struggle with complex patterns.
# Ratios seem to be difficult for most models to learn but ratio combinations often lead to some easy performance gains.
# Linear models and neural nets generally do better with normalized features. Neural nets especially need features scaled to values not too far from 0. Tree-based models (like random forests and XGBoost) can sometimes benefit from normalization, but usually much less so.
# Tree-based models (e.g. random forests, XGBoost) can learn to approximate most feature combinations, but may still improve when important combinations are explicitly provided, especially with limited data.
# Counts are especially helpful for tree models, since these models don't have a natural way of aggregating information across many features at once.


import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson, kstest
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt

def reshape_data(data, plot: bool = False):
    """
    This function reshapes data using different methods based on its skewness.

    Parameters:
    data (pd.Series or np.ndarray): The data to be reshaped.
    plot (bool): Whether to plot the original and transformed data distributions.

    Returns:
    pd.DataFrame: A DataFrame with KS statistics and p-values for each transformation.
    
    See: https://machinelearningmastery.com/skewness-be-gone-transformative-tricks-for-data-scientists/
    """
    # Ensure data is a pandas Series for skewness calculation
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    data = data.dropna()
    
    # Calculate skewness
    skewness = data.skew()
    print("Skewness: ", skewness)
    
    # Initialize dictionary for transformations
    transformations_dict = {}

    if skewness > 0:
        print("Data is right-skewed.")
        
        if (data > 0).all():
            transformations_dict["log"] = np.log(data) # Logarithmic transformation
            transformations_dict["log1p"] = np.log1p(data) # Logarithmic transformation
            transformations_dict["sqrt"] = np.sqrt(data) # Square root transformation
            transformed_data, _ = boxcox(data) # Box-Cox transformation
            transformations_dict["boxcox"] = transformed_data
            
        if (data > -1).all():
            transformations_dict["log1p"] = np.log1p(data) # Logarithmic transformation
        
        transformations_dict["yeojohnson"], _ = yeojohnson(data) # Yeo-Johnson transformation
        transformations_dict["quantile"] = QuantileTransformer(output_distribution='normal').fit_transform(data.values.reshape(-1, 1)).flatten() # Quantile transformation
        
    else:
        print("Data is left-skewed.")
        
        if (data > 0).all():
            transformations_dict["log"] = np.log(data) # Logarithmic transformation
            transformations_dict["log1p"] = np.log1p(data) # Logarithmic transformation
            transformations_dict["sqrt"] = np.sqrt(data) # Square root transformation
            transformed_data, _ = boxcox(data) # Box-Cox transformation
            transformations_dict["boxcox"] = transformed_data
        
        transformations_dict["cbrt"] = np.cbrt(data) # Cube root transformation
        transformations_dict["yeojohnson"], _ = yeojohnson(data) # Yeo-Johnson transformation
        transformations_dict["quantile"] = QuantileTransformer(output_distribution='normal').fit_transform(data.values.reshape(-1, 1)).flatten() # Quantile transformation
    
    # Perform Kolmogorov-Smirnov Test
    ks_test_results = {}
    for transformation, transformed_data in transformations_dict.items():
        standardized_data = (transformed_data - np.mean(transformed_data)) / np.std(transformed_data)
        ks_stat, ks_p_value = kstest(standardized_data, 'norm')
        ks_test_results[transformation] = (ks_stat, ks_p_value, pd.Series(standardized_data).skew())
    
    # Convert results to DataFrame for easier comparison
    ks_test_results_df = pd.DataFrame.from_dict(ks_test_results, orient='index', columns=['KS Statistic', 'P-Value', 'Skewness'])
    print(ks_test_results_df.round(5))
    
    # Optionally plot the results
    if plot:
        plt.figure(figsize=(12, 8))
        for label, transformed_data in transformations_dict.items():
            plt.hist(transformed_data, bins=30, alpha=0.5, label=label)
        plt.legend()
        plt.title('Distribution of Transformed Data')
        plt.show()
    
    return ks_test_results_df


# clustering
def kmeans_clustering(X, n_clusters=10):

    from sklearn.cluster import KMeans

    # Standardize with StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Fit the KMeans model to X_scaled and create the cluster labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=None)  # , n_init=10
    X["Cluster"] = kmeans.fit_predict(X_scaled)

    # Convert Cluster to category
    X["Cluster"] = X["Cluster"].astype("category")

    return X

def clustering_algorithms(X, eps=0.3, min_samples=10, n_components=2, n_clusters=3):
    """
    Apply four clustering algorithms to the data X: DBSCAN, hierarchical clustering,
    spectral clustering, and Gaussian mixture models.

    Args:
    X (DataFrame or array): The data to be clustered.
    eps (float, optional): The maximum distance between two samples for DBSCAN.
    min_samples (int, optional): The number of samples in a neighborhood for DBSCAN.
    n_components (int, optional): The number of dimensions for spectral clustering.
    n_clusters (int, optional): The number of clusters for hierarchical clustering,
                                  spectral clustering, and Gaussian mixture models.

    Returns:
    dict: A dictionary of dataframes, where the keys are the names of the algorithms
          and the values are the dataframes with the cluster labels.
    """
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
    from sklearn.mixture import GaussianMixture

    # Standardize with StandardScaler
    scaler = StandardScaler()
    X_scaled = (
        pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        if isinstance(X, pd.DataFrame)
        else pd.DataFrame(
            scaler.fit_transform(X),
            columns=["Feature " + str(i) for i in range(X.shape[1])],
        )
    )

    # DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    # This algorithm groups together points that are close in the feature space,
    # marking points in sparse areas as outliers.
    #useful for identifying clusters of different shapes and sizes,
    # and for handling outliers. It is particularly effective when the clusters are
    # separated by areas of low density.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    X_dbscan = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_dbscan["Cluster"] = pd.Categorical(dbscan.fit_predict(X_scaled))
    X_dbscan["Cluster"] = X_dbscan["Cluster"].cat.add_categories(["Noise"])
    X_dbscan.loc[X_dbscan["Cluster"] == -1, "Cluster"] = "Noise"

    # Hierarchical Clustering
    # This is a family of algorithms that build nested clusters by creating a tree
    # of clusters. Hierarchical clustering can be agglomerative (starting with
    # each point as a separate cluster and merging them) or divisive (starting with
    # all points in one cluster and splitting them).
    # useful for understanding the structure of the data and for identifying a hierarchy of
    # clusters.
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    X_hierarchical = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_hierarchical["Cluster"] = pd.Categorical(hierarchical.fit_predict(X_scaled))

    # Spectral Clustering
    # This algorithm uses the eigenvalues of a similarity matrix to reduce the
    # dimensionality of the data before clustering in a lower-dimensional space.
    # It can be useful when the clusters are not well-separated in the original
    # feature space, or when the number of features is large.
    spectral = SpectralClustering(n_clusters=n_clusters, n_components=n_components)
    X_spectral = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_spectral["Cluster"] = pd.Categorical(spectral.fit_predict(X_scaled))

    # Gaussian Mixture Models (GMM)
    # GMMs model the data as a mixture of Gaussian distributions. They can be useful
    # when the clusters are not well-separated or when they have different shapes
    # or sizes. GMMs can also be used for density estimation and for
    # identifying outliers.
    gmm = GaussianMixture(n_components=n_clusters)
    X_gmm = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_gmm["Cluster"] = pd.Categorical(gmm.fit_predict(X_scaled))

    return {
        "DBSCAN": X_dbscan,
        "Hierarchical Clustering": X_hierarchical,
        "Spectral Clustering": X_spectral,
        "Gaussian Mixture Models": X_gmm,
    }


def plot_clusters(X, y):
    Xy = X.copy()
    Xy["Cluster"] = Xy.Cluster.astype("category")
    Xy[y.name] = y
    sns.relplot(
        x="value",
        y=y.name,
        hue="Cluster",
        col="variable",
        # height=4,
        # aspect=1,
        facet_kws={"sharex": False},
        col_wrap=3,
        data=Xy.melt(
            value_vars=X.columns,
            id_vars=[y.name, "Cluster"],
        ),
    )

    sns.catplot(x=y.name, y="Cluster", data=Xy, kind="boxen")


def kmeans_cluster_distance(X, n_clusters=10):

    from sklearn.cluster import KMeans

    # Standardize with StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    kmeans = KMeans(n_clusters=n_clusters, random_state=None)  # , n_init=10
    # Fit the KMeans model to X_scaled and create the cluster labels
    # Create the cluster-distance features using `fit_transform`
    X_cd = kmeans.fit_transform(
        X_scaled
    )  # each row contains the distance to each centroid

    # Label features and join to dataset
    X_cd = pd.DataFrame(
        X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])], index=X.index
    )
    X = X.join(X_cd)  # add distance to each centroid in columns

    return X


### DIMENSIONALITY REDUCTION ###
# analyses
# Dimensionality reduction: When your features are highly redundant (multicollinear, specifically), PCA will partition out the redundancy into one or more near-zero variance components, which you can then drop since they will contain little or no information.
# Anomaly detection: Unusual variation, not apparent from the original features, will often show up in the low-variance components. These components could be highly informative in an anomaly or outlier detection task.
# Noise reduction: A collection of sensor readings will often share some common background noise. PCA can sometimes collect the (informative) signal into a smaller number of features while leaving the noise alone, thus boosting the signal-to-noise ratio.
# Decorrelation: Some ML algorithms struggle with highly-correlated features. PCA transforms correlated features into uncorrelated components, which could be easier for your algorithm to work with.

# PCA Best Practices
# There are a few things to keep in mind when applying PCA:
#    PCA only works with numeric features, like continuous quantities or counts.
#    PCA is sensitive to scale. It's good practice to standardize your data before applying PCA, unless you know you have good reason not to.
#    Consider removing or constraining outliers, since they can have an undue influence on the results.

from sklearn.decomposition import PCA

def apply_pca(X, standardize=True):
    """
    Create principal components from the data.

    Parameters
    ----------
    X : DataFrame
        The input data.
    standardize : bool, optional
        Standardize the features before applying PCA, by default True.

    Returns
    -------
    pca : PCA
        The sklearn PCA object.
    X_pca : DataFrame
        The principal components, the new transformed representation of the data.
    loadings : DataFrame
        The loadings matrix, coefficients that define the components. Shows how much each original feature contributes to each component.
    """
    # Standardize
    if standardize:
        # Standardize with StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)

    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=X.columns,
    )

    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    """
    Plot the explained variance and cumulative variance of the PCA components.

    Parameters
    ----------
    pca : PCA
        The sklearn PCA object.
    width : int, optional
        The width of the figure, by default 8.
    dpi : int, optional
        The dots per inch of the figure, by default 100.

    Returns
    -------
    axs : seaborn.axisgrid.FacetGrid
        The FacetGrid objects of the two subplots.
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(width, 4), dpi=dpi)
    n = pca.n_components_
    grid = np.arange(1, n + 1)

    # Explained variance
    evr = pca.explained_variance_ratio_
    sns.barplot(x=grid, y=evr, ax=axs[0])
    axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))

    # Cumulative Variance
    cv = np.cumsum(evr)
    sns.lineplot(x=np.r_[0, grid], y=np.r_[0, cv], marker="o", ax=axs[1])
    axs[1].set(xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))

    # Set up figure
    fig.tight_layout()

    return axs


def plot_outliers_boxen(X_pca, col_wrap=2):
    """
    Detect outliers in a dataset using the boxen plot.

    Parameters
    ----------
    X_pca : DataFrame
        The input data.
    col_wrap : int, optional
        The number of columns to wrap the subplots, by default 2.

    Returns
    -------
    None
        The function plots the boxen plots to detect outliers.
    """
    # Melt the dataframe to create a categorical variable for each column
    melted_df = pd.melt(X_pca, id_vars=[], value_vars=X_pca.columns)

    # Rename the columns for better readability
    melted_df = melted_df.rename(columns={"variable": "Feature", "value": "Value"})

    # Create the boxen plot
    sns.catplot(
        y="Value",
        col="Feature",
        data=melted_df,
        kind="boxen",
        sharey=False,
        col_wrap=col_wrap,
    )


def get_sorted_outliers(X, X_pca, component="PC1"):
    """
    Get the rows of the DataFrame X sorted by the values of the principal component.
    """
    # Sort the values of the column "component" in descending order
    idx = X_pca[component].sort_values(ascending=False).index

    # Select the rows of the DataFrame "X" based on the sorted row numbers
    sorted_rows = X.iloc[idx]

    return sorted_rows


def biplot(pca, PC_x=0, PC_y=1):
    """Create a biplot for the PCA components.

    Parameters
    ----------
    pca : PCA
        The sklearn PCA object.
        PC_x : int, optional
        The principal component for the x-axis, by default 0.
        PC_y : int, optional
        The principal component for the y-axis, by default 1.

    Usage : biplot(pca, 1, 6)

    By Joachim Schork, https://statisticsglobe.com/biplot-pca-python"""
    score = pca.components_
    coef = np.transpose(pca.components_)
    labels = list(pca.feature_names_in_)

    xs = score[:, 0]
    ys = score[:, 1]
    n = coef.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, s=5, color="orange")

    for i in range(n):
        plt.arrow(0, 0, coef[i, PC_x], coef[i, PC_y], color="purple", alpha=0.5)
        plt.text(
            coef[i, PC_x] * 1.15,
            coef[i, PC_y] * 1.15,
            labels[i],
            color="darkblue",
            ha="center",
            va="center",
        )

    plt.xlabel("PC{}".format(PC_x))
    plt.ylabel("PC{}".format(PC_y))

    plt.figure()


### SCORES ###
# from sklearn.metrics import roc_auc_score
# train_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
# test_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
# print("train",train_auc)
# print("test", test_auc)

# from sklearn.metrics import confusion_matrix
# y_train_hat = clf.predict(X_train)
# y_test_hat = clf.predict(X_test)
# print(confusion_matrix(y_test, y_test_hat))

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_test_hat))

### CROSS VALIDATION ###
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
# print(scores.mean())

# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(model, parameters, cv = 5, scoring = 'roc_auc_ovr', verbose = 1)
# clf.fit(X, y)

### PIPELINE ###


def create_pipeline(X, y, model):
    """
    This function implements a data preprocessing pipeline using scikit-learn.

    Parameters:
    X (pandas.DataFrame): The features.
    y (pandas.Series): The target variable.

    Returns:
    tuple: (X_transformed, pipeline), where X_transformed is the preprocessed data and
            pipeline is the fitted scikit-learn pipeline.
    """

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [
        cname
        for cname in X.columns
        if X[cname].nunique() < 10 and X[cname].dtype == "object"
    ]

    # Select numerical columns
    numerical_cols = [
        cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
    ]
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    
    from sklearn.compose import make_column_selector
    numerical_features = make_column_selector(dtype_include=[np.number, bool])
    categorical_features = make_column_selector(dtype_exclude=[np.number, bool])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        #zero, mean, median, ffill, bfill, interpolate
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        #("imputer", SimpleImputer(strategy="constant", fill_value="missing"),),
        ('onehot', OneHotEncoder()),
        #('ordinal', OrdinalEncoder()),
    ])

    preprocessor = ColumnTransformer(
        (numerical_pipeline, numerical_features),
        (categorical_pipeline, categorical_features)
    )

    ## Define model
    # model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Bundle preprocessing and modeling code in a pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Use the pipeline to transform the data
    X_transformed = pipeline.transform(X)

    from sklearn.model_selection import cross_val_score

    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(
        pipeline, X, y, cv=3, scoring="neg_mean_absolute_error"
    )

    # Preprocessing of training data, fit model
    pipeline.fit(X, y)

    # Preprocessing of validation data, get predictions
    preds = pipeline.predict(X)
    
    print("Mean Absolute Error:", )