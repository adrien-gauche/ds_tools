

### EXPLAINABILITY ###
def explain_with_shap(X, y, model):
    """Explain the predictions of a given model using SHAP.

    Example:
    import xgboost
    X, y = shap.datasets.california()
    model = xgboost.XGBRegressor()
    shap_values = explain_with_shap(X, y, model)
    """
    import shap

    # Fit the model if it hasn't been already
    if not hasattr(model, "feature_importances_"):
        model.fit(X, y)

    feature_names = [i for i in X.columns if X[i].dtype in [np.int64, np.int64]]
    
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values = explainer(X[feature_names])
    
    #Each dot has three characteristics:
    #Vertical location shows what feature it is depicting
    #Color shows whether that feature was high or low for that row of the dataset
    #Horizontal location shows whether the effect of that value caused a higher or lower prediction.


    # Return the SHAP values and a waterfall plot for the first instance
    shap.plots.waterfall(shap_values[0])
    
    small_val_X = X.iloc[:150]
    shap.summary_plot(shap_values[1], small_val_X) #[1] for binary classification
    
#    # Create object that can calculate shap values
#    explainer = shap.TreeExplainer(model)
#
#    # Calculate Shap values
#    shap_values = explainer.shap_values(model)
#    
#    # use Kernel SHAP to explain test set predictions
#    
#    row_to_show = 5
#    data_for_prediction = X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
#    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
#    
#    k_explainer = shap.KernelExplainer(model.predict_proba, X)
#    k_shap_values = k_explainer.shap_values(data_for_prediction)
#    shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
        
    return shap_values

def explain_with_permutation(X, y, pipeline):
    import eli5
    from eli5.sklearn import PermutationImportance

    perm = PermutationImportance(pipeline, random_state=1).fit(X, y)
    eli5.show_weights(perm, feature_names = X.columns.tolist(), top=10)

    # Extract feature importances and feature names from eli5 output
    feature_importances = perm.feature_importances_
    feature_std = perm.feature_importances_std_
    feature_names = X.columns.tolist()

    # Create a DataFrame to store feature importances, std values, and their corresponding feature names
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances, 'std': feature_std})

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    # Plot the feature importances using seaborn
    plt.figure(figsize=(10,8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.errorbar(x=feature_importance_df['importance'], y=range(len(feature_importance_df)), xerr=feature_importance_df['std'], fmt='o', color='black', capsize=5, elinewidth=1, alpha=0.4)
    plt.title('Feature Permutation Importances')
    plt.show()
    
    from sklearn.inspection import PartialDependenceDisplay
    
    feature_to_plot = 'Distance Covered (Kms)'
    disp = PartialDependenceDisplay.from_estimator(pipeline, X, [feature_to_plot])
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    f_names = [('Goal Scored', 'Distance Covered (Kms)')]
    disp = PartialDependenceDisplay.from_estimator(pipeline, X, f_names, ax=ax)
    plt.show()