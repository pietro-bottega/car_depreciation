import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


def select_features(dataset: str) -> pd.DataFrame:
    """
    Reads a dataset from Tabela FIPE data and preprocess.
    Performs feature selection with VarianceThreshold.
    Performs multidimensionality reduction with PCA.

    Paramenters:
    - dataset (str): path with dataframe to be used

    Returns:
    - X_selected_pca_df (pd.DataFrame): dataframe with features reduced to PCs
    """
    fipe_features = pd.read_csv(dataset)

    #-----------------------
    # PRE PROCESSING
    #-----------------------

    print("Preprocessing..")

    ##-----------------------
    ## OneHotEncode CATEGORICAL features
    ##-----------------------

    categorical_cols = ['marca','comb','version','engine','category','transmission_type']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = encoder.fit_transform(fipe_features[categorical_cols])
    fipe_features_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

    # dealing with Non values in categorical columns
    none_patterns = ['_None', '_Other', '_nan']

    columns_to_drop = []
    for col in fipe_features_encoded.columns:
        if any(col.lower().endswith(pattern.lower()) for pattern in none_patterns):
            columns_to_drop.append(col)

    fipe_features_encoded = fipe_features_encoded.drop(columns=columns_to_drop)

    print("Categorical features OneHotEncoded..")
    
    ##-----------------------
    ## Scale NUMERICAL features
    ##-----------------------

    # Getting numerical features and adding to dataframe
    numerical_features = ['doors', 'hp', 'valves']
    fipe_features_processed = pd.concat([fipe_features[numerical_features].reset_index(drop=True), fipe_features_encoded], axis=1)

    # Scaling
    scaler = StandardScaler()
    fipe_features_processed[numerical_features] = scaler.fit_transform(fipe_features_processed[numerical_features])

    print("Numerical features scaled..")

    #-----------------------
    # FEATURE SELECTION
    #-----------------------

    ##-----------------------
    ## VarianceThreshold
    ##-----------------------

    print("Applying VarianceThreshold (.025)")

    selector = VarianceThreshold(threshold=.025 * (1 - .025))
    X_selected = selector.fit_transform(fipe_features_processed)
    selected_feature_names = fipe_features_processed.columns[selector.get_support()]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)

    print(f"Shape of data after VarianceThreshold: {X_selected_df.shape}")
    print(f"Selected features: {selected_feature_names}")

    # Drop highly correlated columns
    X_selected_df = X_selected_df.drop('engine_Diesel', axis=1)

    ##-----------------------
    ## Dimensionality Reduction with PCA
    ##-----------------------

    # PCA does not accept NaN values, present in numerical categories
    X_selected_df_nonan = X_selected_df.drop(['doors', 'hp', 'valves'], axis=1)

    print("Performing PCA for Dimensionality reduction..")
    pca = PCA(n_components=0.95) # number of components that explain 95% of variance
    X_selected_pca = pca.fit_transform(X_selected_df_nonan)
    X_selected_pca_df = pd.DataFrame(X_selected_pca, columns=[f'PC_{i+1}' for i in range(X_selected_pca.shape[1])])

    no_pc = X_selected_pca_df.shape[1]
    print(f"{no_pc} PCs explain 95% of variance")

    return X_selected_pca_df

fipe_features_PCA = select_features('../data/output/fipe_features.csv')
fipe_features_PCA.to_csv("../data/output/fipe_features_PCA.csv")
print("File saved in data/output")