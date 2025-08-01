import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import path

# Loading data

@st.cache_data
def load_data():
    """Load all dataframe from repo"""

    fipe_features_PCA_path = "https://raw.githubusercontent.com/pietro-bottega/car_depreciation/refs/heads/issue22/data/output/fipe_features_PCA.csv"
    fipe_features_path = "https://raw.githubusercontent.com/pietro-bottega/car_depreciation/refs/heads/issue22/data/output/fipe_features.csv"
    fipe_data_path = "https://raw.githubusercontent.com/pietro-bottega/car_depreciation/refs/heads/issue22/data/output/fipe_data.csv"

    try:
        fipe_features_PCA = pd.read_csv(fipe_features_PCA_path, encoding='latin1')
        fipe_features = pd.read_csv(fipe_features_path, encoding='latin1')
        fipe_data = pd.read_csv(fipe_data_path, encoding='latin1')
        return fipe_features_PCA, fipe_features, fipe_data
    except Exception as e:
        st.error(f"Error loading repo data: {e}")
        return None, None, None

fipe_features_PCA, fipe_features, fipe_data = load_data()

# Model training

@st.cache_resource
def get_knn_model(data_pca):
    """Creates and trains the model"""
    if data_pca is not None:
        knn_model = NearestNeighbors(n_neighbors=100, metric='euclidean')
        knn_model.fit(data_pca)
        return knn_model
    return None

if fipe_features_PCA is not None and fipe_features is not None and fipe_data is not None:
    knn_model = get_knn_model(fipe_features_PCA)
else:
    knn_model = None

# Complementary functions

def car_selector(car_index: int):
    """Returns data from selected car based on index"""
    selected_car_loc = fipe_features.iloc[car_index]
    selected_car_filter = selected_car_loc[['modelo','marca']]
    return pd.DataFrame(selected_car_loc[['modelo','marca']]).T

def get_latest_value(group):
    """Gets latest value from a group of car models"""
    sorted_group = group.sort_values(by=['anoref','mesref', 'anomod'], ascending=[False, False,False])
    return sorted_group.iloc[0]['valor']

def cars_finder(car_index: int):
    """
    Performs KNN to find nearest neighboors.
    Creates a table with top 3 models and their prices.
    """

    target_car_features = fipe_features_PCA.iloc[[car_index]]
    distances, indices = knn_model.kneighbors(target_car_features)

    indices = indices[0]
    distances = distances[0]

    selected_cars = fipe_features.loc[indices]
    selected_cars['distance'] = distances

    brand_mask = selected_cars['marca'] != selected_cars.iloc[0]['marca']
    selected_cars_brands = selected_cars[brand_mask].head(5)

    models_list = selected_cars_brands['modelo_id'].tolist()
    fipe_data_filtered = fipe_data[fipe_data['modelo_id'].isin(models_list)]

    model_latest_values = pd.DataFrame(fipe_data_filtered.groupby('modelo_id').apply(get_latest_value, include_groups=False), columns=['price'])
    final_view = pd.merge(selected_cars_brands, model_latest_values, on='modelo_id', how = 'left')

    return final_view[['modelo','marca','distance','price']]

# Create the Streamlit interface

st.title("Find similar cars")

if knn_model is None:
    st.warning("Error while loading models, review logs.")
else:
    fipe_features['display_name'] = fipe_features['marca'] + " - " + fipe_features['modelo']
    model_otions = fipe_features['display_name'].sort_values().tolist()

    selected_model = st.selectbox(
        label="Select a car model from the list:",
        options=model_otions,
        index=None,
        placeholder="Select a car to find similar models"
    )

    if selected_model:
        try:
            target_car_index = fipe_features[fipe_features['display_name'] == selected_model].index[0]

            with st.spinner("Calculating.."):
                selected_model_df = car_selector(target_car_index)
                similar_models_df = cars_finder(target_car_index)
                
            st.subheader("Selected model:")
            st.dataframe(selected_model_df)

            st.subheader("Similar models:")
            st.dataframe(similar_models_df)

            st.info("Lower distance, higher similarity. Price considered here is the lastest record in Tabela FIPE")

        except Exception as e:
            st.error(f"Unexpected error: {e}")