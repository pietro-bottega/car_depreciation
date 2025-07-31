import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)

fipe_features_PCA_path = "../data/output/fipe_features_PCA.csv"
fipe_features_path = "../data/output/fipe_features.csv"
fipe_data_path = "../data/output/fipe_data.csv"

fipe_features_PCA = pd.read_csv(fipe_features_PCA_path)
fipe_features = pd.read_csv(fipe_features_path)
fipe_data = pd.read_csv(fipe_data_path)

no_models = fipe_features.shape[0] -1

def car_selector(selected_car):
    selected_car_index = int(selected_car)
    selected_car_loc = fipe_features.iloc[selected_car_index]
    selected_car_filter = selected_car_loc[['modelo','marca']]
    
    return pd.DataFrame(selected_car_filter)

def get_latest_value(group):
    sorted_group = group.sort_values(by=['anoref','mesref', 'anomod'], ascending=[False, False,False])
    top_performer = sorted_group.iloc[0]
    return top_performer['valor']

def cars_finder(target_car):
    """
    Performs KNN to find nearest neighboors.
    Creates a table with top 3 models and their prices.
    """

    knn_model = NearestNeighbors(n_neighbors=100, metric='euclidean')
    knn_model.fit(fipe_features_PCA)

    target_car_index = target_car
    target_car_features = fipe_features_PCA.iloc[[target_car_index]]

    distances, indices = knn_model.kneighbors(target_car_features)

    indices = indices[0]
    distances = distances[0]

    selected_cars = fipe_features.loc[indices]
    selected_cars['distance'] = distances

    brand_mask = selected_cars['marca'] != selected_cars.iloc[0]['marca']
    selected_cars_brands = selected_cars[brand_mask]
    selected_cars_brands = selected_cars_brands.head(5)

    models_list = selected_cars_brands['modelo_id'].tolist()
    fipe_data_filtered = fipe_data[fipe_data['modelo_id'].isin(models_list)]

    model_latest_values = pd.DataFrame(fipe_data_filtered.groupby('modelo_id').apply(get_latest_value, include_groups=False), columns=['price'])
    final_view = pd.merge(selected_cars_brands, model_latest_values, on='modelo_id', how = 'left')

    return final_view[['modelo','marca','distance','price']]



# Create the Streamlit interface

st.title("Find similar cars")

user_input = st.text_input(f"Select a car model from {no_models} available", key="target")

display_selected = car_selector(user_input)
display_final_view = cars_finder(user_input)

if user_input is not None:
    st.write("Selected model:")
    display_selected
    st.write("Similar models:")
    display_final_view
    st.write("Lower distance is better. Price considered is the lastest registry in Tabela FIPE")