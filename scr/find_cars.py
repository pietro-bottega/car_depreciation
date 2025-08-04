import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

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

    target_car_index = model_no
    target_car_features = fipe_features_PCA.iloc[[target_car_index]]

    distances_pca, indices = knn_model.kneighbors(target_car_features)

    indices = indices[0]
    distances = distances[0]

    selected_cars = fipe_features.loc[indices]
    selected_cars['similarity'] = distances

    brand_mask = selected_cars['marca'] != selected_cars.iloc[0]['marca']
    selected_cars_brands = selected_cars[brand_mask]
    selected_cars_brands = selected_cars_brands.head(5)

    models_list = selected_cars_brands['modelo_id'].tolist()
    fipe_data_filtered = fipe_data[fipe_data['modelo_id'].isin(models_list)]

    model_latest_values = pd.DataFrame(fipe_data_filtered.groupby('modelo_id').apply(get_latest_value, include_groups=False), columns=['price'])
    final_view = pd.merge(selected_cars_brands, model_latest_values, on='modelo_id', how = 'left')

    return final_view

def create_TSNE(dataset):
    """
    Plots models into 2d dimensions for easier visualization.

    Paramenters:
    - dataset (str): path with dataframe to be used

    Returns:
    - X_TSNE (pd.DataFrame): dataframe with coordinates into 2d space of each car model
    """
    X = pd.read_csv(dataset)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42
    )

    X_TSNE = tsne.fit_transform(X)

    return pd.DataFrame(X_TSNE) 

car_models_TSNE = create_TSNE('../data/output/fipe_features_PCA.csv')
car_models_TSNE.to_csv('../data/output/car_models_TSNE.csv')

fipe_features_PCA = pd.read_csv("../data/output/fipe_features_PCA.csv")
fipe_features = pd.read_csv("../data/output/fipe_features.csv")
fipe_data = pd.read_csv("../data/output/fipe_data.csv")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne_v2 = TSNE(
    n_components=2,      # We want a 2D plot
    perplexity=30,       # A good default value, relates to the number of nearest neighbors
    n_iter=1000,         # Number of optimization iterations
    random_state=42      # For reproducible results
)
X_tsne_v2 = tsne_v2.fit_transform(fipe_features_PCA)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))
plt.scatter(
    x=X_tsne_v2[:, 0],
    y=X_tsne_v2[:, 1])

plt.title('t-SNE Visualization of Feature Space for KNN')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

print("Created model and TSNE map")