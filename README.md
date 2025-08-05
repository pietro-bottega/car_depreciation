# Similar Car Finder Project

This project analyzes vehicle data to identify and suggest similar car models based on their technical and market features. The core of the project involves data processing, dimensionality reduction, and applying a nearest neighbors algorithm to find similarities.
<br>

> [!TIP]
> This [**interactive app**](https://car-finder.streamlit.app) can be used to explore results.

---

## Methodology

The project's methodology follows a structured data science workflow to ensure robust and meaningful results. The process begins with raw data collection and culminates in a model capable of identifying similar vehicles.

The main steps are:
1.  **Data Acquisition and Cleaning**: Data was sourced from the [*Tabela FIPE*:](https://veiculos.fipe.org.br), a national survey of monthly average car prices on resalle for end customers in Brazil. I leveraged the [fipe-crawler](https://github.com/rafaelgou/fipe-crawler/) created by @rafaelgou.
2.  **Feature Engineering and Selection**: Relevant features were selected and engineered, extracted from the `model` description. To reduce noise and computational complexity, low-variance features were removed using `VarianceThreshold`.
3.  **Dimensionality Reduction**: Principal Component Analysis (PCA) was applied to reduce the number of features while retaining the maximum possible information from the dataset.
4.  **Model Application**: A K-Nearest Neighbors (KNN) model was used on the reduced dataset to find the closest "neighbors" (i.e., most similar cars) for any given vehicle.
5.  **Visualization**: To visually inspect the results and the effectiveness of the dimensionality reduction, t-SNE was used to plot the high-dimensional data in a 2D space.

---

## Usage

Modules performing main steps of analysis:
1. `feature_extraction.py`: extracting model features via regex patterns.
2. `feature_selection.py`: selecting features via variance threshold and correlation, dimensionality reduction with PCA.
3. `find_cars.py`: implementing similarity model via K-Nearest Neighboors.

---

## Tools and Technologies

The project was developed using Python and a standard stack of data science libraries:

* **Python 3.x**
* **Pandas & NumPy**: For data manipulation and numerical operations.
* **Scikit-learn**: For implementing machine learning models and preprocessing techniques, including `VarianceThreshold`, `PCA`, `TSNE`, and `NearestNeighbors`.
* **Matplotlib & Seaborn**: For data visualization.
* **Jupyter Notebook**: As the primary environment for development and analysis.
* **Streamlit**: to build an interactive app.

---

## Data Visualization and Feature Selection

Visualizations were crucial for understanding the data and validating the feature selection and dimensionality reduction steps.

### Dataset Overview

The dataset is derived from the **Tabela FIPE**, which is a primary reference for average vehicle market prices in Brazil. It contains a wide array of technical specifications and pricing information for a very extensive list of car models.
<br>


#### Distribuition of car values:
![Distribuition of car values](https://raw.githubusercontent.com/pietro-bottega/car_depreciation/refs/heads/issue22/assets/car_value_no_outlier_histplot.png)

### VarianceThreshold

This technique is used as a baseline for feature selection. It removes all features whose variance doesn't meet a certain threshold. In practice, this is used to eliminate features with little to no variation across all samples (e.g., a feature that is 'True' for 99% of the cars), as they provide little to no information for the model.

#### Feature Variance:
![Feature Variance](https://raw.githubusercontent.com/pietro-bottega/car_depreciation/refs/heads/issue22/assets/features_variance.png)

### Principal Component Analysis (PCA)

PCA is a dimensionality reduction method used to transform a large set of variables into a smaller one that still contains most of the information in the large set. The optimal number of components was selected by plotting the cumulative explained variance against the number of components. The goal was to find the "elbow point" where adding another component yields diminishing returns, thus retaining over 95% of the variance with significantly fewer features.

#### PCA Explained Variance
![PCA Explained Variance](https://raw.githubusercontent.com/pietro-bottega/car_depreciation/refs/heads/issue22/assets/pca_cumulative_explained_variance.png)

### t-SNE Visualization

t-SNE (t-distributed Stochastic Neighbor Embedding) is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map. It's particularly useful for revealing the underlying structure of the data, such as clusters. In this project, it was used to visually confirm that the feature space created by PCA effectively groups similar cars together, validating the model's ability to find meaningful similarities.

#### t-SNE plot into 2 dimensions
![t-SNE Plot](https://raw.githubusercontent.com/pietro-bottega/car_depreciation/refs/heads/issue22/assets/t-SNE_2d_v2.png)