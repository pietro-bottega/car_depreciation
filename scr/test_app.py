import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import path
import sys


dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)

fipe_features_PCA_path = "../data/output/fipe_features_PCA.csv"

fipe_features_PCA = pd.read_csv(fipe_features_PCA_path)
fipe_features = pd.read_csv("../data/output/fipe_features.csv")
fipe_data = pd.read_csv("../data/output/fipe_data.csv")

print(fipe_features_PCA.head(5))