# Predictor of car depreciation

**Problem to solve and expected outcome:**
From the moment your car is out of the dealership, it is a used car and depreciation starts.
In Brazil, Tabela FIPE serves as a guide to the industry of historical average price offered for used cars, but it does not provide predictions of future depreciation from newly bought cars.
Based on the FIPE historicals, and a set of features from each car (technical specs, sales data..), the model can input a car that the user intends to buy, and outputs what would be the depreciation in a 3-5 years horizon.

**Data sources:**
- [Tabela FIPE:](https://veiculos.fipe.org.br) national survey of monthly average car prices on resalle for end customers in Brazil.

**Model:**
- Use KNN to measure similarity between cars, and use yearly depreciation from that custer, weighted by distance to centroid.