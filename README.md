# Machine Learning (ML) Study on Metastable Halide Perovskites
This repository houses a comprehensive database for pressure- and strain-tuned metastable halide perovskite systems, specifically γ-CsPbI₃, along with a variety of pre-trained ML models designed for predicting band gap and enthalpy based on structural characteristics.

Our research, which involves generating and employing this dataset to uncover the structure-property relationships within perturbed metastable halide perovskite systems through the application of various ML models, is detailed in the subsequent paper:

## Table of Contents
- [Database](#Database)
  * [Pressure](https://github.com/mhan8/Metastable_ML/tree/main/Database/Pressure)
  * [Strain](https://github.com/mhan8/Metastable_ML/tree/main/Database/Strain)
- [Pre-trained ML Models](#Pre-trained)
  * [ALIGNN](https://github.com/mhan8/Metastable_ML/tree/main/ML_Models/ALIGNN)
  * [CGCNN](https://github.com/mhan8/Metastable_ML/tree/main/ML_Models/CGCNN)
  * [Linear_regression](https://github.com/mhan8/Metastable_ML/tree/main/ML_Models/Linear_regression)
  * [Random_forest](https://github.com/mhan8/Metastable_ML/tree/main/ML_Models/Random_forest)

## Database
The database is available in .csv format, encompassing properties like band gap, enthalpy, and total energy, as well as structural features including octahedral tilting, lattice parameters, and Pb-I bond lengths, ensuring comprehensive access to the data. CIF files for CGCNN and ALIGNN predictions have been uploaded.

## Pre-trained ML Models
This repository includes four pre-trained ML models:
* Two classical models, **linear regression** and **random forest**, which users can easily apply using the provided Python code.
* Two advanced graph neural network (GNN) models, the **Crystal Graph Convolutional Neural Network (CGCNN)** and the **Atomistic Line Graph Neural Network (ALIGNN)**.

For those interested in delving deeper into the capabilities of the GNN models, we recommend visiting the following repositories for additional insights and applications:
* [CGCNN](https://github.com/txie-93/cgcnn) (Tian Xie and Jeffrey C. Grossman, _Physical Review Letters_, 2018)
* [ALIGNN](https://github.com/usnistgov/alignn.git) (Kamal Choudhary and Brian DeCost, _npj Computational Materials_, 2021)
