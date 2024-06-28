import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
data = pd.read_csv('home_data.csv')  # Asegúrate de proporcionar la ubicación correcta del archivo

print(data.info())