import numpy as np
from sklearn.metrics import confusion_matrix

# Etiquetas reales y predicciones del modelo
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])

# Calcula la matriz de confusión
confusion = confusion_matrix(y_true, y_pred)

# Imprime la matriz de confusión
print(confusion)
