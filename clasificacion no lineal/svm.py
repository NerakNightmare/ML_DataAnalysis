
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.metrics import accuracy_score
x , y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)

rng = np.random.RandomState(2)
x +=1 * rng.uniform(size = x.shape)
lineary_separable = (x , y)

datasets = [lineary_separable,
            make_moons(noise=0.1),
            make_circles(noise = 0.1, factor = 0.5) ]

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

fig = plt.figure(figsize=(9,3))
h = 0.02

for i, ds in enumerate(datasets):
    ax = plt.subplot(1,3,i+1)
    x, y = ds
    x = StandardScaler().fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x,y)
    
    model = svm.SVC(kernel='rbf')
    model.fit(xtrain, ytrain)
    score_train = model.score(xtrain, ytrain)
    score_test = model.score(xtest, ytest)
    
#dibujo

    xmin, xmax = x[:, 0].min()-0.5, x[:,0].max()+0.5
    ymin, ymax = x[:, 0].min()-0.5, x[:,0].max()+0.5
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))
    
    if(hasattr(model, 'decision_function')):
        zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()])
        
    zz =  zz.reshape(xx.shape)
    
    ax.contourf(xx, yy, zz, cmap=cm, alpha = 0.8)
    
    ax.scatter(xtrain[:,0], xtrain[:,1],c = ytrain, cmap=cm_bright, edgecolors = 'k')

    ax.scatter(xtest[:,0], xtest[:,1],c = ytest, cmap=cm_bright, edgecolors = 'k', alpha = 0.6)
    
    
    
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.text(xmax - 0.3, ymin+0.7, '%.2f'%score_train,
           size = 15, horizontalalignment = 'right')
    
    ax.text(xmax - 0.3, ymin+0.3, '%.2f'%score_test,
           size = 15, horizontalalignment = 'right')

import numpy as np
from sklearn.metrics import confusion_matrix

# Etiquetas reales y predicciones del modelo
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])

# Calcula la matriz de confusión
confusion = confusion_matrix(y_true, y_pred)

# Imprime la matriz de confusión
print(confusion)

accuracy = accuracy_score(ytest, zz)
print(f'Precisión del modelo SVM: {accuracy * 100:.2f}%')
plt.tight_layout() 
plt.show()

 
    