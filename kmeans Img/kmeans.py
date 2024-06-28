import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.cluster import KMeans
from sklearn.utils import shuffle

img = mpimg.imread("gato.jpg")

plt.figure()
plt.title("Imagen original")

plt.imshow(img)

img = np.array(img, dtype = np.float64) / 255.0

w, h, c = img.shape
n_cluster = 2
n_samples = 1000

img_array = np.reshape(img, (w*h, c))
image_sample = shuffle(img_array) [:n_samples]

model = KMeans(n_clusters = n_cluster).fit(image_sample)

labels = model.predict(img_array)
img_labels = np.reshape(labels, (w , h))

img_final = np.zeros((w, h, c))

for i in range(w):
    for j in range(h):
        img_final[i,j,:] = model.cluster_centers_[img_labels[i,j]][:]



plt.figure()
plt.title("Imagen Cuantizada")
plt.imshow(img_final)

plt.show()

#from sklearn.metrics import silhouette_score

#score = silhouette_score(img_array, labels)

#print("Score: ", score)


