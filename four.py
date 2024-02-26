import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import Birch, KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

# Генерация синтетических данных
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Стандартизация данных
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Ввод параметров из консоли
n_clusters = int(input("Enter the maximum number of clusters: "))
threshold = float(input("Enter the merging threshold: "))

# Создание модели Birch с заданными параметрами
birch_model = Birch(threshold=threshold, n_clusters=n_clusters)
birch_model.fit(X_std)
birch_labels = birch_model.predict(X_std)

# Создание моделей KMeans и AgglomerativeClustering для сравнения
kmeans_model = KMeans(n_clusters=4, random_state=0)
kmeans_labels = kmeans_model.fit_predict(X_std)

agg_model = AgglomerativeClustering(n_clusters=4)
agg_labels = agg_model.fit_predict(X_std)

# Оценка качества кластеризации
birch_silhouette = silhouette_score(X_std, birch_labels)
kmeans_silhouette = silhouette_score(X_std, kmeans_labels)
agg_silhouette = silhouette_score(X_std, agg_labels)

# Визуализация результатов
plt.figure(figsize=(12, 6))

# Подготовка цветов для кластеров
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf']), int(max(birch_labels) + 1))))
colors = np.append(colors, ["#000000"])  # Добавляем цвет для шумов

# Birch
plt.subplot(131)
plt.scatter(X_std[:, 0], X_std[:, 1], s=10, color=colors[birch_labels])
plt.title(f'Birch (Silhouette Score: {birch_silhouette:.3f})')

# KMeans
plt.subplot(132)
plt.scatter(X_std[:, 0], X_std[:, 1], s=10, color=colors[kmeans_labels])
plt.title(f'KMeans (Silhouette Score: {kmeans_silhouette:.3f})')

# Agglomerative Clustering
plt.subplot(133)
plt.scatter(X_std[:, 0], X_std[:, 1], s=10, color=colors[agg_labels])
plt.title(f'Agglomerative Clustering (Silhouette Score: {agg_silhouette:.3f})')

plt.show()

print("Birch Parameters:")
print(f"Threshold: {birch_model.threshold}")
print(f"Number of Clusters: {birch_model.n_clusters}")

kmeans_centers = kmeans_model.cluster_centers_
print("\nKMeans Cluster Centers:")
print(kmeans_centers)

print("\nAgglomerative Clustering Parameters:")
print(f"Number of Clusters: {agg_model.n_clusters}")
