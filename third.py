import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_classification

# Генерируем набор данных без выбросов
X, _ = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Генерируем новые необычные точки данных
new_outliers = np.random.uniform(low=-4, high=4, size=(50, 2))

# Добавляем новые необычные точки данных к исходному набору данных
X_with_outliers = np.vstack([X, new_outliers])

# Обучаем ансамбли и One-Class SVM на наборе данных без выбросов
models = {
    "Случайный лес": RandomForestClassifier(n_estimators=100),
    "Баггинг": BaggingClassifier(n_estimators=100),
    "Метод опорных векторов одного класса SVM": OneClassSVM()
}

plt.figure(figsize=(18, 6))

for i, (name, model) in enumerate(models.items()):
    if name != "Метод опорных векторов одного класса SVM":
        y = np.zeros(len(X_with_outliers))
        y[-len(new_outliers):] = 1  # Marks new outliers as positive instances
        model.fit(X_with_outliers, y)
        y_pred = model.predict(X_with_outliers)
    else:
        model.fit(X_with_outliers)
        y_pred = model.predict(X_with_outliers)
        y_pred[y_pred == 1] = 0  # Revert predictions for inliers
        y_pred[y_pred == -1] = 1  # Mark outliers as positive instances

    plt.subplot(1, 3, i + 1)
    plt.title(name)
    plt.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=y_pred, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
