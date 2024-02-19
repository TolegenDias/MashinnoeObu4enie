import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def calculate_distance(point, center, metric='euclidean'):
    if metric == 'euclidean':
        return np.linalg.norm(point - center)
    elif metric == 'chebyshev':
        return np.max(np.abs(point - center))
    elif metric == 'manhattan':
        return np.sum(np.abs(point - center))
    else:
        raise ValueError("Unsupported metric")

def calculate_user_class(user_vector, centers):
    distances = {class_name: calculate_distance(user_vector, center) for class_name, center in centers.items()}
    return min(distances, key=distances.get)

def get_class_color(class_name):
    class_colors = {'budget': 'b', 'medium': 'g', 'expensive': 'r', 'premium': 'm'}  # Add 'premium' class color
    return class_colors.get(class_name, 'k')  # 'k' (black) for unknown class

# Загрузка данных из CSV-файла с указанием разделителя точка с запятой
dataFrame = pd.read_csv("C:\\Users\\77711\\Desktop\\МОЕ\\laptops.csv", delimiter=';')

# Удаление символа "$" из столбца "price" и преобразование в числовой формат
dataFrame['price'] = pd.to_numeric(dataFrame['price'].replace('[\$,]', '', regex=True), errors='coerce')

# Удаление строк с пропущенными значениями
dataFrame = dataFrame.dropna(subset=['price', 'Processor'])

# Создание нового столбца 'Class', где '2' - средний, '1' - дорогой, '0' - бюджетный, 'premium' - премиум
dataFrame['Class'] = pd.cut(dataFrame['price'], bins=[-np.inf, 30000, 60000, np.inf], labels=[0, 2, 1], include_lowest=True).astype(int)

# Ensure 'Processor' column contains numeric values
dataFrame['Processor'] = pd.to_numeric(dataFrame['Processor'], errors='coerce')

# Update 'Class' based on the number of cores and price
dataFrame.loc[dataFrame['Processor'] <= 2, 'Class'] = 0  # Бюджетный класс
dataFrame.loc[(dataFrame['Processor'] > 2) & (dataFrame['Processor'] <= 4), 'Class'] = 0  # Бюджетный класс
dataFrame.loc[(dataFrame['Processor'] > 4) & (dataFrame['Processor'] <= 6), 'Class'] = 2  # Средний класс
dataFrame.loc[(dataFrame['Processor'] >= 2) & (dataFrame['Processor'] <= 4) & (dataFrame['price'] > 60000), 'Class'] = 5  # Присвоение 'premium' класса

# центры классов на вычисляемые значения на основе данных
centers = {
    'budget': np.array([dataFrame[dataFrame['Class'] == 0]['price'].mean(), dataFrame[dataFrame['Class'] == 0]['Processor'].mean()]),
    'medium': np.array([dataFrame[dataFrame['Class'] == 2]['price'].mean(), dataFrame[dataFrame['Class'] == 2]['Processor'].mean()]),
    'expensive': np.array([dataFrame[dataFrame['Class'] == 1]['price'].mean(), dataFrame[dataFrame['Class'] == 1]['Processor'].mean()]),
    'premium': np.array([dataFrame[dataFrame['Class'] == 5]['price'].mean(), dataFrame[dataFrame['Class'] == 5]['Processor'].mean()])
}


# Ввод цены и количества ядер
user_price = float(input("Введите цену: "))
user_cores = float(input("Введите количество ядер:"))

# Создание вектора для вычисления расстояний
user_vector = np.array([user_price, user_cores])

# Вычисление расстояний для каждой метрики
for metric in ['euclidean', 'chebyshev', 'manhattan']:
    distance = calculate_distance(user_vector, centers['budget'], metric)
    print(f"Расстояние до бюджетного класса (метрика {metric}): {distance}")

    distance = calculate_distance(user_vector, centers['medium'], metric)
    print(f"Расстояние до среднего класса (метрика {metric}): {distance}")

    distance = calculate_distance(user_vector, centers['expensive'], metric)
    print(f"Расстояние до дорогого класса (метрика {metric}): {distance}")

# Выбор колонок для оси X и оси Y
x_column = 'price'  # Название колонки для оси X
y_column = 'Processor'  # Название колонки для оси Y

# Разделение данных на классы
expensive_class = dataFrame[dataFrame['Class'] == 1]
medium_class = dataFrame[dataFrame['Class'] == 2]
budget_class = dataFrame[dataFrame['Class'] == 0]
premium_class = dataFrame[dataFrame['Class'] == 5]  # Выбор элементов 'premium' класса

# Построение графика с цветом, соответствующим классу пользователя
plt.scatter(expensive_class[x_column], expensive_class[y_column], marker='o', color='r', label='Дорогой класс')
plt.scatter(medium_class[x_column], medium_class[y_column], marker='o', color='g', label='Средний класс')
plt.scatter(budget_class[x_column], budget_class[y_column], marker='o', color='b', label='Бюджетный класс')
plt.scatter(premium_class[x_column], premium_class[y_column], marker='o', color='m', label='Премиум класс')  # Добавление премиум класса

# Визуализация пользовательского ввода с цветом в соответствии с классом
user_class = calculate_user_class(user_vector, centers)
user_color = get_class_color(user_class)
plt.scatter(user_price, user_cores, marker='x', color=user_color, label='Ваш выбор')

# Добавление подписей и заголовка
plt.xlabel('Цена')
plt.ylabel('Количество ядер')
plt.title('График данных из CSV файла')

# Ручная установка значений на оси X
tick_values = np.arange(10000, dataFrame[x_column].max() + 10000, 10000)
plt.xticks(tick_values, ['${}'.format(int(val)) for val in tick_values])

plt.xlim(10000, 100000)

# Добавление легенды
plt.legend()

# Отображение графика
plt.show()

# Вывод результата в консоль
class_names_r = {'budget': 'бюджетный', 'medium': 'средний', 'expensive': 'дорогой', 'premium': 'премиум'}
print(f"Ваш элемент относится к классу: {class_names_r.get(user_class, user_class)}")

