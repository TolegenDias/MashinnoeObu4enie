import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

# Функция для вычисления медианы
def calculate_median(data):
    return np.median(data)

# Функция для вычисления отклонения
def calculate_deviation(data):
    mean = np.mean(data)
    deviation = np.sqrt(np.mean((data - mean)**2))
    return deviation

# Функция для вычисления симметрии
def calculate_symmetry(data):
    skewness = np.mean((data - np.mean(data))**3) / np.std(data)**3
    return skewness






# Функция для построения графика коэффициента корреляции
def plot_correlation(data1, data2):
    min_size = min(len(data1), len(data2))
    plt.scatter(data1[:min_size], data2[:min_size], marker='o')
    plt.xlabel('Процент заряда')
    plt.ylabel('Время для зарядки')
    plt.title('Зависимость времени для зарядки от процента заряда')
    plt.grid(True)
    plt.show()

# Установка random_state для воспроизводимости
np.random.seed(42)

# Генерация случайного количества элементов для data1
size_data1 = np.random.randint(1, 51)
size_data2 = size_data1  # Делаем размер data2 таким же, как и у data1

# Генерация данных для data1 (процент заряда от 1 до 100)
data1 = np.random.randint(1, 101, size=size_data1)

# Генерация данных для data2 с обратной зависимостью и ограничением от 1 до 10
data2 = np.maximum(1, 10 - 0.1 * data1 + np.random.normal(0, 1, size=size_data2))

# Вычисление и вывод результатов
median1 = calculate_median(data1)
median2 = calculate_median(data2)
print(f"Медиана Процент заряда: {median1}")
print(f"Медиана Время для зарядки: {median2}")

# Используем scipy.stats.mode для вычисления моды
mode_result1 = mode(data1)
mode_result2 = mode(data2)
print(f"Мода Процент заряда: {mode_result1.mode}")
print(f"Мода Время для зарядки: {mode_result2.mode}")

deviation1 = calculate_deviation(data1)
deviation2 = calculate_deviation(data2)
print(f"Отклонение Процент заряда: {deviation1}")
print(f"Отклонение Время для зарядки: {deviation2}")

symmetry1 = calculate_symmetry(data1)
symmetry2 = calculate_symmetry(data2)
print(f"Симметрия Процент заряда: {symmetry1}")
print(f"Симметрия Время для зарядки: {symmetry2}")

# Построение графика коэффициента корреляции
plot_correlation(data1, data2)
