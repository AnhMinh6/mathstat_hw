import random
import matplotlib.pyplot as plt
import numpy as np

"""Дискретное равномерное распределение I"""

def discrete_uniform_rand_variable(theta):
    s = random.random()
    k = 1
    distribution_func = 1 / theta
    while s > distribution_func:
        k += 1
        distribution_func = k / theta
    return k


theta = 79
def discrete_uniform_sample(n):
    sample = []
    for i in range(n):
        k = discrete_uniform_rand_variable(theta)
        sample.append(k)
    return sample

print("Выборка из дискретного равномерного распределения:", discrete_uniform_sample(10))

# Построение графика
sample_size = 1000
sample = discrete_uniform_sample(sample_size)
plt.hist(sample, bins=range(1, theta + 2), density=True, color='lightblue', 
         edgecolor='black', align='left', label='Смоделированная выборка')

k_values = np.arange(1, theta + 1)
distribution_law = []
for k in k_values:
    probability = 1 / theta
    distribution_law.append(probability)

plt.plot(k_values, distribution_law, 'o-', color='blue', linewidth=1, 
         markersize=2, label='Закон распределения')

plt.title(f'Дискретное равномерное распределение I\n(θ={theta})')
plt.xlabel('Значение случайной величины')
plt.ylabel('Вероятность')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Дополнительная проверка для большей наглядности
print(f"\nСтатистика для θ={theta}:")
print(f"Размер выборки: {sample_size}")
print(f"Минимальное значение: {min(sample)}")
print(f"Максимальное значение: {max(sample)}")
print(f"Среднее значение: {np.mean(sample):.2f}")
print(f"Теоретическое среднее: {(1 + theta) / 2:.2f}")

# Проверка равномерности
print(f"\nПроверка равномерности (первые 10 значений):")
for i in range(1, 11):
    count = sample.count(i)
    probability = count / sample_size
    print(f"X={i}: теоретическая вероятность={1/theta:.4f}, эмпирическая={probability:.4f}")