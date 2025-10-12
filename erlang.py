import random
import math
import matplotlib.pyplot as plt
import numpy as np

def erlang_rand_variable(m, theta):
    """
    Моделирование случайной величины с распределением Эрланга
    """
    X = 0
    for i in range(m):
        u = random.random()
        y = -math.log(u) / theta
        X += y
    return X

def erlang_sample(n, m, theta):
    sample = []
    for i in range(n):
        x = erlang_rand_variable(m, theta)
        sample.append(x)
    return sample

m = 3
theta = 1/6

sample_size = 1000
sample = erlang_sample(sample_size, m, theta)

plt.figure(figsize=(10, 6))
plt.hist(sample, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black', label='Смоделированная выборка')

x_values = np.linspace(0, max(sample), 1000)
pdf_values = (theta**m / math.factorial(m-1)) * x_values**(m-1) * np.exp(-theta * x_values)
plt.plot(x_values, pdf_values, 'r-', linewidth=2, label='Теоретическая плотность')

plt.title(f'Распределение Эрланга\n(m={m}, θ={theta})')
plt.xlabel('Значение случайной величины')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Теоретическое среднее: {m/theta}")
print(f"Эмпирическое среднее: {np.mean(sample):.2f}")
print(f"Теоретическая дисперсия: {m/theta**2}")
print(f"Эмпирическая дисперсия: {np.var(sample):.2f}")