import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Загрузим данные
with open('sample_generation.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Функция распределения Эрланга (теоретическая CDF)
def erlang_cdf(x, m=3, theta=1/6):
    """Функция распределения Эрланга"""
    return stats.gamma.cdf(x, a=m, scale=1/theta)

# 3. Вычисление статистики D_n по описанному алгоритму
def compute_Dn(sample, theoretical_cdf, cdf_args=()):
    """
    Вычисляет статистику Колмогорова D_n по вариационному ряду.
    
    Параметры:
    - sample: список/массив значений выборки
    - theoretical_cdf: функция теоретического распределения F(x)
    - cdf_args: дополнительные аргументы для теоретической CDF
    
    Возвращает:
    - D_n: значение статистики Колмогорова
    - Dn_plus: D_n^+
    - Dn_minus: D_n^-
    """
    # Сортируем выборку (вариационный ряд)
    sorted_sample = np.sort(sample)
    n = len(sorted_sample)
    
    # Вычисляем F(x_(k))
    F_values = theoretical_cdf(sorted_sample, *cdf_args)
    
    # Вычисляем D_n^+
    k_values = np.arange(1, n + 1) / n
    Dn_plus = np.max(np.abs(k_values - F_values))
    
    # Вычисляем D_n^-
    k_minus_values = np.arange(0, n) / n  # (k-1)/n
    Dn_minus = np.max(np.abs(F_values - k_minus_values))
    
    # Итоговая статистика D_n
    D_n = max(Dn_plus, Dn_minus)
    
    return D_n, Dn_plus, Dn_minus

# 4. Статистика с поправкой Большева
def compute_Sn(D_n, n):
    """Вычисляет статистику с поправкой Большева"""
    return (6 * n * D_n + 1) / (6 * np.sqrt(n))

# 5. Функция для вычисления p-value по распределению Колмогорова
def kolmogorov_p_value(statistic, n, use_boshev=False):
    """
    Вычисляет p-value для статистики Колмогорова.
    
    Параметры:
    - statistic: значение статистики (D_n или S_n)
    - n: объем выборки
    - use_boshev: если True, statistic считается как S_n
    
    Возвращает:
    - p-value
    """
    if use_boshev:
        # Для статистики с поправкой Большева используем распределение Колмогорова
        stat = statistic
    else:
        # Для классической статистики преобразуем в sqrt(n)*D_n
        stat = np.sqrt(n) * statistic
    
    # Вычисляем p-value через функцию выживания распределения Колмогорова
    # P(sqrt(n)*D_n >= stat) = 1 - K(stat)
    # Используем scipy для точного вычисления
    p_value = 1 - stats.kstwo.cdf(stat, n)
    return p_value

# 6. Функция применения критерия Колмогорова
def perform_kolmogorov_test(sample, theoretical_cdf, cdf_args=(), alpha=0.05):
    """
    Применяет критерий Колмогорова к выборке.
    
    Возвращает словарь с результатами.
    """
    n = len(sample)
    
    # 1. Вычисляем классическую статистику D_n
    D_n, Dn_plus, Dn_minus = compute_Dn(sample, theoretical_cdf, cdf_args)
    
    # 2. Вычисляем статистику с поправкой Большева
    S_n = compute_Sn(D_n, n)
    
    # 3. Вычисляем p-value для обеих статистик
    p_value_classic = kolmogorov_p_value(D_n, n, use_boshev=False)
    p_value_boshev = kolmogorov_p_value(S_n, n, use_boshev=True)
    
    # 4. Принимаем решение для обеих статистик
    reject_classic = p_value_classic < alpha
    reject_boshev = p_value_boshev < alpha
    
    return {
        'n': n,
        'D_n': D_n,
        'Dn_plus': Dn_plus,
        'Dn_minus': Dn_minus,
        'sqrt_n_Dn': np.sqrt(n) * D_n,
        'S_n': S_n,
        'p_value_classic': p_value_classic,
        'p_value_boshev': p_value_boshev,
        'reject_classic': reject_classic,
        'reject_boshev': reject_boshev,
        'alpha': alpha
    }

# 7. Анализ выборок распределения Эрланга
print("=" * 80)
print("АНАЛИЗ КРИТЕРИЕМ КОЛМОГОРОВА (распределение Эрланга)")
print("=" * 80)

alpha = 0.05
results_erlang = {}

# Проходим по всем выборкам Эрланга
erlang_samples = data['erlang']

for size_str, samples_list in erlang_samples.items():
    size = int(size_str)
    print(f"\nРазмер выборки n = {size}:")
    print("-" * 50)
    
    results_erlang[size] = []
    
    for idx, sample in enumerate(samples_list):
        # Применяем критерий Колмогорова
        result = perform_kolmogorov_test(
            sample, 
            erlang_cdf, 
            cdf_args=(3, 1/6),  # m=3, theta=1/6
            alpha=alpha
        )
        
        results_erlang[size].append(result)
        
        # Выводим краткую информацию
        print(f"  Выборка {idx + 1}:")
        print(f"    D_n = {result['D_n']:.6f}")
        print(f"    √n·D_n = {result['sqrt_n_Dn']:.6f}")
        print(f"    S_n (Большев) = {result['S_n']:.6f}")
        print(f"    p-value (классический) = {result['p_value_classic']:.4f}")
        print(f"    p-value (с поправкой) = {result['p_value_boshev']:.4f}")
        print(f"    Решение (α={alpha}): ", end="")
        
        if result['reject_classic']:
            print(f"Классический: ОТКЛОНЯЕМ H₀", end="")
        else:
            print(f"Классический: НЕ ОТКЛОНЯЕМ H₀", end="")
            
        if result['reject_boshev'] != result['reject_classic']:
            print(f", С поправкой: {'ОТКЛОНЯЕМ' if result['reject_boshev'] else 'НЕ ОТКЛОНЯЕМ'} H₀")
        else:
            print()

# 8. Сводная статистика по размерам выборок
print("\n" + "=" * 80)
print("СВОДНАЯ СТАТИСТИКА ПО РАЗМЕРАМ ВЫБОРОК")
print("=" * 80)

for size, test_results in results_erlang.items():
    # Средние значения
    avg_Dn = np.mean([r['D_n'] for r in test_results])
    avg_p_classic = np.mean([r['p_value_classic'] for r in test_results])
    avg_p_boshev = np.mean([r['p_value_boshev'] for r in test_results])
    
    # Доля отвергнутых гипотез
    reject_rate_classic = np.mean([r['reject_classic'] for r in test_results])
    reject_rate_boshev = np.mean([r['reject_boshev'] for r in test_results])
    
    print(f"\nn = {size}:")
    print(f"  Среднее D_n: {avg_Dn:.6f}")
    print(f"  Средний p-value (классический): {avg_p_classic:.4f}")
    print(f"  Средний p-value (с поправкой): {avg_p_boshev:.4f}")
    print(f"  Доля отвержений H₀ (классический): {reject_rate_classic:.2f}")
    print(f"  Доля отвержений H₀ (с поправкой): {reject_rate_boshev:.2f}")
    print(f"  Ожидаемая доля отвержений при верной H₀: {alpha:.2f}")

# 9. Визуализация зависимости D_n от объема выборки
print("\n" + "=" * 80)
print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 80)

# Подготовка данных для графиков
sizes = sorted(results_erlang.keys())
mean_Dn_values = []
mean_p_values_classic = []
mean_p_values_boshev = []

for size in sizes:
    test_results = results_erlang[size]
    mean_Dn_values.append(np.mean([r['D_n'] for r in test_results]))
    mean_p_values_classic.append(np.mean([r['p_value_classic'] for r in test_results]))
    mean_p_values_boshev.append(np.mean([r['p_value_boshev'] for r in test_results]))

# Создаем графики
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. График зависимости D_n от n
ax1 = axes[0, 0]
ax1.plot(sizes, mean_Dn_values, 'bo-', linewidth=2, markersize=6)
ax1.set_xlabel('Объем выборки (n)', fontsize=12)
ax1.set_ylabel('Среднее значение D_n', fontsize=12)
ax1.set_title('Зависимость статистики D_n от объема выборки', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# 2. График зависимости √n·D_n от n
ax2 = axes[0, 1]
sqrt_n_Dn_values = [np.sqrt(size) * Dn for size, Dn in zip(sizes, mean_Dn_values)]
ax2.plot(sizes, sqrt_n_Dn_values, 'ro-', linewidth=2, markersize=6)
ax2.set_xlabel('Объем выборки (n)', fontsize=12)
ax2.set_ylabel('Среднее значение √n·D_n', fontsize=12)
ax2.set_title('Зависимость √n·D_n от объема выборки', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# 3. График p-value (классический)
ax3 = axes[1, 0]
ax3.plot(sizes, mean_p_values_classic, 'go-', linewidth=2, markersize=6)
ax3.axhline(y=alpha, color='r', linestyle='--', alpha=0.7, label=f'α={alpha}')
ax3.set_xlabel('Объем выборки (n)', fontsize=12)
ax3.set_ylabel('Средний p-value', fontsize=12)
ax3.set_title('Средний p-value (классический критерий)', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xscale('log')

# 4. График p-value (с поправкой Большева)
ax4 = axes[1, 1]
ax4.plot(sizes, mean_p_values_boshev, 'mo-', linewidth=2, markersize=6)
ax4.axhline(y=alpha, color='r', linestyle='--', alpha=0.7, label=f'α={alpha}')
ax4.set_xlabel('Объем выборки (n)', fontsize=12)
ax4.set_ylabel('Средний p-value', fontsize=12)
ax4.set_title('Средний p-value (с поправкой Большева)', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xscale('log')

plt.tight_layout()
plt.savefig('kolmogorov_test_results.png', dpi=150, bbox_inches='tight')
plt.show()

# 10. Сохранение полных результатов в файл
output_data = {
    'parameters': {
        'erlang': {'m': 3, 'theta': 1/6},
        'alpha': alpha
    },
    'results': results_erlang
}

with open('kolmogorov_test_results.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4, default=str)

print("\nРезультаты сохранены в файлы:")
print("1. kolmogorov_test_results.json - полные численные результаты")
print("2. kolmogorov_test_results.png - графики")

# 11. Анализ для дискретного равномерного распределения (только для сравнения)
print("\n" + "=" * 80)
print("ПРИМЕЧАНИЕ ПО ДИСКРЕТНОМУ РАВНОМЕРНОМУ РАСПРЕДЕЛЕНИЮ")
print("=" * 80)
print("Критерий Колмогорова предназначен для непрерывных распределений.")
print("Для дискретного равномерного распределения его применение некорректно.")
print("Для дискретных распределений рекомендуется использовать критерий хи-квадрат.")