import json

"""4. Вычисление выборочных моментов"""

# Параметры распределений
theta_uniform = 79
m_erlang = 3
theta_erlang = 1/6

# Теоретические значения
theoretical_uniform_mean = (1 + theta_uniform) / 2
theoretical_uniform_variance = (theta_uniform**2 - 1) / 12

theoretical_erlang_mean = m_erlang / theta_erlang
theoretical_erlang_variance = m_erlang / (theta_erlang**2)

with open('sample_generation.json', 'r', encoding='utf-8') as f:
    result = json.load(f)

sample_mean = {}
sample_variance = {}
for distribution, data in result.items():
    sample_mean[distribution] = {}
    sample_variance[distribution] = {}
    for size, samples in data.items():
        for sample in samples:
            if int(size) not in sample_mean[distribution]:
                sample_mean[distribution][int(size)] = []
                sample_variance[distribution][int(size)] = []
            mean = sum(sample) / len(sample)
            sample_mean[distribution][int(size)].append(round(mean, 5))
            s = 0
            for i in sample:
                s += (i - mean) ** 2
            sample_variance[distribution][int(size)].append(round(s / len(sample), 5))

def print_distribution_stats(dist_name, sample_mean_dict, sample_variance_dict, theoretical_mean, theoretical_variance):
    print("=" * 80)
    print(f"{dist_name}")
    print("=" * 80)
    print(f"Теоретическое математическое ожидание: {theoretical_mean}")
    print(f"Теоретическая дисперсия: {theoretical_variance:.2f}")
    print()
    
    sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
    
    print("ВЫБОРОЧНЫЕ СРЕДНИЕ:")
    print("-" * 80)
    for size in sizes:
        if size in sample_mean_dict:
            means = sample_mean_dict[size]
            mean_str = ", ".join([f"{m:8.3f}" for m in means])
            avg_mean = sum(means) / len(means)
            print(f"n = {size:4d}: [{mean_str}] | Среднее: {avg_mean:7.3f}")
    
    print("\nВЫБОРОЧНЫЕ ДИСПЕРСИИ:")
    print("-" * 80)
    for size in sizes:
        if size in sample_variance_dict:
            variances = sample_variance_dict[size]
            var_str = ", ".join([f"{v:8.1f}" for v in variances])
            avg_var = sum(variances) / len(variances)
            print(f"n = {size:4d}: [{var_str}] | Среднее: {avg_var:7.1f}")
    print()

# Вывод результатов
print_distribution_stats(
    "ДИСКРЕТНОЕ РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ I",
    sample_mean["discrete_uniform"],
    sample_variance["discrete_uniform"],
    theoretical_uniform_mean,
    theoretical_uniform_variance
)

print_distribution_stats(
    "РАСПРЕДЕЛЕНИЕ ЭРЛАНГА",
    sample_mean["erlang"],
    sample_variance["erlang"],
    theoretical_erlang_mean,
    theoretical_erlang_variance
)