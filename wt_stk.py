import random

# 初始化变量
N = 1000000  # 总选择次数
start_count = 10000  # 开始统计的位置
range_num = 16  # 数字范围

# 随机选择1000亿次
choices = [random.randint(1, range_num) for _ in range(N)]

# 统计和前1到16期相同的次数
count_same = [0] * 16

for i in range(start_count, N):
    for j in range(1, 17):
        if choices[i] == choices[i - j]:
            count_same[j - 1] += 1

# 找出次数最多的期数
max_count = max(count_same)
most_frequent_period = count_same.index(max_count) + 1

print(f'前 {most_frequent_period} 期的次数最多，总次数为 {max_count}')

import tushare

print(tushare.__version__)