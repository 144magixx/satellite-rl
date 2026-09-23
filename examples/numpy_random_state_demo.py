import numpy as np

np.random.seed(None)
state = np.random.get_state()

# 第一个元素固定为 2147483648
print("第一个元素:", state[1][0])  # 总是 2147483648
print("前5个元素:", state[1][:5])  # 除第一个外，其他元素每次不同