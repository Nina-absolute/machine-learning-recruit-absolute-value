from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(
    n_samples=2000,
    noise=0.2,
    random_state=20
)
# 生成数据集

plt.scatter(X[:, 0], X[:, 1], c=y)
# 用 matplotlib 画出散点图（scatter plot）
# c 是 scatter() 的一个参数，用于设置点的颜色

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("make_moons Dataset")
# 给图表加上图例、标题

plt.show()