from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from pathlib import Path

X, y = make_moons(
    n_samples=2000,
    noise=0.2,
    random_state=30
)
# 生成数据集

output_dir = Path(__file__).parent.parent / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)
# 确保 outputs 目录存在

plt.scatter(X[:, 0], X[:, 1], c=y)
# 用 matplotlib 画出散点图（scatter plot）
# c 是 scatter() 的一个参数，用于设置点的颜色

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("make_moons Dataset")
# 给图表加上图例、标题

plt.savefig(output_dir / 'raw_data.png', dpi=200)
plt.show()