# 程序运行命令 + 结果：
### 1. 安装 Miniconda
![](./../screenshot/install-miniconda.png)

### 2. 创建独立的 Python 虚拟环境
```bash
conda create --name ml_task python=3.12 -y
conda activate ml_task
```
![](./../screenshot/conda-env-list.png)

### 3. 检验 GPU & 安装 CUDA
![](./../screenshot/nvidia-smi.png)

安装 CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 4. 参考 PyTorch 官方安装页安装 PyTorch，并安装 numpy、matplotlib 和 scikit-learn
安装 PyTorch
![](./../screenshot/pytorch-version.png)
![](./../screenshot/pytorch-download1.png)
![](./../screenshot/pytorch-download2.png)

安装 numpy、matplotlib 和 scikit-learn
```bash
pip install numpy matplotlib scikit-learn pandas
```
![](./../screenshot/scikit-learn-install.png)
![](./../screenshot/matplotlib-install.png)

### 5. 输出 Python 和 PyTorch 的版本
![](./../screenshot/Python-version.png)
![](./../screenshot/torch-version.png)

### 6. Python 热身代码 & 张量计算

![](./../screenshot/result-screenshot.png)

### 7. 验证 CUDA 可用 & 使用 GPU
![](./../screenshot/available-cuda.png)
![](./../screenshot/GPUresult-screenshot.png)
