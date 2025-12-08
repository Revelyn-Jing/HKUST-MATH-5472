不使用 Conformal Prediction 会怎样？
====================================

本仓库配合课程 MATH5472 的课程论文，探索在缺少 conformal prediction 时，不同不确定性量化方法（贝叶斯推断、Bootstrap、分位数回归等）在覆盖率和区间长度上的表现。代码主要通过合成数据和 UCI Energy Efficiency 数据集（`data/ENB2012_data.xlsx`）复现与对比实验，论文与图表存放在 `report/` 目录。

目录结构
--------
- `code/Bayesian/`：贝叶斯方法相关 notebook  
  - `BayesianRidge.ipynb`：在线性合成数据上对比完整贝叶斯 Ridge 与 split conformal，在不同置信度下输出覆盖率和区间宽度（生成 `report/figure/FullBayes_vs_SplitCP.png`）。
  - `EmpiricalBayesianRidge.ipynb`：用经验贝叶斯估计噪声方差后再做 Ridge，对比 split conformal（生成 `report/figure/EmpiricalBayes_vs_SplitCP.png`）。
  - `BNN.ipynb`：在 ENB2012 数据集上训练变分 BNN，通过 MC 采样估计预测均值/方差，并与 split conformal 结果对比（生成 `report/figure/BNN_vs_SplitCP.png`）。
- `code/Boostrap/`：Bootstrap 与 conformal 对比  
  - `bootstrap_efficient.ipynb`：在线性+非线性合成数据上，比较残差 Bootstrap 预测区间与 split conformal 的覆盖率、区间长度。
  - `bootstrap_SLOW.ipynb`：在 ENB2012 数据集上，使用 MLP 进行多次自举训练，评估 Bootstrap 预测区间与 split conformal 的精度与耗时。
- `code/Quantile_Regression/main.ipynb`：在 ENB2012 数据集上，训练 MLP 分位数回归（上下分位数）并与 split conformal 直接对比覆盖率和区间长度。
- `data/ENB2012_data.xlsx`：UCI Energy Efficiency 数据集（前 8 列为特征，`Y1` 为 Heating Load）。如需替换数据，保持列顺序一致。
- `report/`：论文 (`essay.tex` / `essay.pdf`) 及参考文献，`report/figure/` 存放生成的对比图。

环境准备
--------
- Python 3.9+，建议使用虚拟环境。
- 依赖：`numpy`、`pandas`、`scikit-learn`、`torch`、`matplotlib`、`scipy`、`openpyxl`、`jupyter`。示例安装：
  ```bash
  pip install numpy pandas scikit-learn torch matplotlib scipy openpyxl jupyter
  ```
- GPU 可选，用于加速 `BNN.ipynb` 和部分 MLP 实验。

运行方式
--------
1. 在项目根目录启动 Jupyter：`jupyter notebook` 或 `jupyter lab`。
2. 按需打开对应 notebook，依次运行所有单元：
   - **贝叶斯 Ridge 系列**：直接运行即可生成覆盖率/区间宽度曲线并保存图片到 `report/figure/`。
   - **BNN**：读取 `data/ENB2012_data.xlsx`，训练变分 BNN 后通过蒙特卡洛采样估计预测分布，与 split conformal 比较（运行时间受采样次数和硬件影响）。
   - **Bootstrap（efficient/slow）**：`run_experiment()` 或主流程会打印覆盖率与区间平均长度；`bootstrap_SLOW` 还会输出训练/推理耗时以展示开销。
   - **Quantile Regression**：训练上下分位数 MLP，计算覆盖率与区间长度，并与 split conformal 的结果并列打印。

结果与报告
---------
- 关键对比图：`report/figure/BNN_vs_SplitCP.png`、`EmpiricalBayes_vs_SplitCP.png`、`FullBayes_vs_SplitCP.png`。
- 论文全文：`report/essay.pdf`（LaTeX 源码在同目录）。

说明
----
- notebook 默认使用仓库自带的数据和相对路径；若调整数据路径或实验配置，请同步修改 notebook 中的读写路径和超参数。
- 若需命令行形式复现实验，可将 notebook 中的核心函数抽取为脚本，保持相同的数据拆分与随机种子。
