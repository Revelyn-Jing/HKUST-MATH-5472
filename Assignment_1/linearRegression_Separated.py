import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# 处理路径兼容性（支持脚本和 Jupyter）
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path(".").resolve()

file_path = current_dir / "data" / "GaltonFamilies.csv"

# 读取数据
GaltonFamilies = pd.read_csv(file_path)

print("Data shape:", GaltonFamilies.shape)
print("Columns:", GaltonFamilies.columns.tolist())


GaltonFamilies = pd.get_dummies(GaltonFamilies, columns=['gender'], prefix='gender', drop_first=True)
features = ['father','mother','midparentHeight', 'gender_male']  
target = 'childHeight'
# 也可以尝试只用 midparentHeight（最经典）
# features = ['midparentHeight']

X = GaltonFamilies[features]
y = GaltonFamilies[target]

# 设置随机种子，确保可复现
np.random.seed(10)
nrep = 100
MSE_gender = np.zeros(nrep)
MSE_whole = np.zeros(nrep)


# 不分性别进行建模
for rep in range(nrep):
    # 划分训练集和测试集（80% 训练，其余测试）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rep)
    
    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 记录 MSE
    MSE_whole[rep] = mean_squared_error(y_test, y_pred)

# 输出平均 MSE
mean_mse_whole = MSE_whole.mean()
std_mse_whole = MSE_whole.std()
print(f"单一线性回归模型 Mean MSE = {mean_mse_whole:.4f} ± {std_mse_whole:.4f}")


# 绘制 MSE_whole 分布
plt.figure(figsize=(8, 6))
sns.histplot(MSE_whole, kde=True, bins=20)
plt.title('Distribution of MSE (Single Linear Regression Model)')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frequency')
plt.axvline(mean_mse_whole, color='red', linestyle='--', label=f'Mean MSE = {mean_mse_whole:.4f}')
plt.legend()
plt.show()




# 分别对男性和女性建立不同的线性模型
male_data = GaltonFamilies[GaltonFamilies['gender_male'] == 1]
female_data = GaltonFamilies[GaltonFamilies['gender_male'] == 0]

features = ['father','mother','midparentHeight']  # 性别作为分类变量已经分开了

for rep in range(nrep):
    # 分别划分男女数据
    # 正确的训练集索引采样方式
    n_male = len(male_data)
    n_female = len(female_data)

    n_male_train = int(n_male * 0.8)
    n_female_train = int(n_female * 0.8)

    male_train_idx = np.random.choice(n_male, n_male_train, replace=False)
    female_train_idx = np.random.choice(n_female, n_female_train, replace=False)

    # 获取训练集和测试集
    male_train = male_data.iloc[male_train_idx]
    male_test = male_data.drop(male_data.index[male_train_idx])

    female_train = female_data.iloc[female_train_idx]
    female_test = female_data.drop(female_data.index[female_train_idx])
    
    male_train = male_data.iloc[male_train_idx]
    male_test = male_data.drop(male_data.index[male_train_idx])
    
    female_train = female_data.iloc[female_train_idx]
    female_test = female_data.drop(female_data.index[female_train_idx])
    
    # 方法1：分别训练男女模型
    model_male = LinearRegression()
    model_male.fit(male_train[features], male_train[target])
    
    model_female = LinearRegression()
    model_female.fit(female_train[features], female_train[target])
    
    # 分别预测
    male_pred = model_male.predict(male_test[features])
    female_pred = model_female.predict(female_test[features])
    
    # 合并结果计算MSE
    all_true = pd.concat([male_test[target], female_test[target]])
    all_pred = np.concatenate([male_pred, female_pred])
    
    MSE_gender[rep] = mean_squared_error(all_true, all_pred)


mean_mse_gender = MSE_gender.mean()
std_mse_gender = MSE_gender.std()
print(f"分性别线性回归模型 Mean MSE_gender = {mean_mse_gender:.4f} ± {std_mse_gender:.4f}")

# 绘制 MSE 分布
plt.figure(figsize=(8, 6))
sns.histplot(MSE_gender, kde=True, bins=20)
plt.title('Distribution of MSE (Separated Linear Regression Models)')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frequency')
plt.axvline(mean_mse_gender, color='red', linestyle='--', label=f'Mean MSE = {mean_mse_gender:.4f}')
plt.legend()
plt.show()


# 绘制箱形图比较
plt.figure(figsize=(10, 6))
box_data = [MSE_whole, MSE_gender]
box_labels = ['Single Model', 'Gender-Separated Model']


box = plt.boxplot(box_data, labels=box_labels, patch_artist=True,
                  medianprops=dict(color='red', linewidth=2),
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  whiskerprops=dict(color='blue'),
                  capprops=dict(color='blue'),
                  flierprops=dict(markerfacecolor='red', marker='o', markersize=5, linestyle='none'))

# 添加标题和标签
plt.title('Comparison of MSE Distributions Across 100 Repetitions', fontsize=14, fontweight='bold')
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在图上标注均值
means = [mean_mse_whole, mean_mse_gender]
for i, mean in enumerate(means):
    plt.scatter(i+1, mean, color='green', s=100, zorder=5, label='Mean' if i == 0 else "", edgecolor='black')

plt.legend()

# 保存图像（可选）
fig_dir = current_dir / "fig"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "MSE_comparison_boxplot.png", dpi=300, bbox_inches='tight')

plt.show()