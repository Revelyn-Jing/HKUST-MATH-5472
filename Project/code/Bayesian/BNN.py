import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# 确保在 CPU 上运行，除非环境明确支持 GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --- 实验参数 ---
HIDDEN_SIZE = 64 # 使用较大的隐藏层，模拟一个Well-Specified Model
PRIOR_SIGMA = 1.0 # BNN Prior N(0, 1)
N_MC = 100       # BNN Monte Carlo 采样次数

# ----------------------------
# 1. 数据生成（线性模型）
# ----------------------------
def generate_data(n_train=100, n_test=100, d=5, seed=0):
    """
    线性模型数据：y = X @ w_true + eps, eps ~ N(0,1)
    """
    np.random.seed(seed)
    X_train = np.random.uniform(-10, 10, size=(n_train, d)).astype(np.float32)
    X_test  = np.random.uniform(-10, 10, size=(n_test,  d)).astype(np.float32)
    
    # 确保权重也是 float32
    w_true  = np.random.randn(d).astype(np.float32)
    
    y_train = X_train @ w_true + np.random.randn(n_train).astype(np.float32)
    y_test  = X_test  @ w_true + np.random.randn(n_test).astype(np.float32)

    # 再保险：强制 cast 一次
    y_train = y_train.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    return X_train, y_train, X_test, y_test, w_true
def generate_data_misspec(n_train=100, n_test=100, d=5, seed=1):
    """
    Case B：模型错配
    真模型：非线性 + heavy-tail 噪声（t 分布）
    但仍然用 BNN(高斯似然) / NN + CP 去拟合
    """
    np.random.seed(seed)
    X_train = np.random.uniform(-3, 3, size=(n_train, d)).astype(np.float32)
    X_test  = np.random.uniform(-3, 3, size=(n_test,  d)).astype(np.float32)

    # 非线性真函数，只用前几个维度
    def f(x):
        # x: (n, d)
        return (
            np.sin(x[:, 0])
            + 0.5 * x[:, 1] ** 2
            - 0.3 * np.exp(0.3 * x[:, 2])
        ).astype(np.float32)

    # heavy-tail 噪声：t(df=3)，再缩放一下
    df = 3
    eps_train = (stats.t.rvs(df, size=n_train) * 1.0).astype(np.float32)
    eps_test  = (stats.t.rvs(df, size=n_test) * 1.0).astype(np.float32)

    y_train = f(X_train) + eps_train
    y_test  = f(X_test) + eps_test

    # 这里只是为了接口一致，w_true 没什么意义
    w_true = np.zeros(d, dtype=np.float32)
    return X_train, y_train, X_test, y_test, w_true


# --- BNN (VI) 模块 ---

class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma):
        super().__init__()
        self.in_features = in_features
        self.prior_sigma = torch.tensor(prior_sigma, dtype=torch.float32).to(DEVICE)
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(-5)
        self.bias_rho.data.fill_(-5)

    def forward(self, input):
        weight_sigma = torch.exp(self.weight_rho)
        bias_sigma = torch.exp(self.bias_rho)

        epsilon_w = torch.randn_like(self.weight_mu)
        epsilon_b = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_sigma * epsilon_w
        bias = self.bias_mu + bias_sigma * epsilon_b
        
        return F.linear(input, weight, bias)

    def kl_divergence(self):
        prior_sigma2 = self.prior_sigma**2
        
        weight_sigma2 = torch.exp(self.weight_rho * 2)
        kl_w = 0.5 * torch.sum(
            2 * torch.log(self.prior_sigma) - 2 * self.weight_rho + 
            (weight_sigma2 + self.weight_mu**2) / prior_sigma2 - 1
        )
        
        bias_sigma2 = torch.exp(self.bias_rho * 2)
        kl_b = 0.5 * torch.sum(
            2 * torch.log(self.prior_sigma) - 2 * self.bias_rho + 
            (bias_sigma2 + self.bias_mu**2) / prior_sigma2 - 1
        )
        return kl_w + kl_b

class BNN(nn.Module):
    def __init__(self, d_in, hidden, prior_sigma):
        super().__init__()
        self.fc1 = BayesLinear(d_in, hidden, prior_sigma)
        self.fc2 = BayesLinear(hidden, hidden, prior_sigma)
        self.fc3 = BayesLinear(hidden, 1, prior_sigma)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def kl_divergence(self):
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.fc3.kl_divergence()

def train_bnn(X, y, hidden=HIDDEN_SIZE, prior_sigma=PRIOR_SIGMA,
              lr=1e-3, epochs=300, batch_size=32, seed=0):
    torch.manual_seed(seed)
    
    model = BNN(d_in=X.shape[1], hidden=hidden, prior_sigma=prior_sigma).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum') 

    X_tensor = torch.from_numpy(X).to(DEVICE)
    y_tensor = torch.from_numpy(y).view(-1, 1).to(DEVICE)
    n = X.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_tensor[idx]
            yb = y_tensor[idx]

            model.train()
            optimizer.zero_grad()
            
            preds = model(xb)
            log_likelihood = criterion(preds, yb)
            
            kl_loss = model.kl_divergence() / (n / batch_size) 
            
            loss = log_likelihood + kl_loss
            
            loss.backward()
            optimizer.step()
    return model

def bnn_vi_intervals(model, X_test, y_test, alpha=0.1, n_mc=N_MC):
    model.eval()
    X_tensor = torch.from_numpy(X_test).to(DEVICE)

    preds_mc = []
    with torch.no_grad():
        for _ in range(n_mc):
            preds = model(X_tensor).cpu().numpy().ravel()
            preds_mc.append(preds)
            
    preds_mc = np.stack(preds_mc, axis=0)

    mean_pred = preds_mc.mean(axis=0)
    std_pred  = preds_mc.std(axis=0, ddof=1) + 1e-6

    # Gaussian Credible Interval
    z = stats.norm.ppf(1 - alpha / 2)
    lower = mean_pred - z * std_pred
    upper = mean_pred + z * std_pred

    # Coverage calculation (matching original structure)
    covered = (y_test >= lower) & (y_test <= upper)
    coverage = covered.mean()
    mean_width = (upper - lower).mean()
    return coverage, mean_width

# --- Split CP (NN) 模块 ---

class MLP(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp(X, y, hidden=HIDDEN_SIZE, lr=1e-3, epochs=300, batch_size=32, weight_decay=1e-4, seed=0):
    torch.manual_seed(seed)
    model = MLP(d_in=X.shape[1], hidden=hidden).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_tensor = torch.from_numpy(X).to(DEVICE)
    y_tensor = torch.from_numpy(y).view(-1, 1).to(DEVICE)

    n = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_tensor[idx]
            yb = y_tensor[idx]

            model.train()
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    return model

def split_conformal_nn_intervals(X_train, y_train, X_test, y_test, hidden=HIDDEN_SIZE, alpha=0.1, seed=0):
    np.random.seed(seed)
    n_train = len(y_train)
    # Randomly split: 50% proper training, 50% calibration
    idx = np.random.permutation(n_train)
    n_cal = n_train // 2
    cal_idx = idx[:n_cal]
    train_idx = idx[n_cal:]

    X_proper, y_proper = X_train[train_idx], y_train[train_idx]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]

    # Train deterministic NN on proper training set
    model = train_mlp(X_proper, y_proper, hidden=hidden, seed=seed)

    # Nonconformity scores on calibration set
    model.eval()
    with torch.no_grad():
        X_cal_tensor = torch.from_numpy(X_cal).to(DEVICE)
        preds_cal = model(X_cal_tensor).cpu().numpy().ravel()
    
    scores = np.abs(y_cal - preds_cal)

    # Quantile with finite-sample correction
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(scores, q_level, method='higher')

    # Predict intervals for the FIXED test set
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).to(DEVICE)
        y_pred_test = model(X_test_tensor).cpu().numpy().ravel()
        
    lower = y_pred_test - q_hat
    upper = y_pred_test + q_hat

    # Coverage on the fixed test set
    covered = (y_test >= lower) & (y_test <= upper)
    coverage = covered.mean()
    mean_width = (upper - lower).mean()
    return coverage, mean_width

# ----------------------------
# 7. 主实验循环（BNN vs Split CP）
# ----------------------------
def run_bnn_vs_cp_experiment():
    
    alphas = np.linspace(0.01, 0.99, 20)  # significance levels
    # BNN 的复杂度由 hidden_size 和 prior_sigma 控制，这里使用单个配置
    config_labels = [f"H={HIDDEN_SIZE}"] 
    n_rep = 10  # repetitions

    # 生成固定的训练集和测试集
    np.random.seed(42)  
    X_train, y_train, X_test, y_test, w_true = generate_data(n_train=100, n_test=100, d=5, seed=42)

    # 容器
    cov_bayes = {label: [] for label in config_labels}
    width_bayes = {label: [] for label in config_labels}
    cov_split = {label: [] for label in config_labels}
    width_split = {label: [] for label in config_labels}

    label = config_labels[0]
    print(f"Running Configuration: {label} (BNN Prior={PRIOR_SIGMA})")

    for alpha in alphas:
        covs_b, widths_b = [], []
        covs_s, widths_s = [], []
        
        for rep in range(n_rep):
            # Bayesian Neural Network (使用完整的训练集)
            cov_b, w_b = bnn_vi_intervals(
                train_bnn(X_train, y_train, hidden=HIDDEN_SIZE, prior_sigma=PRIOR_SIGMA, seed=rep),
                X_test, y_test, alpha=alpha, n_mc=N_MC
            )
            covs_b.append(cov_b); widths_b.append(w_b)
            
            # Split CP (NN): 使用固定的训练集和测试集，只改变内部的分割种子
            cov_s, w_s = split_conformal_nn_intervals(
                X_train, y_train, X_test, y_test, hidden=HIDDEN_SIZE, alpha=alpha, seed=rep
            )
            covs_s.append(cov_s); widths_s.append(w_s)
            
        cov_bayes[label].append(np.mean(covs_b))
        width_bayes[label].append(np.mean(widths_b))
        cov_split[label].append(np.mean(covs_s))
        width_split[label].append(np.mean(widths_s))
        
        print(f"  alpha={alpha:.3f} (Nominal={1-alpha:.3f}): BNN Cov={cov_bayes[label][-1]:.3f}, CP Cov={cov_split[label][-1]:.3f}")



# ----------------------------
# 8. 绘图（四象限结构）
# ----------------------------
    conf_levels = 1 - alphas

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: BNN (Coverage Validity)
    ax = axes[0, 0]
    ax.plot(conf_levels * 100, (1 - np.array(cov_bayes[label])) * 100, label=f'BNN (VI), {label}')
    ax.plot([0, 100], [100, 0], 'k--', label='Ideal')
    ax.set_xlabel('Nominal Confidence ($1-\alpha$) (%)')
    ax.set_ylabel('Empirical $\\alpha$ ($\%$ Outside Interval)')
    ax.set_title('Bayesian NN (VI) - Coverage Validity')
    ax.legend()
    ax.grid(True)
    

    # Top-right: BNN (Width)
    ax = axes[0, 1]
    ax.plot(conf_levels * 100, width_bayes[label], label=f'BNN (VI), {label}')
    ax.set_xlabel('Nominal Confidence ($1-\alpha$) (%)')
    ax.set_ylabel('Mean Interval Width')
    ax.set_title('Bayesian NN (VI) - Width')
    ax.legend()
    ax.grid(True)

    # Bottom-left: Split CP (Coverage Validity) - KEY DIFFERENCE
    ax = axes[1, 0]
    ax.plot(conf_levels * 100, (1 - np.array(cov_split[label])) * 100, label=f'Split CP (NN), {label}')
    ax.plot([0, 100], [100, 0], 'k--', label='Ideal')
    ax.set_xlabel('Nominal Confidence ($1-\alpha$) (%)')
    ax.set_ylabel('Empirical $\\alpha$ ($\%$ Outside Interval)')
    ax.set_title('Split Conformal (NN) - Coverage Validity')
    ax.legend()
    ax.grid(True)
    

    # Bottom-right: Split CP (Width)
    ax = axes[1, 1]
    ax.plot(conf_levels * 100, width_split[label], label=f'Split CP (NN), {label}')
    ax.set_xlabel('Nominal Confidence ($1-\alpha$) (%)')
    ax.set_ylabel('Mean Interval Width')
    ax.set_title('Split Conformal (NN) - Width')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("BNN_VI_vs_Split_CP_NN_comparison.png", dpi=150)
    plt.show()


def run_bnn_vs_cp_experiment_misspec():
    """
    Case B：模型错配实验
    真模型：generate_data_misspec（非线性 + heavy-tail）
    模型：BNN (Gaussian likelihood) + Split CP (NN)
    """
    alphas = np.linspace(0.01, 0.99, 20)
    config_labels = [f"H={HIDDEN_SIZE}"]
    n_rep = 10

    # 固定一套“错配”的训练/测试集
    np.random.seed(123)
    X_train, y_train, X_test, y_test, w_true = generate_data_misspec(
        n_train=100, n_test=100, d=5, seed=123
    )

    cov_bayes = {label: [] for label in config_labels}
    width_bayes = {label: [] for label in config_labels}
    cov_split = {label: [] for label in config_labels}
    width_split = {label: [] for label in config_labels}

    label = config_labels[0]
    print(f"Running MIS-SPEC Configuration: {label} (BNN Prior={PRIOR_SIGMA})")

    for alpha in alphas:
        covs_b, widths_b = [], []
        covs_s, widths_s = [], []

        for rep in range(n_rep):
            # BNN（错配：仍假设高斯线性噪声）
            bnn_model = train_bnn(
                X_train, y_train,
                hidden=HIDDEN_SIZE,
                prior_sigma=PRIOR_SIGMA,
                seed=rep
            )
            cov_b, w_b = bnn_vi_intervals(
                bnn_model, X_test, y_test,
                alpha=alpha, n_mc=N_MC
            )
            covs_b.append(cov_b); widths_b.append(w_b)

            # Split CP (NN)：仍然用同一个 NN 结构
            cov_s, w_s = split_conformal_nn_intervals(
                X_train, y_train, X_test, y_test,
                hidden=HIDDEN_SIZE, alpha=alpha, seed=rep
            )
            covs_s.append(cov_s); widths_s.append(w_s)

        cov_bayes[label].append(np.mean(covs_b))
        width_bayes[label].append(np.mean(widths_b))
        cov_split[label].append(np.mean(covs_s))
        width_split[label].append(np.mean(widths_s))

        print(f"[MIS-SPEC] alpha={alpha:.3f} (Nominal={1-alpha:.3f}): "
              f"BNN Cov={cov_bayes[label][-1]:.3f}, CP Cov={cov_split[label][-1]:.3f}")

    # -------- 画图 --------
    conf_levels = 1 - alphas
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: BNN Coverage
    ax = axes[0, 0]
    ax.plot(conf_levels * 100,
            (1 - np.array(cov_bayes[label])) * 100,
            label=f'BNN (VI), {label}')
    ax.plot([0, 100], [100, 0], 'k--', label='Ideal')
    ax.set_xlabel('Nominal Confidence ($1-\\alpha$) (%)')
    ax.set_ylabel('Empirical $\\alpha$ (% Outside Interval)')
    ax.set_title('Bayesian NN (VI) - Coverage (Misspecified)')
    ax.legend(); ax.grid(True)

    # Top-right: BNN Width
    ax = axes[0, 1]
    ax.plot(conf_levels * 100, width_bayes[label],
            label=f'BNN (VI), {label}')
    ax.set_xlabel('Nominal Confidence ($1-\\alpha$) (%)')
    ax.set_ylabel('Mean Interval Width')
    ax.set_title('Bayesian NN (VI) - Width (Misspecified)')
    ax.legend(); ax.grid(True)

    # Bottom-left: Split CP Coverage
    ax = axes[1, 0]
    ax.plot(conf_levels * 100,
            (1 - np.array(cov_split[label])) * 100,
            label=f'Split CP (NN), {label}')
    ax.plot([0, 100], [100, 0], 'k--', label='Ideal')
    ax.set_xlabel('Nominal Confidence ($1-\\alpha$) (%)')
    ax.set_ylabel('Empirical $\\alpha$ (% Outside Interval)')
    ax.set_title('Split Conformal (NN) - Coverage (Misspecified)')
    ax.legend(); ax.grid(True)

    # Bottom-right: Split CP Width
    ax = axes[1, 1]
    ax.plot(conf_levels * 100, width_split[label],
            label=f'Split CP (NN), {label}')
    ax.set_xlabel('Nominal Confidence ($1-\\alpha$) (%)')
    ax.set_ylabel('Mean Interval Width')
    ax.set_title('Split Conformal (NN) - Width (Misspecified)')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig("BNN_VI_vs_Split_CP_NN_misspec.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    # run_bnn_vs_cp_experiment()
    run_bnn_vs_cp_experiment_misspec()