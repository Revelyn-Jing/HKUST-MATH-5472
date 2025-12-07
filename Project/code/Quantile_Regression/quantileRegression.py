import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# =============================
# 数据生成：三个场景
# =============================

def generate_data(case, n, seed=None):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, 1))

    if case == "linear_correct":
        # 正确的线性 + 高斯噪声
        y = 2.0 + 3.0 * X[:, 0] + rng.normal(scale=1.0, size=n)

    elif case == "nonlinear_misspec":
        # 轻度错配：分段均值 + 异方差
        mean = 2.0 * np.sign(X[:, 0])
        sigma = 0.3 + 0.7 * np.abs(X[:, 0])
        y = mean + rng.normal(scale=sigma, size=n)

    elif case == "heavy_tail_bad":
        # 极度错配：非线性 + heavy-tail + 强异方差
        mean = 3.0 * (X[:, 0] ** 3)

        df = 2.0
        Z = rng.normal(size=n)
        U = rng.chisquare(df, size=n)
        t_noise = Z / np.sqrt(U / df)  # t_2

        scale = 0.3 + 2.0 * np.abs(X[:, 0])
        noise = scale * t_noise
        y = mean + noise

    else:
        raise ValueError("Unknown case")

    return X, y

# =============================
# Split Conformal（基于线性回归残差）
# =============================

def split_conformal_interval(X_train, y_train, X_cal, y_cal, X_test, alpha=0.1):
    lr = LinearRegression().fit(X_train, y_train)

    y_cal_pred = lr.predict(X_cal)
    calib_resid = np.abs(y_cal - y_cal_pred)

    n_cal = len(calib_resid)
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))
    q_hat = np.sort(calib_resid)[k - 1]

    y_test_pred = lr.predict(X_test)
    lower = y_test_pred - q_hat
    upper = y_test_pred + q_hat
    return lower, upper

# =============================
# 线性 Quantile Regression
# =============================

def quantile_regression_interval(X_train, y_train, X_test, alpha=0.1):
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    tau_low = alpha / 2.0
    tau_high = 1.0 - alpha / 2.0

    mod_low = sm.QuantReg(y_train, X_train_sm)
    res_low = mod_low.fit(q=tau_low, max_iter=2000)
    lower = res_low.predict(X_test_sm)

    mod_high = sm.QuantReg(y_train, X_train_sm)
    res_high = mod_high.fit(q=tau_high, max_iter=2000)
    upper = res_high.predict(X_test_sm)

    return lower, upper

# =============================
# 通用：给定 (case, n_train, n_cal) 做多次 Monte Carlo
# =============================

def run_experiment(case_name,
                   n_train=200,
                   n_cal=200,
                   n_test=2000,
                   alpha=0.1,
                   n_rep=100,
                   base_seed=0):
    cp_cov_list, cp_len_list = [], []
    qr_cov_list, qr_len_list = [], []

    for rep in range(n_rep):
        seed = base_seed + rep

        X_train, y_train = generate_data(case_name, n_train, seed=seed)
        X_cal, y_cal = generate_data(case_name, n_cal, seed=seed + 1000)
        X_test, y_test = generate_data(case_name, n_test, seed=seed + 2000)

        # Split CP
        cp_low, cp_up = split_conformal_interval(
            X_train, y_train, X_cal, y_cal, X_test, alpha=alpha
        )
        cp_cov = np.mean((y_test >= cp_low) & (y_test <= cp_up))
        cp_len = np.mean(cp_up - cp_low)
        cp_cov_list.append(cp_cov)
        cp_len_list.append(cp_len)

        # Quantile Regression（train+cal 一起拟合）
        qr_low, qr_up = quantile_regression_interval(
            np.vstack([X_train, X_cal]),
            np.concatenate([y_train, y_cal]),
            X_test,
            alpha=alpha,
        )
        qr_cov = np.mean((y_test >= qr_low) & (y_test <= qr_up))
        qr_len = np.mean(qr_up - qr_low)
        qr_cov_list.append(qr_cov)
        qr_len_list.append(qr_len)

    results = {
        "cp_cov_mean": np.mean(cp_cov_list),
        "cp_cov_std": np.std(cp_cov_list),
        "cp_len_mean": np.mean(cp_len_list),
        "cp_len_std": np.std(cp_len_list),
        "qr_cov_mean": np.mean(qr_cov_list),
        "qr_cov_std": np.std(qr_cov_list),
        "qr_len_mean": np.mean(qr_len_list),
        "qr_len_std": np.std(qr_len_list),
    }
    return results

# =============================
# 实验 1：模型错配
# =============================

def experiment_model_misspec(alpha=0.1):
    print("========== 实验 1：模型错配 ==========")

    print("\nCase A: 线性正确模型（linear_correct）")
    resA = run_experiment("linear_correct",
                          n_train=200, n_cal=200, n_test=2000,
                          alpha=alpha, n_rep=100)
    print(f"Split CP       : cov = {resA['cp_cov_mean']:.3f} ± {resA['cp_cov_std']:.3f}, "
          f"len = {resA['cp_len_mean']:.3f} ± {resA['cp_len_std']:.3f}")
    print(f"Quantile Reg   : cov = {resA['qr_cov_mean']:.3f} ± {resA['qr_cov_std']:.3f}, "
          f"len = {resA['qr_len_mean']:.3f} ± {resA['qr_len_std']:.3f}")

    print("\nCase B: heavy-tail + 强异方差 + 非线性（heavy_tail_bad）")
    # 这里保持训练/校准样本中等，让错配 + heavy-tail 主导
    resB = run_experiment("heavy_tail_bad",
                          n_train=200, n_cal=200, n_test=2000,
                          alpha=alpha, n_rep=100)
    print(f"Split CP       : cov = {resB['cp_cov_mean']:.3f} ± {resB['cp_cov_std']:.3f}, "
          f"len = {resB['cp_len_mean']:.3f} ± {resB['cp_len_std']:.3f}")
    print(f"Quantile Reg   : cov = {resB['qr_cov_mean']:.3f} ± {resB['qr_cov_std']:.3f}, "
          f"len = {resB['qr_len_mean']:.3f} ± {resB['qr_len_std']:.3f}")

# =============================
# 实验 2：有限样本 n 的影响（同一个 heavy_tail_bad 模型）
# =============================

def experiment_finite_sample(alpha=0.1):
    print("\n========== 实验 2：有限样本 ==========")
    sample_sizes = [50, 100, 200, 500]

    for n in sample_sizes:
        print(f"\nheavy_tail_bad, n_train = n_cal = {n}")
        res = run_experiment("heavy_tail_bad",
                             n_train=n, n_cal=n, n_test=2000,
                             alpha=alpha, n_rep=100)
        print(f"Split CP       : cov = {res['cp_cov_mean']:.3f} ± {res['cp_cov_std']:.3f}, "
              f"len = {res['cp_len_mean']:.3f} ± {res['cp_len_std']:.3f}")
        print(f"Quantile Reg   : cov = {res['qr_cov_mean']:.3f} ± {res['qr_cov_std']:.3f}, "
              f"len = {res['qr_len_mean']:.3f} ± {res['qr_len_std']:.3f}")

# =============================
# 主程序
# =============================

if __name__ == "__main__":
    alpha = 0.1  # 90% 区间

    experiment_model_misspec(alpha=alpha)
    experiment_finite_sample(alpha=alpha)
