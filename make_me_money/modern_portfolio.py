import matplotlib.pyplot as plt
import numpy as np


def make_me_a_portfolio_please(mean_return, cov_returns):
    def softmax(w):
        a = np.exp(w)
        return a / a.sum()

    cov_np = cov_returns.to_numpy()
    N = 100000
    D = len(mean_return)
    returns = np.zeros(N)
    risks = np.zeros(N)
    random_weights = []
    for i in range(N):
        w = softmax(np.random.randn(D))
        np.random.shuffle(w)
        random_weights.append(w)
        ret = mean_return.dot(w)
        risk = np.sqrt(w.dot(cov_np).dot(w))
        returns[i] = ret
        risks[i] = risk

    single_asset_returns = np.zeros(D)
    single_asset_risks = np.zeros(D)
    for i in range(D):
        ret = mean_return[i]
        risk = np.sqrt(cov_np[i, i])

        single_asset_returns[i] = ret
        single_asset_risks[i] = risk

    bounds = [(0, None)] * D

    from scipy.optimize import linprog
    A_eq = np.ones((1, D))
    b_eq = np.ones(1)
    N = 100
    # minimize
    res = linprog(mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    min_return = res.fun
    res = linprog(-mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    max_return = -res.fun

    target_returns = np.linspace(min_return, max_return, num=N)
    from scipy.optimize import minimize
    def get_portfolio_variance(weights):
        return weights.dot(cov_returns).dot(weights)

    def target_return_constraint(weights, target):
        return weights.dot(mean_return) - target

    def portfolio_constraint(weights):
        return weights.sum() - 1

    constraints = [
        {
            'type': 'eq',
            'fun': target_return_constraint,
            'args': [target_returns[0]],  # will be updated in loop
        },
        {
            'type': 'eq',
            'fun': portfolio_constraint,
        }
    ]

    res = minimize(
        fun=get_portfolio_variance,
        x0=np.ones(D) / D,  # uniform
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )

    optimized_risks = []
    for target in target_returns:
        # set target return constraint
        constraints[0]['args'] = [target]

        res = minimize(
            fun=get_portfolio_variance,
            x0=np.ones(D) / D,  # uniform
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
        )
        optimized_risks.append(np.sqrt(res.fun))
        if res.status != 0:
            print(res)

    def neg_sharpe_ratio(weights):
        mean = weights.dot(mean_return)
        sd = np.sqrt(weights.dot(cov_np).dot(weights))
        return -mean / sd

    res = minimize(
        fun=neg_sharpe_ratio,
        x0=np.ones(D) / D,  # uniform
        method='SLSQP',
        constraints={
            'type': 'eq',
            'fun': portfolio_constraint,
        },
        bounds=bounds,
    )

    best_sr, best_w = -res.fun, res.x

    mc_best_w = None
    mc_best_sr = float('-inf')
    for i, (risk, ret) in enumerate(zip(risks, returns)):
        sr = ret / risk
        if sr > mc_best_sr:
            mc_best_sr = sr
            mc_best_w = random_weights[i]
    print(mc_best_w, mc_best_sr)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(risks, returns, alpha=0.1);

    # found by optimization
    opt_risk = np.sqrt(best_w.dot(cov_np).dot(best_w))
    opt_ret = mean_return.dot(best_w)
    plt.scatter([opt_risk], [opt_ret], c='red');

    # found by monte carlo simulation
    mc_risk = np.sqrt(mc_best_w.dot(cov_np).dot(mc_best_w))
    mc_ret = mean_return.dot(mc_best_w)
    plt.scatter([mc_risk], [mc_ret], c='pink');
    plt.show()