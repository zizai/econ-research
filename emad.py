import numpy as np
import pandas as pd
import seaborn as sns

class Economy:
    def __init__(self, productivity0, labor0, capital0, lamb_L, lamb_K, CL, CK, DL, DK, rho, gamma=0.5,
                 innovation=False, capacity_factor=10, response_type='poisson', log_period=50):
        self.productivity = productivity0
        self.labor = labor0
        self.capital = capital0
        self.lamb_L = lamb_L
        self.lamb_K = lamb_K
        self.lamb_A = 0
        self.CL = CL
        self.CK = CK
        self.DL = DL
        self.DK = DK
        self.rho = rho
        self.gamma = gamma
        self.innovation = innovation
        self.capacity_factor = capacity_factor
        self.response_type = response_type
        self.log_period = log_period

        self.steps = 0
        self.dt = 0.001

        self.labor_hist = [labor0]
        self.capital_hist = [capital0]
        self.production_hist = [self.production]

    def update(self):
        labor = np.copy(self.labor)
        capital = np.copy(self.capital)
        productivity = np.copy(self.productivity)

        # labor += self.dt * self.lamb_L * self.labor - self.dt * self.labor * np.dot(self.CL, self.capital)
        # capital += self.dt * self.capital * np.dot(self.CK, self.labor) - self.dt * self.lamb_K * self.capital

        # capacity = self.production * self.capacity_factor
        capacity = self.productivity + self.labor + self.capital

        # do_LK = np.where(np.random.uniform(size=self.labor.shape) > 0.5, 1, 0) * self.labor ** gamma / capacity
        # do_LL = np.where(np.random.uniform(size=self.labor.shape) > 0.5, 1, 0) * self.labor ** gamma / capacity
        # do_KK = np.where(np.random.uniform(size=self.capital.shape) > 0.5, 1, 0) * self.capital ** gamma / capacity
        if self.response_type == 'basic':
            do_LK = self.labor ** self.gamma / capacity
            do_LL = self.labor ** self.gamma / capacity
            do_KK = self.capital ** self.gamma / capacity
        elif self.response_type == 'poisson':
            do_LK = np.clip(np.random.poisson(lam=self.labor ** self.gamma) / capacity, 0, 1)
            do_LL = np.clip(np.random.poisson(lam=self.labor ** self.gamma) / capacity, 0, 1)
            do_KK = np.clip(np.random.poisson(lam=self.capital * self.gamma) / capacity, 0, 1)
        elif self.response_type == 'seasonal':
            do_LK = (np.sin(2 * np.pi * self.steps * self.dt) + 1) / 2 * (self.labor ** self.gamma / capacity)
            do_LL = (np.sin(2 * np.pi * self.steps * self.dt) + 1) / 2 * (self.labor ** self.gamma / capacity)
            do_KK = (np.sin(2 * np.pi * self.steps * self.dt) + 1) / 2 * (self.labor ** self.gamma / capacity)
        elif self.response_type == 'constant':
            do_LK = 0.5
            do_LL = 1.
            do_KK = 1.
        elif self.response_type == 'monster':
            do_LK = self.labor ** 0.5 / (self.productivity * self.capital ** 0.5)
            do_LL = 1.
            do_KK = 1.
        else:
            raise NotImplementedError

        # if self.response_type == 'constant':
        #     labor += self.dt * self.lamb_L * self.labor  - self.dt * self.labor * np.dot(self.CL, self.capital)
        #     capital += self.dt * self.capital * np.dot(self.CL.T, self.labor) - self.dt * self.lamb_K * self.capital
        # else:
        # labor += self.dt * self.lamb_L * self.labor - self.dt * (do_LK * self.labor) * np.dot(self.CL, self.capital)
        labor += self.dt * self.lamb_L * self.labor * (1 - do_LK) - self.dt * (do_LK * self.labor) * np.dot(self.CL,
                                                                                                            self.capital)
        # labor += self.dt * self.lamb_L * self.labor - self.dt * self.labor * np.dot(self.CL, (self.capital / (1 + 0.1 * self.capital)))
        # labor += self.dt * self.lamb_L * self.labor * (1 - self.labor / capacity) - self.dt * (self.labor / (1 + 0.1 * self.labor)) * np.dot(self.CL, (self.capital / (1 + 0.1 * self.capital)))
        capital += self.dt * self.capital * np.dot(self.CK, do_LK * self.labor) - self.dt * self.lamb_K * self.capital
        # capital += self.dt * self.capital * np.dot(self.CL.T, do_LK * self.labor) - self.dt * self.lamb_K * self.capital
        # capital += self.dt * self.capital * np.dot(self.CK, (self.labor / (1 + 0.1 * self.labor))) - self.dt * self.lamb_K * self.capital
        # capital += self.dt * (self.capital / (1 + 0.1 * self.capital)) * np.dot(self.CK, self.labor) - self.dt * self.lamb_K * self.capital

        labor -= self.dt * do_LL * self.labor * np.dot(self.DL, self.labor)
        # labor -= do_LL * self.dt * self.labor * np.dot(self.DL, self.labor - 1)
        # labor -= self.dt * self.labor * np.dot(self.DL, np.maximum(self.labor - 1, 0))
        # labor -= self.dt * np.maximum(self.labor - 1, 0) * np.dot(self.DL, self.labor)
        capital -= self.dt * do_KK * self.capital * np.dot(self.DK, self.capital)
        # capital -= self.dt * self.capital * np.dot(self.DK, self.capital - 1)
        # capital -= self.dt * self.capital * np.dot(self.DK, np.maximum(self.capital - 1, 0))

        if self.innovation:
            productivity += self.dt * self.productivity * (
                        do_LL * 0.1 * np.dot(self.DL, self.labor) + do_KK * 0.1 * np.dot(self.DK, self.capital))

        self.labor = labor
        self.capital = capital
        self.productivity = productivity
        self.steps += 1

        if self.steps % self.log_period == 0:
            self.labor_hist.append(self.labor)
            self.capital_hist.append(self.capital)
            self.production_hist.append(self.production)

    @property
    def production(self):
        return self.productivity * self.labor ** self.rho * self.capital ** (1 - self.rho)

    def plot_classes(self, hist, label):
        data = np.concatenate(hist)
        data = pd.DataFrame(data, columns=[label])
        N = len(hist[0])
        T = len(hist)
        sector = [i for i in range(N)] * T
        data['sector'] = sector
        steps = []
        for t in range(T):
            steps += [t * self.log_period] * N
        data['steps'] = steps

        sns.set_style('whitegrid')
        ax = sns.lineplot(x='steps', y=label, hue='sector', data=data)

        for line in ax.lines:
            line.set_alpha(0.5)

    def plot_aggregate(self):
        prod_hist, labor_hist, capital_hist = self.production_hist, self.labor_hist, self.capital_hist
        agg_prod_hist = [np.sum(row) for row in prod_hist]
        agg_labor_hist = [np.sum(row) for row in labor_hist]
        agg_capital_hist = [np.sum(row) for row in capital_hist]
        data = np.concatenate([agg_prod_hist, agg_labor_hist, agg_capital_hist])
        data = pd.DataFrame(data, columns=['quantity'])
        T = len(agg_prod_hist)
        data['variable'] = ['production'] * T + ['labor'] * T + ['capital'] * T
        data['steps'] = [t * self.log_period for t in range(T)] + [t * self.log_period for t in range(T)] + [t * self.log_period for t in range(T)]

        sns.set_style('whitegrid')
        sns.lineplot(x='steps', y='quantity', hue='variable', data=data)
