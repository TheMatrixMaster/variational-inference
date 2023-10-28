import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal
from typing import List, Dict, Tuple

from abc import ABC, abstractmethod


class Distribution(ABC):
    @abstractmethod
    def pdf(self, x):
        pass


class Bernoulli(Distribution):
    p: float

    def __init__(self, p) -> None:
        super().__init__()
        self.p = p

    def pdf(self, x):
        return self.p**x * (1-self.p)**(1-x)
    
    def logpdf(self, x):
        return x*np.log(self.p) + (1-x)*np.log(1-self.p)
    
    def expectation(self):
        return self.p
    
    def expectation_log(self):
        return self.p * np.log(self.p) + (1-self.p) * np.log(1-self.p)
    
    def update_p(self, p):
        self.p = p


class Normal(Distribution):
    mu: float
    sigma: float

    def __init__(self, mu, sigma) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x):
        return multivariate_normal.pdf(x, mean=self.mu, cov=self.sigma)
    
    def logpdf(self, x):
        return multivariate_normal.logpdf(x, mean=self.mu, cov=self.sigma)
    
    def expectation(self):
        return self.mu
    
    def update_mu(self, mu) -> None:
        self.mu = mu

    def update_sigma(self, sigma) -> None:
        self.sigma = sigma


class Variational(Distribution):
    m: int
    latents: List[Tuple[Normal, Bernoulli]]

    def __init__(self, m) -> None:
        super().__init__()
        self.m: int = m

    def setup(self, mu: float, precision: float, gamma: float) -> None:
        self.latents = [(Normal(mu, 1/precision), Bernoulli(gamma)) for _ in range(self.m)]

    def pdf(self, beta, s):
        return np.prod([self.latents[i][0].pdf(beta[i]) * self.latents[i][1].pdf(s[i]) for i in range(self.m)])
    
    def logpdf(self, beta, s):
        return np.sum([self.latents[i][0].logpdf(beta[i]) + self.latents[i][1].logpdf(s[i]) for i in range(self.m)])

    def update_mu(self, mu, j):
        self.latents[j][0].update_mu(mu)

    def update_sigma(self, precision, j):
        self.latents[j][0].update_sigma(1/precision)

    def update_gamma(self, gamma, j):
        self.latents[j][1].update_p(gamma)

    def get_mu(self):
        return np.array([self.latents[i][0].mu for i in range(self.m)])
    
    def get_sigma(self):
        return np.array([self.latents[i][0].sigma for i in range(self.m)])
    
    def get_gamma(self):
        return np.array([self.latents[i][1].p for i in range(self.m)])
    

class EM_algorithm:

    def __init__(self, var: Variational, hparams: Dict[str, str]) -> None:
        self.var = var
        self.hparams = hparams

    def setup(self, **initial_latents) -> None:
        self.var.setup(**initial_latents)

    def E_step(self, mbeta: np.ndarray, ld: np.ndarray, n: int) -> None:
        """
        Update the latent distribution parameters using the other latents parameters,  
        the hyperparameters and the observed data.
        """
        for j in range(self.var.m):
            new_precision = n * ld[j][j] * self.hparams["tau_epsilon"] + self.hparams["tau_beta"]

            new_mu = n*self.hparams["tau_epsilon"]/new_precision *  \
                (mbeta[j] - np.sum(np.delete(np.prod(np.vstack([    \
                    self.var.get_mu(), self.var.get_gamma(), ld[j]]), axis=0), j)))

            new_uj = np.log(self.hparams["pi"] / (1-self.hparams["pi"])) \
                + 0.5 * np.log(self.hparams["tau_beta"] / new_precision) \
                + new_precision/2*(new_mu**2)

            new_gamma = 1 / (1 + np.exp(-new_uj))

            self.var.update_mu(new_mu, j)
            self.var.update_sigma(new_precision, j)
            self.var.update_gamma(new_gamma, j)

        # After a full cycle of updates, we cap gamma to avoid numerical instability
        for j in range(self.var.m):
            self.var.update_gamma(np.clip(self.var.latents[j][1].p, 0.01, 0.99), j)

    def M_step(self) -> None:
        """
        Update the hyperparameters using the current latent parameter estimates and the data.
        In this tutorial, we don't update the tau_epsilon hyperparameter for simplicity.
        """
        new_tau_epsilon = self.hparams["tau_epsilon"]

        new_tau_beta_inv = np.sum(np.multiply(  \
            self.var.get_gamma(), np.power(self.var.get_mu(), 2) + self.var.get_sigma()))   \
                / np.sum(self.var.get_gamma())
        
        new_pi = 1/self.var.m * np.sum(self.var.get_gamma())

        self.hparams["tau_epsilon"] = new_tau_epsilon
        self.hparams["tau_beta"] = 1/new_tau_beta_inv
        self.hparams["pi"] = new_pi

    def compute_elbo(self, mbeta: np.ndarray, ld: np.ndarray, n: int) -> float:
        """
        Compute the evidence lower bound (ELBO) of the model by using the current variational 
        distribution and the joint likelihood of the data and the latent variables. These 
        distributions are parameterized by our current estimates of hyperparameter values and 
        latent distribution parameters.
        """
        exp_var_s = np.sum([v[1].expectation_log() for v in self.var.latents])
        exp_var_beta = -0.5 * np.log(self.hparams["tau_beta"]) * np.sum(self.var.get_gamma())

        summand = np.multiply(self.var.get_gamma(), \
                                 np.power(self.var.get_mu(), 2) + self.var.get_sigma())
        
        exp_true_beta = -0.5 * self.hparams["tau_beta"] * np.sum(summand)
        exp_true_s = np.sum(self.var.get_gamma() * np.log(self.hparams["pi"])   \
                            + (1 - self.var.get_gamma()) * np.log(1 - self.hparams["pi"]))
        
        double_summand = 0
        for j in range(self.var.m):
            for k in range(j+1, self.var.m):
                gamma_j = self.var.latents[j][1].expectation()
                mu_j = self.var.latents[j][0].expectation()
                gamma_k = self.var.latents[k][1].expectation()
                mu_k = self.var.latents[k][0].expectation()
                double_summand += gamma_j*mu_j*gamma_k*mu_k*ld[j][k]
        
        exp_true_y = 0.5*n*np.log(self.hparams["tau_epsilon"])  \
            - 0.5*self.hparams["tau_epsilon"]*n \
            + self.hparams["tau_epsilon"]*np.multiply(self.var.get_gamma(), self.var.get_mu())@(n*mbeta)    \
            - 0.5*n*self.hparams["tau_epsilon"]*np.sum(summand*ld.diagonal())   \
            - self.hparams["tau_epsilon"]*(n*double_summand)
        
        return exp_true_y + exp_true_beta + exp_true_s - exp_var_beta - exp_var_s
    
    def run(self, mbeta: np.ndarray, ld: np.ndarray, n: int, max_iter: int, tol: float=1e-3) -> List[float]:
        """
        Run the EM algorithm for a given number of iterations or until convergence.
        """
        elbo = []
        for i in range(max_iter):
            self.E_step(mbeta, ld, n)
            self.M_step()
            elbo.append(self.compute_elbo(mbeta, ld, n))
            if i > 0 and abs(elbo[-1] - elbo[-2]) < tol:
                break
        return elbo
    
    def plot_elbo(self, elbo: List[float]) -> None:
        plt.scatter(np.arange(len(elbo)), elbo, s=10)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.title("ELBO as a function of EM iteration")
        plt.show()

    def plot_pip(self, causal_snps: List[int]) -> None:
        pips = self.var.get_gamma()
        x = np.arange(len(pips))
        c = ['red' if v in causal_snps else 'black' for v in x]
        plt.scatter(x, pips, s=10, c=c)
        plt.xlabel("SNP index")
        plt.ylabel("Posterior inclusion probability")
        plt.title("Inferred posterior inclusion probability of each SNP")
        custom_lines = [Line2D([0], [0], color='black', lw=2), Line2D([0], [0], color='red', lw=2)]
        plt.legend(custom_lines, ['Non-causal', 'Causal'])
        plt.show()

    @staticmethod
    def plot_preds(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
        b, a = np.polyfit(y_pred, y_true, deg=1)
        plt.scatter(y_pred, y_true, s=8)
        plt.plot(y_pred, a + b*y_pred, color="red", lw=1.5)
        plt.xlabel("Predicted phenotype")
        plt.ylabel("True phenotype")
        plt.title(title)
        plt.show()
    

if __name__ == "__main__":
    m = 100
    N_train = 439
    N_test = 50
    dir = os.getcwd()

    # ------------------------------------------------------------
    # Part 1. Mean Field Variational Inference
    # ------------------------------------------------------------
    ld = pd.read_csv(dir+"/data/LD.csv", header=0, index_col=0)
    mbeta = pd.read_csv(dir+"/data/beta_marginal.csv", header=0, index_col=0)["V1"].to_numpy()
    snp_map = ld.index.to_numpy()

    var = Variational(m)
    em = EM_algorithm(var, {"tau_epsilon": 1, "tau_beta": 200, "pi": 0.01})
    em.setup(mu=0, precision=1, gamma=0.01)

    elbo = em.run(mbeta, ld.to_numpy(), N_train, 15)
    em.plot_elbo(elbo)

    # ------------------------------------------------------------
    # Part 2. Evaluating PRS prediction
    # ------------------------------------------------------------
    X_train = pd.read_csv(dir+"/data/X_train.csv", header=0, index_col=0).to_numpy()
    y_train = pd.read_csv(dir+"/data/y_train.csv", header=0, index_col=0)["V1"].to_numpy()
    X_test = pd.read_csv(dir+"/data/X_test.csv", header=0, index_col=0).to_numpy()
    y_test = pd.read_csv(dir+"/data/y_test.csv", header=0, index_col=0)["V1"].to_numpy()

    y_train_pred = X_train@(var.get_gamma()*var.get_mu())
    y_test_pred = X_test@(var.get_gamma()*var.get_mu())

    y_train_pred_mar = X_train@mbeta
    y_test_pred_mar = X_test@mbeta

    train_pearson = np.corrcoef(y_train, y_train_pred)[0, 1]
    test_pearson = np.corrcoef(y_test, y_test_pred)[0, 1]

    train_pearson_mar = np.corrcoef(y_train, y_train_pred_mar)[0, 1]
    test_pearson_mar = np.corrcoef(y_test, y_test_pred_mar)[0, 1]

    em.plot_preds(y_train, y_train_pred, "Predicted vs. true phenotype (training set), r={:.2f}".format(train_pearson))
    em.plot_preds(y_test, y_test_pred, "Predicted vs. true phenotype (test set), r={:.2f}".format(test_pearson))
    em.plot_preds(y_train, y_train_pred_mar, "Predicted vs. true phenotype (training set), marginal model, r={:.2f}".format(train_pearson_mar))
    em.plot_preds(y_test, y_test_pred_mar, "Predicted vs. true phenotype (test set), marginal model, r={:.2f}".format(test_pearson_mar))

    # ------------------------------------------------------------
    # Part 3. Evaluating Fine-mapping
    # ------------------------------------------------------------
    causal_snps = ["rs9482449", "rs7771989", "rs2169092"]
    causal_snps = np.where(np.isin(snp_map, causal_snps))[0]
    em.plot_pip(causal_snps)
