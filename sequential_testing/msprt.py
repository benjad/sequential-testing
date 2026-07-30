import numpy as np
from sequential_testing.validations import validate_probability, validate_lenght
from scipy.stats import norm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class MSPRT(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    def optimal_mixture_variance(self, alpha: float, sigma: float, m: float) -> float:
        """
        Calculates the mixture variance (https://arxiv.org/pdf/1512.04922)

        Parameters
        ----------
            alpha (float): The level of significance .
            sigma (float): The standard deviation used in the calculation of variance.
            m (int): The maximum number of observations that the user is willing to wait.

        """
        b = (2 * np.log(1 / alpha)) / np.sqrt(m * sigma**2)
        return round(sigma**2 * (norm.cdf(-b) / ((1 / b) * norm.pdf(b) - norm.cdf(-b))), 2)


class MSPRTBinomial(MSPRT):
    """
    The mixture sequential probability ratio test for binomial distribution (https://dl.acm.org/doi/epdf/10.1145/3097983.3097992)

    Parameters
    ----------
    alpha (float):The level of significance or the probability of making a Type I error (false positive).
    beta (float): The probability of making a Type II error (false negative).
    sigma (float): The standard deviation used in the calculation of variance.
    h0 (float): The null Hypothesis.
    m (int): The maximum number of observations that the user is willing to wait.
    tau (float): The mixture variance.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        sigma: float,
        tau: float,
        h0: float,
        m: int,
    ):
        validate_probability(alpha)
        validate_probability(beta)
        # Input arguments
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.m = m
        self.tau = (
            self.tau if tau else self.optimal_mixture_variance(self.alpha, self.sigma, self.m)
        )
        self.h0 = h0
        # Necessary arguments
        self.b = (1 - self.beta) / self.alpha
        self.lr = []
        self.h0_rejected = []
        self.h0_acepted = []

    def _likelihood_ratio(self, x_values, y_values, tau):
        n = len(x_values)
        Vn = np.mean(x_values) * (1 - np.mean(x_values)) + np.mean(y_values) * (
            1 - np.mean(y_values)
        )

        result = np.sqrt((Vn) / (Vn + n * tau**2)) * np.exp(
            ((n) ** 2 * tau**2 * (np.mean(y_values - x_values) - self.h0) ** 2)
            / (2 * Vn * (Vn + n * tau**2))
        )

        return result

    def _update_llr(self, x_values, y_values):
        new_llr = self._likelihood_ratio(x_values, y_values, self.tau)
        self.lr.append(new_llr)
        return new_llr

    def _hipothesis_evaluation(self, llr, step):
        if llr > self.b:
            self.h0_rejected.extend([(llr, step)])

    def fit(self, x_values, y_values):
        validate_lenght(x_values, y_values)
        lenght = len(x_values)
        self.lr.clear()
        for i in range(lenght):
            _x = x_values[: i + 1]
            _y = y_values[: i + 1]
            llr = self._update_llr(_x, _y)
            self._hipothesis_evaluation(llr, i + 1)

    def plot(self):
        x = np.arange(1, len(self.lr) + 1)
        y = self.lr
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="LR", color="blue")
        plt.axhline(0, color="red", linestyle="--")
        plt.axhline(self.b, color="green", linestyle="--", label="Upper Boundary (Reject H0)")
        plt.fill_between(x, 0, self.b, color="gray", alpha=0.1)
        plt.xlabel("Step")
        plt.ylabel("Likelihood Ratio (LR)")
        plt.title("Sequential A/B Test: LR vs Decision Boundaries")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class MSPRTNormal(MSPRT):
    """
    The mixture sequential probability ratio test for normal distribution (https://dl.acm.org/doi/epdf/10.1145/3097983.3097992)

    Parameters
    ----------
    alpha (float):The level of significance or the probability of making a Type I error (false positive).
    beta (float): The probability of making a Type II error (false negative).
    sigma (float): The standard deviation used in the calculation of variance.
    h0 (float): The null Hypothesis.
    m (int): The maximum number of observations that the user is willing to wait.
    tau (float): The mixture variance.
    """

    def __init__(self, alpha: float, beta: float, sigma: float, tau: float, h0: float, m: int):
        validate_probability(alpha)
        validate_probability(beta)
        # Input arguments
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.h0 = h0
        self.m = m
        self.tau = (
            self.tau if tau else self.optimal_mixture_variance(self.alpha, self.sigma, self.m)
        )
        # Necessary arguments
        self.b = (1 - self.beta) / self.alpha
        self.lr = []
        self.h0_rejected = []
        self.h0_acepted = []

    def _likelihood_ratio(self, x_values, y_values, sigma, tau):
        n = len(x_values)
        root_part = np.sqrt(2 * sigma**2 / (2 * sigma**2 + (n * tau**2)))
        exponential_part = np.exp(
            (n**2 * tau**2 * (np.mean(y_values) - np.mean(x_values) - self.h0) ** 2)
            / (4 * sigma**2 * (2 * sigma**2 + n * tau**2))
        )
        result = root_part * exponential_part
        return result

    def _update_llr(self, x_values, y_values):
        new_llr = self._likelihood_ratio(x_values, y_values, self.sigma, self.tau)
        self.lr.append(new_llr)
        return new_llr

    def _hipothesis_evaluation(self, llr, step):
        if llr > self.b:
            self.h0_rejected.extend([(llr, step)])

    def fit(self, x_values, y_values):
        validate_lenght(x_values, y_values)
        lenght = len(x_values)
        self.lr.clear()
        for i in range(lenght):
            _x = x_values[: i + 1]
            _y = y_values[: i + 1]
            llr = self._update_llr(_x, _y)
            self._hipothesis_evaluation(llr, i + 1)

    def plot(self):
        x = np.arange(1, len(self.lr) + 1)
        y = self.lr
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="LR", color="blue")
        plt.axhline(0, color="red", linestyle="--")
        plt.axhline(self.b, color="green", linestyle="--", label="Upper Boundary (Reject H0)")
        plt.fill_between(x, 0, self.b, color="gray", alpha=0.1)
        plt.xlabel("Step")
        plt.ylabel("Likelihood Ratio (LR)")
        plt.title("Sequential A/B Test: LR vs Decision Boundaries")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
