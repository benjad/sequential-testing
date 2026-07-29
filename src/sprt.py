import numpy as np
from scipy.stats import norm
from src.validations import validate_probability, validate_lenght, validate_options
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class SPRT(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def plot(self):
        pass


class SPRTBinomial(SPRT):
    """
     The sequential probability ratio test for binomial distribution

    Parameters
    ----------
    alpha (float):The level of significance or the probability of making a Type I error (false positive).
    beta (float): The probability of making a Type II error (false negative).
    h0 (float): The null Hypothesis.
    h1 (float): The alternative Hypothesis.
    method {"fixed", "two-arms"}: estimation method.
        "fixed": Null Hypothesis is a fixed value entered by the user.
        "two-arms": Null Hypothesis stays that there is no difference between two arms of data.
                  Ho is set to 0.5 and H1 to 0.5 plus the uplift.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        h0: float,
        h1: float,
        method: str = "fixed",
    ):
        # Input arguments
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.h0 = h0
        self.h1 = h1
        # Necessary arguments
        self.a = np.log(self.alpha / (1 -self.beta))
        self.b = np.log((1 - self.beta) / self.alpha)
        self.lr = []
        self.h0_rejected = []
        self.h0_acepted = []
        validate_probability(alpha)
        validate_probability(beta)
        validate_options(method, ["fixed", "two-arms"])

    def _likelihood(self, x, n, p) -> bool:
        return x * np.log(p) + (n - x) * np.log(1 - p)

    def _update_llr(self, values) -> list:
        n = len(values)
        x = np.array(values).sum()
        new_llr = self._likelihood(x, n, self.h1) - self._likelihood(x, n, self.h0)
        self.lr.append(new_llr)
        return new_llr

    def _hipothesis_evaluation(self, llr, step):
        if llr < self.a:
            self.h0_acepted.extend([(llr, step)])
        if llr > self.b:
            self.h0_rejected.extend([(llr, step)])

    def fit(self, x_values, y_values=None):
        self.lr.clear()
        if self.method == "fixed":
            for i in range(len(x_values)):
                _values = x_values[: i + 1]
                llr = self._update_llr(_values)
                self._hipothesis_evaluation(llr, i + 1)

        if self.method == "two-arms":
            uplift = self.h1 / self.h0 - 1
            for i in range(len(x_values)):
                _x = x_values[: i + 1]
                _y = y_values[: i + 1]
                x = np.array(_x).sum()
                y = np.array(_y).sum()
                new_llr = self._likelihood(y, x + y, 0.5 * (1 + uplift)) - self._likelihood(
                    y, x + y, 0.5
                )
                self.lr.append(new_llr)
                self._hipothesis_evaluation(new_llr, i + 1)

    def plot(self):
        x = np.arange(1, len(self.lr) + 1)
        y = self.lr
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="LLR", color="blue")
        plt.axhline(self.b, color="red", linestyle="--", label="Upper Boundary (Reject H0)")
        plt.axhline(self.a, color="green", linestyle="--", label="Lower Boundary (Accept H0)")
        plt.fill_between(x, self.a, self.b, color="gray", alpha=0.1)
        plt.xlabel("Step")
        plt.ylabel("Log-Likelihood Ratio (LLR)")
        plt.title("Sequential A/B Test: LLR vs Decision Boundaries")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class SPRTNormal(SPRT):
    """
    The sequential probability ratio test for normal distribution

    Parameters
    ----------
    alpha (float):  The level of significance for the test.
    beta (float): The probability of making a Type II error (false negative).
    sigma (float): The standard deviation used in the calculation of variance.
    h0 (float): The null Hypothesis. 
    h1 (float): The alternative Hypothesis.
    """

    def __init__(self, alpha: float, beta: float, sigma: float, h0: float, h1: float):
        # Input arguments
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.h0 = h0
        self.h1 = h1
        # Necessary arguments
        self.a = np.log(self.alpha / (1 -self.beta))
        self.b = np.log((1 - self.beta) / self.alpha)
        self.lr = []
        self.h0_rejected = []
        self.h0_acepted = []
        validate_probability(alpha)
        validate_probability(beta)

    def _likelihood(self, x, mu, sigma) -> float:
        return np.sum(norm.logpdf(x, loc=mu, scale=sigma))

    def _update_llr(self, values) -> list:
        new_llr = self._likelihood(values, self.h1, self.sigma) - self._likelihood(
            values, self.h0, self.sigma
        )
        self.lr.append(new_llr)
        return new_llr

    def _hipothesis_evaluation(self, llr, step):
        if llr < self.a:
            self.h0_acepted.extend([(llr, step)])
        if llr > self.b:
            self.h0_rejected.extend([(llr, step)])

    def fit(self, x_values, y_values):
        validate_lenght(x_values, y_values)
        self.lr.clear()
        values = y_values - x_values
        for i in range(len(values)):
            _values = values[: i + 1]
            llr = self._update_llr(_values)
            self._hipothesis_evaluation(llr, i + 1)

    def plot(self):
        x = np.arange(1, len(self.lr) + 1)
        y = self.lr
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="LLR", color="blue")
        plt.axhline(self.b, color="red", linestyle="--", label="Upper Boundary (Reject H0)") 
        plt.axhline(self.a, color="green", linestyle="--", label="Lower Boundary (Accept H0)")
        plt.fill_between(x, self.a, self.b, color="gray", alpha=0.1)
        plt.xlabel("Step")
        plt.ylabel("Log-Likelihood Ratio (LLR)")
        plt.title("Sequential A/B Test: LLR vs Decision Boundaries")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
