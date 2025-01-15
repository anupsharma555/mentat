
import numpy as np
from scipy.optimize import minimize


def bt_prob(i, j, beta):
    """
    Probability that item i is preferred over item j in BT given beta
    """
    return np.exp(beta[i]) / (np.exp(beta[i]) + np.exp(beta[j]))


class BradleyTerry:
    """"""
    def __init__(self, data: list, k: int = 5):
        self.__data = data
        self.__k = k
        self.__betas = np.zeros(k)

        self.__fitres = self.fit()

    @staticmethod
    def bradley_terry_neg_log_likelihood(betas, data):
        """
        beta: 1D array of length
        data: list of (winner, loser) pairs.
        
        Returns: scalar negative log-likelihood for the Bradley-Terry model.
        """

        nll = 0.0
        for winner, loser in data:
            numerator = np.exp(betas[winner])
            denominator = numerator + np.exp(betas[loser])
            nll += np.log(denominator) - np.log(numerator)
        
        return nll

    def prob(self, i, j):
        """
        Probability that item i is preferred over item j in BT given current beta values
        """
        return np.exp(self.__betas[i]) / (np.exp(self.__betas[i]) + np.exp(self.__betas[j]))
    
    def fit(self):
        """"""

        constraints = ({
            'type': 'eq',
            'fun': self.sum_to_zero_constraint
        },)

        # Optimize using scipy
        result = minimize(
            fun=self.bradley_terry_neg_log_likelihood,
            x0=self.__betas,
            args=(self.__data),
            constraints=constraints,
            method='trust-constr',  # or 'SLSQP'
            options={'disp': True}
        )
        self.__betas = result.x

        # if not result.success:
        #     raise Warning("Optimization did not converge")
        
        return result
    
    def return_probs(self):
        """Return softmaxed betas to get absolut probability for each answer"""

        summed = np.array([np.exp(self.__betas[i]) for i in range(self.__k)]).sum()
        return np.array([np.exp(self.__betas[i])/summed for i in range(self.__k)])

    @staticmethod
    def sum_to_zero_constraint(beta):
        return np.sum(beta)
    
    @property
    def data(self):
        return self.__data
    
    @property
    def k(self):
        return self.__k
    
    @property
    def betas(self):
        return self.__betas
    
    @property
    def fitres(self):
        return self.__fitres


def pairwise_wins(annotator_arr):
    """Code helper to create pairwise wins"""

    # Convert list of 1D arrays into 2D array (to be sure)
    test_mat = np.array(annotator_arr)
    k = test_mat.shape[-1]
    data = []

    for i in range(k):
        # Calculate column-wise comparison matrix
        comp_mat = np.array([test_mat[:, i] > test_mat[:, j] for j in range(k)]) 
        row_idx, col_idx = np.nonzero(comp_mat.T)
        # Create win pairs for current index i (i cannot win against itself)
        pairs = [(i, v) for v in col_idx]  
        assert len(pairs) == comp_mat.sum(), "Number of win-pairs does not match comparison matrix entries"
        # Add pairs to data
        data += pairs

    return data


def main():
    k = 5

    # Example case
    pairwise_data = [
        (0, 1), (0, 2), (1, 2), (3, 4), (3, 0),
        (0, 4), (2, 4), (1, 3), (2, 3), (0, 3),
    ]

    bt7274 = BradleyTerry(data=pairwise_data, k=k)

    print("Optimization status:", bt7274.fitres.message)
    print("Estimated beta parameters (one per item):")
    for i, b in enumerate(bt7274.betas):
        print(f"  Item {i}: {b:.4f}")

    i, j = 0, 1
    prob_i_beats_j = bt7274.prob(i, j)
    print(f"\nProbability that item {i} is preferred over item {j}: {prob_i_beats_j:.4f}")

if __name__ == "__main__":
    main()

