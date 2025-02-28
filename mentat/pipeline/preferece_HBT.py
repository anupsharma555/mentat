import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple

from mentat.pipeline import helper_functions

def create_pairwise_data(df: pd.DataFrame) -> List[Tuple]:
    """
    Convert DataFrame to list of (q_no, winner, loser, rater_id, count) tuples
    """
    pairwise_data = []
    
    for _, row in df.iterrows():
        q_no = row['q_no']
        rater = row['rater_id']
        responses = np.array(row['response'])
        
        # Compare each pair of options
        for i in range(len(responses)):
            for j in range(len(responses)):
                if i != j:
                    # Count how many times i was rated higher than j
                    if responses[i] > responses[j]:
                        pairwise_data.append((q_no, i, j, rater, 1))
    
    # Aggregate counts for same comparisons
    df_pairs = pd.DataFrame(pairwise_data, 
                          columns=['q_no', 'winner', 'loser', 'rater', 'count'])
    df_pairs = df_pairs.groupby(['q_no', 'winner', 'loser', 'rater'])['count'].sum().reset_index()
    
    return [tuple(row) for row in df_pairs.values]


class HierarchicalBradleyTerry:
    def __init__(self, df: pd.DataFrame, k: int = 5):
        self.k = k  # number of options per question
        self.pairwise_data = create_pairwise_data(df)
        
        # Get unique questions and raters
        self.questions = sorted(df['q_no'].unique())
        self.raters = sorted(df['rater_id'].unique())
        self.n_questions = len(self.questions)
        self.n_raters = len(self.raters)
        
        # Create mappings for quick lookup
        self.q_idx = {q: i for i, q in enumerate(self.questions)}
        self.rater_idx = {r: i for i, r in enumerate(self.raters)}
        
        # Initialize parameters
        self.betas = np.zeros((self.n_questions, k))  # one beta vector per question
        self.slopes = np.ones(self.n_raters)  # one slope (gamma) per rater
        self.offsets = np.zeros(self.n_raters)  # one offset (alpha) per rater
        
        self.fit_result = None
        self.loss_curve = None
    
    def _pack_params(self) -> np.ndarray:
        """Pack all parameters into a single vector for optimization"""
        return np.concatenate([
            self.betas.flatten(),
            self.slopes,
            self.offsets
        ])
    
    def _unpack_params(self, params: np.ndarray) -> None:
        """Unpack parameter vector into individual parameters"""
        n_betas = self.n_questions * self.k
        self.betas = params[:n_betas].reshape(self.n_questions, self.k)
        self.slopes = params[n_betas:n_betas + self.n_raters]
        self.offsets = params[n_betas + self.n_raters:]
    
    def neg_log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood for hierarchical Bradley-Terry model
        """
        self._unpack_params(params)
        nll = 0.0
        
        for q_no, winner, loser, rater, count in self.pairwise_data:
            q_idx = self.q_idx[q_no]
            r_idx = self.rater_idx[rater]
            
            # Get parameters
            beta_diff = self.betas[q_idx, winner] - self.betas[q_idx, loser]
            gamma = self.slopes[r_idx]
            alpha = self.offsets[r_idx]
            
            # Compute log probability using logaddexp for numerical stability
            nll += count * np.logaddexp(0, -(gamma * beta_diff + alpha))
            
        # Add l2 regularization for slopes and offsets
        nll += 5. * np.sum((self.slopes - 1.) ** 2)
        nll += 5. * np.sum(self.offsets ** 2)
        
        return nll
    
    def sum_to_zero_constraints(self, params: np.ndarray) -> np.ndarray:
        """Ensure betas for each question sum to zero"""
        betas = params[:self.n_questions * self.k].reshape(self.n_questions, self.k)
        return betas.sum(axis=1)
    
    def fit(self) -> None:
        """Fit the hierarchical Bradley-Terry model"""
        x0 = self._pack_params()
        
        # Set up constraints
        constraints = []
        for i in range(self.n_questions):
            constraints.append({
                'type': 'eq',
                'fun': lambda x, i=i: self.sum_to_zero_constraints(x)[i]
            })
        
        # Set up bounds
        bounds = []
        # Bounds for betas (unconstrained)
        bounds.extend([(-np.inf, np.inf)] * (self.n_questions * self.k))
        # Bounds for slopes (e.g., must be between 0.5 and 1.5)
        bounds.extend([(0.5, 2.0)] * self.n_raters)
        # bounds.extend([(0.9999999, 1.000001)] * self.n_raters)
        # Bounds for offsets (e.g., must be between -1 and 1)
        bounds.extend([(-3., 3.)] * self.n_raters)
        # bounds.extend([(-0.000001, 0.000001)] * self.n_raters)

        # Optimize
        self.fit_result = minimize(
            fun=self.neg_log_likelihood,
            x0=x0,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP',
            options={'disp': False, 'maxiter': 1000}
        )
        
        # Update parameters with optimal values
        if self.fit_result.success:
            self._unpack_params(self.fit_result.x)
        else:
            raise Warning("Optimization did not converge")
    
    def get_prob(self, q_no: int, i: int, j: int, rater: str) -> float:
        """
        Get probability that option i is preferred over j for given question and rater
        """
        q_idx = self.q_idx[q_no]
        r_idx = self.rater_idx[rater]
        
        beta_diff = self.betas[q_idx, i] - self.betas[q_idx, j]
        gamma = self.slopes[r_idx]
        alpha = self.offsets[r_idx]
        
        return 1 / (1 + np.exp(-(gamma * beta_diff + alpha)))
    
    def get_rater_parameters(self) -> Dict:
        """Return dictionary of rater parameters"""
        return {
            rater: {
                'slope': self.slopes[i],
                'offset': self.offsets[i]
            }
            for rater, i in self.rater_idx.items()
        }
    
    def get_question_betas(self, q_no: int) -> np.ndarray:
        """Return beta parameters for a specific question"""
        return self.betas[self.q_idx[q_no]]
    
    def get_answer_probabilities(self, q_no: int, method: str = 'raw') -> np.ndarray:
        """
        Return softmaxed probabilities for all answers for a specific question.
        
        Args:
            q_no: Question number
            method: How to handle annotator parameters:
                   'raw' - use only betas (ignore annotator parameters)
                   'average' - average probabilities across all annotators
        
        Returns:
            Array of probabilities summing to 1
        """
        betas = self.get_question_betas(q_no)
        
        if method == 'raw':
            # Simply softmax the betas (like in original implementation)
            exp_betas = np.exp(betas)
            return exp_betas / exp_betas.sum()
        
        elif method == 'average':
            # Average probabilities across all annotators
            all_probs = np.zeros((self.n_raters, self.k))
            
            for rater, r_idx in self.rater_idx.items():
                gamma = self.slopes[r_idx]
                alpha = self.offsets[r_idx]
                
                # Compute pairwise probs for this annotator
                scaled_betas = gamma * betas + alpha
                exp_scaled = np.exp(scaled_betas)
                all_probs[r_idx] = exp_scaled / exp_scaled.sum()
            
            # Return mean probability across annotators
            return all_probs.mean(axis=0)
        
        else:
            raise ValueError("method must be one of: 'raw', 'average'")
    
def main(response_data = None):

    if response_data is None:
        # Data Import
        directory = './annotated_data/'
        # directory = './test_annotated_data_simple/'

        raw_data = helper_functions.import_raw_annotations(directory)
        helper_functions.annotation_data_check(raw_data)
        print(raw_data.shape)

        # Processing data
        processed_data = helper_functions.process_raw_data_annotations(raw_data)
        response_data, annotator_individual_data = processed_data

    model = HierarchicalBradleyTerry(response_data)
    model.fit()

    # Get annotator characteristics
    rater_params = model.get_rater_parameters()
    # print("Rater parameters:")
    # for rp in rater_params:
    #     print(rp)
    #     print(rater_params[rp])

    # Get question-specific preferences
    q_no = 32
    betas = model.get_question_betas(q_no)
    # print(f"Betas for question {q_no}:", betas)

    return model

if __name__ == "__main__":
    main()