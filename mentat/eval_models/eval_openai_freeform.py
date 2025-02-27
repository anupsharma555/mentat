from openai import OpenAI, PermissionDeniedError
import numpy as np
import pandas as pd
import math
from datasets import load_from_disk
import os
from datetime import datetime
from tqdm import tqdm


client = OpenAI()
# candidate_tokens_default = [" A", " B", " C", " D", " E"]
candidate_tokens_default = ["A", "B", "C", "D", "E"]

def transform_to_dict(objects):
    # Create a dictionary comprehension to extract token and logprob
    result = {obj.token: obj.logprob for obj in objects if hasattr(obj, 'token') and hasattr(obj, 'logprob')}
    return result


def calculate_cross_entropy(true_probs, model_probs, eps: float = 1e-12):
    """Helper function to calcualte CE"""

    # Cross-entropy: H(p_true, p_model) = - sum_i p_true_i * log(p_model_i)
    cross_entropy = -sum(
        p_true * math.log(p_model + eps)
        for p_true, p_model in zip(true_probs, model_probs)
    )
    return cross_entropy

def calculate_model_probs(candidate_logprobs):
    """"""
    # Convert logprobs -> normalized probabilities
        #    p_i = exp(lp_i) / sum_j exp(lp_j)
        #    (some may be -inf if not in top-20)
    logsumexp_val = np.logaddexp.reduce(candidate_logprobs)
    model_probs = [math.exp(lp - logsumexp_val) for lp in candidate_logprobs]

    return model_probs

def check_is_correct(true_probs, model_probs):
    """"""
    # Accuracy: Check if the max-probability prediction matches the true label
    true_label_index = true_probs.index(max(true_probs))
    predicted_label_index = model_probs.index(max(model_probs))
    return  true_label_index == predicted_label_index

def get_candidate_freeform_for_prompt(
    client,
    model_name: str,
    prompt: str,
    candidate_tokens=None,
    topk: int = 20
):
    """"""

    prompt = prompt[:-2] + " (write your reply in only one short sentence!):"
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = [
            client.chat.completions.create(
            model=model_name,
            messages=messages,
            # max_completion_tokens=10,
            temperature=1.0,
        )
        for j in range(10)
    ]

    return response

def get_candidate_logprobs(logprobs_dict, candidate_tokens):
    """Helper to retrieve logprobs for answer candidate token"""

    candidate_logprobs = []
    for token in candidate_tokens:
        if token in logprobs_dict:
            candidate_logprobs.append(logprobs_dict[token])
        else:
            # Not in top-k => effectively probability is 0
            candidate_logprobs.append(float("-inf"))

    return candidate_logprobs




def evaluate_dataset_on_model(dataset, client, model_name="gpt-3.5-turbo"):
    """
    Expects each sample in dataset to have:
      - "prompt": a string ending in "Answer (single letter):"
      - "answer_probabilities": the ground truth distribution across 5 possible letters
                                e.g. [0.1, 0.2, 0.5, 0.2, 0.0]
    We'll assume the 5 letters are [A, B, C, D, E] (leading space form).

    Returns:
      - The average cross-entropy over the dataset.
      - The average accuracy over the dataset.
    """
    candidate_tokens = candidate_tokens_default

    total_cross_entropy = 0.0
    total_correct_predictions = 0
    num_samples = 0

    eval_result = []

    # for sample in dataset:
    for sample in tqdm(dataset):
        # if sample["category"] != "treatment":
        # if sample["category"] != "triage":
        if sample["category"] not in ["diagnosis", "triage", "treatment"]:
            continue
        # print(sample["q_id"])
        # print(sample)
        prompt = sample["prompt_freeform"]
        # print(sample["q_id"], true_probs)
        model_response = get_candidate_freeform_for_prompt(
            client=client,
            model_name=model_name,
            prompt=prompt,
            candidate_tokens=candidate_tokens,
            topk=20,
        )
        model_response_alt = [completion_message.choices[0].message.content for completion_message in model_response]
        current_result = {**sample}
        current_result["model_response"] = model_response
        current_result["model_response_alternative"] = model_response_alt
        current_result["candidate_logprobs"] = None
        current_result["cross_entropy"] = None
        current_result["is_correct"] = None
        eval_result.append(current_result)

    avg_ce = total_cross_entropy / num_samples if num_samples > 0 else 0.0
    avg_accuracy = total_correct_predictions / num_samples if num_samples > 0 else 0.0

    return avg_ce, avg_accuracy, pd.DataFrame(eval_result)

def main():
    file_names = [
        "mentat_data_base", 
    ]
    models = [
        # "gpt-4o-mini-2024-07-18", 
        # "gpt-4o-2024-08-06", 
        "o1-2024-12-17",
        # "o1-mini-2024-09-12"
    ]
    for f_i, f in enumerate(file_names):
        for m_i, m in enumerate(models):
            print(f_i, f, m_i, m)
            store_location = os.path.join(os.getcwd() + "/eval_data", f)
            dataset = load_from_disk(store_location)
            print(dataset.shape)

            average_ce, average_acc, eval_results = evaluate_dataset_on_model(dataset, client, m)
            print("Average Cross Entropy:", average_ce)
            print("Average Accuracy:", average_acc)

            current_datetime = datetime.now()
            datetime_string = current_datetime.strftime("_%Y_%m_%d_%H%M")
            eval_results.to_pickle(f'eval_results_{m}_{f}{datetime_string}.pkl')


if __name__ == "__main__":
    main()
