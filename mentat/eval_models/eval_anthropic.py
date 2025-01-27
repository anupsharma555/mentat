
from anthropic import Anthropic
import numpy as np
import pandas as pd
import math
from datasets import load_from_disk
import os
from datetime import datetime
from tqdm import tqdm


from mentat.eval_models.eval_openai import transform_to_dict, calculate_cross_entropy, calculate_model_probs
from mentat.eval_models.eval_openai import check_is_correct, get_candidate_logprobs, candidate_tokens_default

client = Anthropic()




# def transform_to_dict(objects):
#     # Create a dictionary comprehension to extract token and logprob
#     result = {obj.token: obj.logprob for obj in objects if hasattr(obj, 'token') and hasattr(obj, 'logprob')}
#     return result


# def calculate_cross_entropy(true_probs, model_probs, eps: float = 1e-12):
#     """Helper function to calcualte CE"""

#     # Cross-entropy: H(p_true, p_model) = - sum_i p_true_i * log(p_model_i)
#     cross_entropy = -sum(
#         p_true * math.log(p_model + eps)
#         for p_true, p_model in zip(true_probs, model_probs)
#     )
#     return cross_entropy

# def calculate_model_probs(candidate_logprobs):
#     """"""
#     # Convert logprobs -> normalized probabilities
#         #    p_i = exp(lp_i) / sum_j exp(lp_j)
#         #    (some may be -inf if not in top-20)
#     logsumexp_val = np.logaddexp.reduce(candidate_logprobs)
#     model_probs = [math.exp(lp - logsumexp_val) for lp in candidate_logprobs]

#     return model_probs

# def check_is_correct(true_probs, model_probs):
#     """"""
#     # Accuracy: Check if the max-probability prediction matches the true label
#     true_label_index = true_probs.index(max(true_probs))
#     predicted_label_index = model_probs.index(max(model_probs))
#     return  true_label_index == predicted_label_index

# def get_candidate_logprobs(logprobs_dict, candidate_tokens):
#     """Helper to retrieve logprobs for answer candidate token"""

#     candidate_logprobs = []
#     for token in candidate_tokens:
#         if token in logprobs_dict:
#             candidate_logprobs.append(logprobs_dict[token])
#         else:
#             # Not in top-k => effectively probability is 0
#             candidate_logprobs.append(float("-inf"))

#     return candidate_logprobs

def get_candidate_logprobs_for_prompt(
    client,
    model_name: str,
    prompt: str,
    candidate_tokens=None,
    topk: int = 20
):
    """"""

    if candidate_tokens is None:
        candidate_tokens = candidate_tokens_default

    # Hotfix, due to Claude's being too dumb to reply in just a letter...
    prompt = prompt[:-16] + "only reply with a single letter!): "

    messages = [
        {"role": "user", "content": prompt},
    ]
    response = client.messages.create(
        model=model_name,
        max_tokens=500,
        messages=messages,
        temperature=0.0,
    )

    top_probs_dict = {response.content[0].text[0]: 1.}
    # print(response)
    # print(top_probs_dict)
    # 1. / 0.


    # Retrieve logprobs for each candidate token
    candidate_logprobs = get_candidate_logprobs(top_probs_dict, candidate_tokens)

    return candidate_logprobs, response





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
        # print(sample["q_id"])
        # print(sample)
        prompt = sample["prompt_mcq"]
        true_probs = sample["labels"] 
        # print(sample["q_id"], true_probs)
        candidate_logprobs, model_response = get_candidate_logprobs_for_prompt(
            client=client,
            model_name=model_name,
            prompt=prompt,
            candidate_tokens=candidate_tokens,
            topk=20,
        )
        
        model_probs = calculate_model_probs(candidate_logprobs)
        cross_entropy = calculate_cross_entropy(true_probs, model_probs)
        is_correct = check_is_correct(true_probs, model_probs)
        if is_correct:
            total_correct_predictions += 1

        total_cross_entropy += cross_entropy
        num_samples += 1

        current_result = {**sample}
        current_result["model_response"] = model_response
        current_result["candidate_logprobs"] = candidate_logprobs
        current_result["cross_entropy"] = cross_entropy
        current_result["is_correct"] = is_correct
        eval_result.append(current_result)

        # 1. / 0.

    avg_ce = total_cross_entropy / num_samples if num_samples > 0 else 0.0
    avg_accuracy = total_correct_predictions / num_samples if num_samples > 0 else 0.0

    return avg_ce, avg_accuracy, pd.DataFrame(eval_result)

def main():
    file_names = [
        "mentat_data_base", 
        "mentat_data_gender", 
        "mentat_data_nat", 
        "mentat_data_age"
    ]
    models = [
        # "claude-3-5-sonnet-20241022",
        # "claude-3-5-haiku-20241022",
        # "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]
    for f_i, f in enumerate(file_names):
        for m_i, m in enumerate(models):
            print(f_i, f, m_i, m)
            store_location = os.path.join(os.getcwd() + "/eval_data", f)
            dataset = load_from_disk(store_location) # .select(range(30))
            print(dataset.shape)

            average_ce, average_acc, eval_results = evaluate_dataset_on_model(dataset, client, m)
            print("Average Cross Entropy:", average_ce)
            print("Average Accuracy:", average_acc)

            current_datetime = datetime.now()
            datetime_string = current_datetime.strftime("_%Y_%m_%d_%H%M")
            eval_results.to_pickle(f'eval_results_{m}_{f}{datetime_string}.pkl')


if __name__ == "__main__":
    main()

            
"""
mentat_data_base
    claude-3-5-sonnet-20241022
    Average Cross Entropy: 8.249455583109409
    Average Accuracy: 0.7704918032786885

    claude-3-5-haiku-20241022
    Average Cross Entropy: 9.679189486823029
    Average Accuracy: 0.7103825136612022

    claude-3-opus-20240229
    Average Cross Entropy: 9.203810906644376
    Average Accuracy: 0.73224043715847

    claude-3-haiku-20240307
    Average Cross Entropy: 9.72093733658344
    Average Accuracy: 0.7049180327868853

mentat_data_gender
    claude-3-5-sonnet-20241022
    Average Cross Entropy: 7.793683284379184
    Average Accuracy: 0.7887067395264117

    claude-3-5-haiku-20241022
    Average Cross Entropy: 8.965739287970091
    Average Accuracy: 0.73224043715847

    claude-3-opus-20240229
    Average Cross Entropy: 8.559704003041661
    Average Accuracy: 0.7486338797814208

    claude-3-haiku-20240307
    Average Cross Entropy: 10.22155348742272
    Average Accuracy: 0.6903460837887068

mentat_data_nat
    claude-3-5-sonnet-20241022
    Average Cross Entropy: 7.92764441903427
    Average Accuracy: 0.7841530054644809

    claude-3-5-haiku-20241022
    Average Cross Entropy: 9.182310008639554
    Average Accuracy: 0.7240437158469946

    claude-3-opus-20240229
    Average Cross Entropy: nan
    Average Accuracy: 0.7531876138433515

    claude-3-haiku-20240307
    Average Cross Entropy: 10.06768052606014
    Average Accuracy: 0.6839708561020036


mentat_data_age
    claude-3-5-sonnet-20241022
    Average Cross Entropy: 7.993479839510974
    Average Accuracy: 0.780327868852459

    claude-3-5-haiku-20241022
    Average Cross Entropy: 8.924858489760883
    Average Accuracy: 0.7278688524590164

    claude-3-opus-20240229
    Average Cross Entropy: 8.326237865816784
    Average Accuracy: 0.7530054644808744
    
    claude-3-haiku-20240307
    Average Cross Entropy: 10.064544061465691
    Average Accuracy: 0.6808743169398908
"""