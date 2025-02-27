
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
