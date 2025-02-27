
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
        client.messages.create(
            model=model_name,
            max_tokens=1000,
            messages=messages,
            temperature=1.0,
        )
        for j in range(10)
    ]
    print(response)

    return response


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
        if sample["category"] != "treatment":
        # if sample["category"] != "triage":
        # if sample["category"] != "diagnosis":
            continue
        print(sample["q_id"], sample["category"])
        # print(sample)
        prompt = sample["prompt_freeform"]
        # print(sample["q_id"], true_probs)
        model_responses = get_candidate_freeform_for_prompt(
            client=client,
            model_name=model_name,
            prompt=prompt,
            candidate_tokens=candidate_tokens,
            topk=20,
        )

        current_result = {**sample}
        current_result["model_response"] = model_responses
        current_result["candidate_logprobs"] = None
        current_result["cross_entropy"] = None
        current_result["is_correct"] = None
        eval_result.append(current_result)

        # 1. / 0.

    avg_ce = total_cross_entropy / num_samples if num_samples > 0 else 0.0
    avg_accuracy = total_correct_predictions / num_samples if num_samples > 0 else 0.0

    return avg_ce, avg_accuracy, pd.DataFrame(eval_result)

def main():
    file_names = [
        "mentat_data_base", 
    ]
    models = [
        "claude-3-5-sonnet-20241022",
        # "claude-3-5-haiku-20241022",
        # "claude-3-opus-20240229",
        # "claude-3-haiku-20240307",
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
            eval_results.to_pickle(f'eval_results_freeform_{m}_{f}{datetime_string}.pkl')


if __name__ == "__main__":
    main()
