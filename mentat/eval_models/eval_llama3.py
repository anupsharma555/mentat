#!/usr/bin/env python3

import os
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


candidate_tokens_default = ["A", "B", "C", "D", "E"]
# candidate_tokens_default = [" A", " B", " C", " D", " E"]

SYSTEM_MESSAGE = (
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information."
)


def calculate_cross_entropy(true_probs, model_probs, eps: float = 1e-12):
    """
    Cross-entropy: H(p_true, p_model) = - sum_i p_true_i * log(p_model_i)
    """
    cross_entropy = -sum(
        p_true * math.log(p_model + eps)
        for p_true, p_model in zip(true_probs, model_probs)
    )
    return cross_entropy

def calculate_model_probs(candidate_logprobs):
    """
    Convert logprobs -> normalized probabilities for each candidate.
    Uses the log-sum-exp trick.
    """
    logsumexp_val = np.logaddexp.reduce(candidate_logprobs)
    model_probs = [math.exp(lp - logsumexp_val) for lp in candidate_logprobs]
    return model_probs

def check_is_correct(true_probs, model_probs):
    """Checks if the argmax of `model_probs` matches the argmax of `true_probs`"""
    true_label_index = true_probs.index(max(true_probs))
    predicted_label_index = model_probs.index(max(model_probs))
    return (true_label_index == predicted_label_index)

def format_llama3_prompt(system_message: str, user_message: str) -> str:
    """Formats a prompt according to Llama 3 instructions"""

    prompt = "<|begin_of_text|>"
    if system_message:
        prompt += "<|start_header_id|>system<|end_header_id|>\n" + system_message.strip() + "<|eot_id|>"
    prompt += "<|start_header_id|>user<|end_header_id|>\n" + user_message.strip()[:-24] + "<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>" + "Answer: "

    return prompt

def get_candidate_logprobs_for_prompt(
    client,
    model_name: str,
    prompt: str,
    candidate_tokens=None,
    topk: int = 20,
    system_message: str = SYSTEM_MESSAGE
):
    """
    Wraps the provided prompt in the Llama 3 format, tokenizes it, and computes the next-token
    log probabilities for each candidate. Returns:
      - candidate_logprobs: list of floats (log probabilities) corresponding to candidate_tokens
      - model_response: a dict resembling the OpenAI response structure.
    """
    if candidate_tokens is None:
        candidate_tokens = candidate_tokens_default

    model = client["model"]         # AutoModelForCausalLM
    tokenizer = client["tokenizer"] # AutoTokenizer
    device = client["device"]       # torch device

    prompt = prompt.strip()
    full_prompt = format_llama3_prompt(system_message, prompt)
    encoded_input = tokenizer(full_prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = encoded_input["input_ids"]

    # Get next-token logits
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]  # shape: [vocab_size]

    log_probs = torch.log_softmax(next_token_logits, dim=-1) 

    # Build candidate_logprobs in the same order as candidate_tokens
    candidate_logprobs = []
    for token in candidate_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            candidate_logprobs.append(float("-inf"))
        else:
            candidate_logprobs.append(log_probs[token_id].item())

    # # Debug to manually check top-k tokens to see if model performs well
    # topk = min(topk, 25)  # Ensure topk is not greater than 25
    # topk_values, topk_indices = torch.topk(log_probs, topk)
    # topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices.tolist())
    # print("Top 25 tokens with their log probabilities:")
    # for i in range(topk):
    #     print(f"{topk_tokens[i]} (id {topk_indices[i]}): {topk_values[i].item()}")

    # Construct a dict that resembles the OpenAI response structure for downstream analysis.
    model_response = {
        "choices": [
            {
                "message": {
                    "content": ""
                }
            }
        ]
    }

    return candidate_logprobs, model_response


def evaluate_dataset_on_model(dataset, client, model_name="llama3"):
    """
    Expects each sample in the dataset to have:
      - "prompt_mcq": string ending with the multiple choice question prompt.
      - "labels": a list of 5 probabilities across [A, B, C, D, E]
    Returns average cross-entropy, average accuracy, and a DataFrame with per-sample results.
    """
    candidate_tokens = candidate_tokens_default

    total_cross_entropy = 0.0
    total_correct_predictions = 0
    num_samples = 0

    eval_result = []

    # Evaluate sample-by-sample
    for sample in tqdm(dataset):
        prompt = sample["prompt_mcq"]
        true_probs = sample["labels"] 
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

        # Store results in the same format
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
    # Select one of the instruct models:
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "Henrychur/MMed-Llama-3-8B-EnIns"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    client = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }

    # List of dataset directories and model alias names.
    file_names = [
        "mentat_data_base",
        "mentat_data_gender",
        "mentat_data_nat",
        "mentat_data_age"
    ]
    models = [
        # "llama3_1_8b_instruct",
        # "llama3_2_3b_instruct",
        "llama3_8b_mmeds",
    ]

    for f in file_names:
        for m in models:
            print(f"Evaluating dataset '{f}' on local model alias '{m}'")
            store_location = os.path.join(os.getcwd(), "eval_data", f)
            dataset = load_from_disk(store_location)  # Optionally, select a subset e.g. .select(range(20))
            print(dataset.shape)

            average_ce, average_acc, eval_results = evaluate_dataset_on_model(dataset, client, m)
            print("Average Cross Entropy:", average_ce)
            print("Average Accuracy:", average_acc)

            # Save results with a timestamp.
            current_datetime = datetime.now()
            datetime_string = current_datetime.strftime("_%Y_%m_%d_%H%M")
            out_file = f"eval_results_{m}_{f}{datetime_string}.pkl"
            eval_results.to_pickle(out_file)
            print("Saved results to:", out_file)


if __name__ == "__main__":
    main()
