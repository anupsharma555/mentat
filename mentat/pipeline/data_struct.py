
from dataclasses import dataclass, asdict
import pandas as pd
from datasets import Dataset
import unicodedata

@dataclass
class QuestionData:
    q_id: int
    category: str
    answer_a: str
    answer_b: str
    answer_c: str
    answer_d: str
    answer_e: str
    creator_truth: list
    text_male: str = None
    text_female: str = None
    text_nonbinary: str = None
    possible_modifiers: list = None
    comment: str = None


def convert_questions_to_df(input_list: list):
    """"""
    return pd.DataFrame([asdict(entry) for entry in input_list])

def convert_questions_to_dictlist(input_list: list):
    """"""
    return [asdict(entry) for entry in input_list]

def convert_df_to_huggingface(dict_list):
    """"""
    return Dataset.from_list(dict_list)

def clean_text(text: str, do_lower: bool = False) -> str:
    """Basic string cleaning"""

    cleaned = str(text).strip()
    if do_lower:
        cleaned = cleaned.lower()
    cleaned = unicodedata.normalize('NFC', cleaned)
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("‘", "'").replace("’", "'")
    
    return cleaned

def check_modifiers(entry):
    """"""

    output = []
    if isinstance(entry, str):
        if entry.find("<AGE>") >= 0:
            output.append("age")
        if entry.find("<NAT>") >= 0:
            output.append("nat")
    
    return output



# from dataclasses import dataclass, asdict
# from typing import Optional, List
# from datasets import Dataset

# @dataclass
# class DataEntry:
#     text: str
#     label: Optional[str] = None
#     metadata: Optional[dict] = None

# # 1. Prepare a list of dataclass objects
# dataset_entries = [
#     DataEntry(text="Hello, world!", label="greeting", metadata={"lang": "en"}),
#     DataEntry(text="¡Hola Mundo!", label="greeting", metadata={"lang": "es"}),
#     DataEntry(text="Bonjour le monde!", label="greeting", metadata={"lang": "fr"}),
# ]

# # 2. Convert each dataclass to a dict
# dict_list = [asdict(entry) for entry in dataset_entries]

# # 3. Create a Hugging Face Dataset
# hf_dataset = Dataset.from_list(dict_list)
