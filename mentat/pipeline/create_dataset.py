
import os
import pandas as pd
import numpy as np
import pickle
from itertools import product
from datasets import Dataset

from mentat.pipeline import data_struct
from mentat.config import config_params


class MentatDataSet:
    """"""

    def __init__(self, directory: str, filename: str, preference_data_filename: str = None, remove_bad_q_inds: bool = True, overwrite_with_preference: bool = True):
        self._directory = directory
        self._filename = filename
        self._remove_bad_q_inds = remove_bad_q_inds

        # Imported from config files 
        self._train_vs_test_split = config_params.train_vs_test_split
        self._random_seed = config_params.random_seed_train_test
        self._inds_bad_post_annotate = config_params.inds_bad_post_annotate

        self._possible_categories = None
        self._possible_q_ids = None
        self._q_ids_train = None
        self._q_ids_test = None
        self._question_dataset = self._import_raw_questions()
        if overwrite_with_preference:                
            self._question_dataset = self._overwrite_with_preference_labels(
                self._question_dataset, "hbt", preference_data_filename
            )
        self._calculate_train_test_ids()

        

    def _import_raw_questions(self):
        """"""

        file_path = os.path.join(self._directory, self._filename)
        df_raw = pd.read_csv(file_path)
        
        labels = {
            "a": [1., 0., 0., 0., 0.],
            "b": [0., 1., 0., 0., 0.],
            "c": [0., 0., 1., 0., 0.],
            "d": [0., 0., 0., 1., 0.],
            "e": [0., 0., 0., 0., 1.],
        }

        dataset = []
        for index, row in df_raw.iterrows():
            valid_entry = True
            for key_key in ["q_id", "Correct Answer", "a", "b", "c", "d", "e"]:
                if pd.isna(row[key_key]):
                    valid_entry = False
            # Removes Qs that were flagged for whatever reason
            if self._remove_bad_q_inds and int(row["q_id"]) in self._inds_bad_post_annotate:
                valid_entry = False
            if not valid_entry:
                continue


            text_male = None if pd.isna(row["he"]) else data_struct.clean_text(row["he"])
            text_female = None if pd.isna(row["she"]) else data_struct.clean_text(row["she"])
            text_nonbinary = None if pd.isna(row["they"]) else data_struct.clean_text(row["they"])        
            answer_a = data_struct.clean_text(row["a"])
            answer_b = data_struct.clean_text(row["b"])
            answer_c = data_struct.clean_text(row["c"])
            answer_d = data_struct.clean_text(row["d"])
            answer_e = data_struct.clean_text(row["e"])

            # Set possible modifiers
            possible_modifiers = []
            for q_test in [text_male, text_female, text_nonbinary]:
                possible_modifiers.append(data_struct.check_modifiers(q_test))
            
            # Flatten list and remove duplicates
            possible_modifiers = list(set([x for xs in possible_modifiers for x in xs]))

            creator_truth = labels[data_struct.clean_text(row["Correct Answer"], do_lower=True)]

            entry = data_struct.QuestionData(
                q_id=int(row["q_id"]),
                category=data_struct.clean_text(row["Category"], do_lower=True),
                text_male=text_male,
                text_female=text_female,
                text_nonbinary=text_nonbinary,
                answer_a=answer_a,
                answer_b=answer_b,
                answer_c=answer_c,
                answer_d=answer_d,
                answer_e=answer_e,
                possible_modifiers=possible_modifiers,
                comment=data_struct.clean_text(row["Notes"]),
                creator_truth=creator_truth,
                truth_upper_bounds=creator_truth,
                truth_lower_bounds=creator_truth,
            )
            dataset.append(entry)

        df = data_struct.convert_questions_to_df(dataset)

        # Fix minor inconsistencies in-place:
        df.loc[df["category"] == "monitor", "category"] = "monitoring"
        df.loc[df["category"] == "treatment/practice", "category"] = "treatment"

        # Do some meta analysis
        self._possible_categories = np.unique(df["category"].to_numpy())
        self._possible_q_ids = np.unique(df["q_id"].to_numpy())

        print(f"#Raw questions: {self._possible_q_ids.shape[0]}, #Categories {self._possible_categories.shape[0]}")
        print("Categories: ", self._possible_categories)

        for cat in self._possible_categories:
            mask_cat = df["category"] == cat
            count = np.unique(df[mask_cat]["q_id"].to_numpy()).shape[-1]
            count_moddable = df[df["category"] == cat]['possible_modifiers'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()

            print(f"\t{cat}: \t\t#{count} questions (#{count_moddable} moddable)")

        return df
    
    def _calculate_train_test_ids(self):
        """Generate train and test split"""

        q_ids_all = self.possible_q_ids
        rng = np.random.default_rng(self._random_seed)
        q_ids_train = rng.choice(
            q_ids_all, 
            size=int(q_ids_all.shape[0] * self._train_vs_test_split), 
            replace=False,
        )

        q_ids_test = []
        for q_id in q_ids_all:
            if q_id in q_ids_train:
                continue
            q_ids_test.append(q_id)
        q_ids_test = np.array(q_ids_test)

        for q in q_ids_test:
            assert q not in q_ids_train, ValueError(f"q_id {q} is in test and train set")

        self._q_ids_train = q_ids_train
        self._q_ids_test = q_ids_test

    @staticmethod
    def _overwrite_with_preference_labels(
        input_df: pd.DataFrame, preference_type: str = "hbt", file_name: str = None
    ):
        """Function to import preference values to overwrite creator_truth labels"""

        assert preference_type in ["hbt", "bt"], NotImplementedError(f"preference_type = {preference_type}")
        if file_name is None:
            file_name = "analysis_results_feb27.pkl"

        # Loading annotation analysis results
        with open(file_name, 'rb') as f:
            loaded_object = pickle.load(f)
        hbt_scores, _, bt_scores, _, _ = loaded_object 

        if preference_type == "hbt":
            use_vales = hbt_scores
        else:
            use_vales = bt_scores
        for q_id in [int(p) for p in use_vales.keys()]:
            try:
                row_idx = input_df.index[input_df["q_id"] == q_id][0]
                input_df.at[row_idx, "creator_truth"] = use_vales[q_id]['bt_scores'].tolist()
                input_df.at[row_idx, "truth_upper_bounds"] = use_vales[q_id]['ci_upper'].tolist()
                input_df.at[row_idx, "truth_lower_bounds"] = use_vales[q_id]['ci_lower'].tolist()
            except IndexError:
                pass

        return input_df

    @staticmethod
    def create_prompt(q, answer_list):
        """Returns a QA prompt as string"""

        prompt = (
            f"Question: {q}\n\n"
            f"A: {answer_list[0]}\n"
            f"B: {answer_list[1]}\n"
            f"C: {answer_list[2]}\n"
            f"D: {answer_list[3]}\n"
            f"E: {answer_list[4]}\n\n"
            "Answer (single letter): "
        )
        return prompt
    
    @staticmethod
    def create_prompt_freeform(q):
        """Returns a QA prompt as string"""

        prompt = (
            f"Question: {q}\n\n"
            "Answer: "
        )
        return prompt


    def create_eval_dataset(self, n_gender: int, n_nat: int, n_age: int, only_test: bool = True):
        """"""

        # Local, fixed seed random number generator for replicability
        rng = np.random.default_rng(self._random_seed)
        final_eval_dataset = []
        random_pars = config_params.variable_demo_params

        for index, row in self._question_dataset.iterrows():
            nats = random_pars["nat_short"]
            genders = ["text_male", "text_female", "text_nonbinary"]

            assert n_gender >= 1 and n_gender <= 3, NotImplementedError("Only have three selectable gender options (male, female, non-binary)")
            assert n_nat >= 1 and n_nat <= len(nats), NotImplementedError(f"Only have {len(nats)} selectable nat options ({nats})")

            q_id = int(row["q_id"])
            category = row["category"]

            if q_id in self._q_ids_test:
                split = "test"
            elif q_id in self._q_ids_train:
                split = "train"
                if only_test:
                    continue
            else:
                ValueError(f"q_id {q_id} not listed in train or test split")

            for a_i, n_i, g_i in product(range(n_age), range(n_nat), range(n_gender)):

                random_order = rng.choice([0, 1, 2, 3, 4], 5, replace=False)
                labels = [row["creator_truth"][i] for i in random_order]
                labels_upper_bound = [row["truth_upper_bounds"][i] for i in random_order]
                labels_lower_bound = [row["truth_lower_bounds"][i] for i in random_order]       
   
                g = rng.choice(genders, 1, replace=False)[0]     
                n = rng.choice(nats, 1, replace=False)[0]
                a = rng.integers(random_pars["age_range"][0], random_pars["age_range"][1], 1)[0]
                q_text = row[g]

                if q_text is None:
                    g = rng.choice(genders, 3, replace=False)
                    i = 0
                    while q_text is None:
                        g = rng.choice(genders, 3, replace=False)[i]
                        q_text = row[g]

                no_age = False
                if "age" in row["possible_modifiers"]:
                    start_index = q_text.find("<AGE>")
                    delta_index = 5
                    q_text = q_text[:start_index] + f"{a}-year-old" + q_text[start_index + delta_index:]
                else:
                    no_age = True
                    
                no_nat = False
                if "nat" in row["possible_modifiers"]:
                    start_index = q_text.find("<NAT>")
                    delta_index = 5
                    q_text = q_text[:start_index] + f"{n}" + q_text[start_index + delta_index:]
                else:
                    no_nat = True
    
                # Verify that answers do not need to be modified
                for answer in [row["answer_a"], row["answer_b"], row["answer_c"], row["answer_d"], row["answer_e"]]:
                    assert answer.find("<NAT>") < 0 and answer.find("<AGE>") < 0, NotImplementedError("Answers need to be modified.")
    
                answer_list = [row["answer_a"], row["answer_b"], row["answer_c"], row["answer_d"], row["answer_e"]]
                answer_list = [answer_list[i] for i in random_order]
                prompt_mcq = self.create_prompt(q_text, answer_list)
                prompt_freeform = self.create_prompt_freeform(q_text)

                final_eval_dataset.append(
                    {
                        "prompt_mcq": prompt_mcq,
                        "prompt_freeform": prompt_freeform,
                        "labels": labels,
                        "labels_upper_bound": labels_upper_bound,
                        "labels_lower_bound": labels_lower_bound,
                        "gender": g,
                        "nat": n if not no_nat else None,
                        "age": a if not no_age else None,
                        "q_id": q_id,
                        "category": category,
                        "split": split,
                        "random_order": random_order,
                    }
                )

        df = pd.DataFrame(final_eval_dataset)

        return df

    @property
    def question_dataset(self):
        return self._question_dataset
    
    @property
    def possible_q_ids(self):
        return self._possible_q_ids
    
    @property
    def q_ids_test(self):
        return self._q_ids_test
    
    @property
    def q_ids_train(self):
        return self._q_ids_train


def main():
    dataset_class = MentatDataSet(os.getcwd(), "questions_raw_final.csv", overwrite_with_preference=True)
    question_dataset = dataset_class.question_dataset

    eval_dataset_base = dataset_class.create_eval_dataset(n_gender=1, n_nat=1, n_age=1)
    eval_dataset_gender = dataset_class.create_eval_dataset(n_gender=3, n_nat=1, n_age=1)
    eval_dataset_nat = dataset_class.create_eval_dataset(n_gender=1, n_nat=6, n_age=1)
    eval_dataset_age = dataset_class.create_eval_dataset(n_gender=1, n_nat=1, n_age=5)
    # eval_dataset_all = dataset_class.create_eval_dataset(n_gender=3, n_nat=6, n_age=5)

    # print(eval_dataset_base.shape)
    # print(eval_dataset_gender.shape)
    # print(eval_dataset_nat.shape)
    # print(eval_dataset_age.shape)
    # print(eval_dataset_all.shape)

    hf_dataset_base = Dataset.from_pandas(eval_dataset_base)
    hf_dataset_gender = Dataset.from_pandas(eval_dataset_gender)
    hf_dataset_nat = Dataset.from_pandas(eval_dataset_nat)
    hf_dataset_age = Dataset.from_pandas(eval_dataset_age)

    hf_dats = [
        hf_dataset_base, hf_dataset_gender, hf_dataset_nat, hf_dataset_age
    ]
    file_names = [
        "mentat_data_base", "mentat_data_gender", "mentat_data_nat", "mentat_data_age"
    ]
    for i in range (4):
        save_location = os.path.join(os.getcwd() + "/eval_data", file_names[i])
        hf_dats[i].save_to_disk(save_location)


if __name__ == "__main__":
    main()
