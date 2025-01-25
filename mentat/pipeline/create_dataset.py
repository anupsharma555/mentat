
import os
import pandas as pd
import numpy as np
import pickle

from mentat.pipeline import data_struct
from mentat.config import config_params


class MentatDataSet:
    """"""

    def __init__(self, directory: str, filename: str, remove_bad_q_inds: bool = True):
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
        self._question_dataset = self._overwrite_with_preference_labels(
            self._question_dataset, "hbt"
        )

        

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
            if int(row["q_id"]) in self._inds_bad_post_annotate:
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
        """Generate train adn test split"""

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
        input_df: pd.DataFrame, preference_type: str = "hbt"
    ):
        """Function to import preference values to overwrite creator_truth labels"""

        assert preference_type in ["hbt", "bt"], NotImplementedError(f"preference_type = {preference_type}")

        # Loading annotation analysis results
        with open('analysis_results.pkl', 'rb') as f:
            loaded_object = pickle.load(f)
        hbt_scores, hbt_scores_params, bt_scores, bt_scores_typed, means_and_alphas = loaded_object 

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


# todo: write function to turn questions into prompt and set modifiers + randomzies quesiton

# todo: write function to store dataset to a file

# todo: for evals, use accuracy

# final column names
        # prompt labels gender nat age q_id category split

def main():
    dataset_class = MentatDataSet(os.getcwd(), "questions_final.csv")
    question_dataset = dataset_class.question_dataset

    mask = question_dataset["q_id"] == 30
    for k in question_dataset.keys():
        print(k, question_dataset[mask][k])


    mask = question_dataset["q_id"] == 32 
    for k in question_dataset.keys():
        print(k, question_dataset[mask][k])
    

if __name__ == "__main__":
    main()


"""
What is the baseline?
Each question once or three times, but for a different (gender, age, nationality) pair?

Task-specific accuracy and Impact of demographic information at the same time!
--> separate dimensions or create one massive dataset and filter? 
    use all possible nationalities, all genders, 20 ages (uniform) 
    --> big DF and filter for different evals? probably

    is size an issue? 200 * 3 * 20 * 10

add: you can adjust MC sampling for local distirbutions
add: pre-training key

Free-form consistency

White
African American
Black
Hispanic
Asian
Native American

Male
Female
Nonbinary

10 Ages

    200 * 3 * 6 * 10 = 

    Do base accuracy? take one of each question and just run the same set for all?

"""