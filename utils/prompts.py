import pandas as pd
from typing import List


def get_prompt(
    df_train: pd.DataFrame,
    question_ids: List[int],
    question: str,
    choices: List[str]
) -> str:
    prompt: str = ""
    for qid in question_ids:
        prompt_question: str = df_train.iloc[qid]["question"]
        prompt_choices: str = df_train.iloc[qid]["choices"]
        answer: str = df_train.iloc[qid]["answer"].strip()
        if answer == "" or answer[-1] not in ['A', 'B', 'C', 'D']:
            assert False, f"Invalid answer in train dataset: question id {qid}"
        answer = answer[-1]
        prompt += f"{prompt_question}\n{prompt_choices}\nAnswer:\n{answer}\n"
    prompt += f"{question}\n"
    return prompt