import pandas as pd
from typing import List
import logging
import traceback

def get_prompt(
    df_train: pd.DataFrame,
    question_ids: List[int],
    question: str,
    choices: List[str]
) -> str:
    prompt: str = ""
    for qid in question_ids:
        prompt_question: str = df_train.loc[qid]["question"]
        prompt_choices: str = df_train.loc[qid]["choices"]
        try:
            answer: str = df_train.loc[qid]["answer"].strip()
        except Exception as e:
            logging.info(f"Task id: {df_train.loc[qid]['task_id']}")
            logging.info(f"Question: {prompt_question}")
            logging.info(f"Choices: {prompt_choices}")
            logging.info(f"Answer: {df_train.loc[qid]['answer']}")
            logging.error(traceback.format_exc())
        if answer == "" or not answer[-1].isalpha():
            assert False, f"Invalid answer in train dataset: question id {qid}"
        answer = answer[-1]
        prompt += f"{prompt_question}\n{prompt_choices}\nAnswer:\n{answer}\n"
    prompt += f"{question}\n"
    return prompt