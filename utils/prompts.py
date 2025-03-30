import pandas as pd
from typing import List, Dict
import string
import re
import logging
import traceback

def format_choices(choices: List[str]) -> str:
    labeled_choices = [f"{letter}. {option}" for letter, option in zip(string.ascii_uppercase, choices)]
    return "\n".join(labeled_choices)

def clean_question(question):
    """
    Cleans the question string by removing answer choices enclosed in square brackets
    and the trailing 'Answer: ' substring.

    Parameters:
    question (str): The original question string.

    Returns:
    str: The cleaned question string.
    """
    # Remove the answer choices enclosed in square brackets
    question = re.sub(r"\[\s*'.*?'\s*(,\s*'.*?'\s*)*\]", "", question)
    # Remove the trailing 'Answer: ' substring
    question = re.sub(r"Answer:\s*$", "", question)
    # Strip any extraneous whitespace
    return question.strip()

def get_prompt(
    df_train: pd.DataFrame,
    question_ids: List[int],
    question: str,
    choices: List[str]
) -> str:
    prompt: str = ""
    for qid in question_ids:
        prompt_question: str = df_train.loc[qid]["question"]
        prompt_choices: str = format_choices(df_train.loc[qid]["choices"])
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
    question = clean_question(question)
    choices = format_choices(choices)
    prompt += f"{question}\n{choices}\nAnswer:\n"
    return prompt

def get_prompt_chat(
    df_train: pd.DataFrame,
    question_ids: List[int],
    question: str,
    choices: List[str]
) -> List[Dict[str, str]]:
    conversation: List[Dict[str, str]] = []
    for qid in question_ids:
        turn_user = {"role": "user", "content": ""}
        turn_assistant = {"role": "assistant", "content": ""}
        prompt_question: str = df_train.loc[qid]["question"]
        prompt_choices: str = format_choices(df_train.loc[qid]["choices"])
        turn_user["content"] = f"{prompt_question}\n{prompt_choices}"
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
        turn_assistant["content"] = f"{answer}\n"
        conversation.append(turn_user)
        conversation.append(turn_assistant)
        
    question = clean_question(question)
    choices = format_choices(choices)
    conversation.append(
        {
            "role": "user",
            "content": f"{question}\n{choices}"
        }
    )
    return conversation