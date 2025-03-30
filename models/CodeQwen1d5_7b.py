import sys
sys.path.append(".")
from utils.prompts import get_prompt

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd


def run_model1(df_train, df_val):
    codeqwen1d5_name = "Qwen/CodeQwen1.5-7B"
    codeqwen1d5_path = 'checkpoints'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    codeqwen1d5 = AutoModelForCausalLM.from_pretrained(
        codeqwen1d5_path,
        device_map="auto",
        trust_remote_code=True
    )

    codeqwen1d5_tokenizer = AutoTokenizer.from_pretrained(
        codeqwen1d5_path,
        trust_remote_code=True
    )

    generated_answers = []
    results = []

    for index, row in df_val.iterrows():
        print(f"Processing question {index}...")
        prompt = get_prompt(
            df_train = df_train,
            question_ids = [1, 2, 3],
            question = row['question'], 
            choices = row['choices']
        )
        inputs = codeqwen1d5_tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)

        outputs = codeqwen1d5.generate(**inputs, max_new_tokens=1)
        answer_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        generated_answer = codeqwen1d5_tokenizer.decode(answer_tokens[0], skip_special_tokens=True).strip()
        if generated_answer == "":
            generated_answer = "C" # with a minority of empty answer => answer: C

        generated_answers.append(generated_answer)

        results.append({
            'task_id': row['task_id'],
            'answer': generated_answer
        })

    # write results to a csv
    df_results = pd.DataFrame(results)
    df_results.to_csv('submission/codeqwen1d5_7b_submission.csv', index=False)