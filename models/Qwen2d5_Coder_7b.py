import sys
sys.path.append(".")
from utils.prompts import get_prompt
from utils.constants import RANDOM_STATE

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd


def run_model3(df_train, df_val):
    codeqwen2d5_name = "Qwen/Qwen2.5-Coder-7B"
    codeqwen2d5_path = "checkpoints/Qwen2.5-Coder-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    codeqwen2d5 = AutoModelForCausalLM.from_pretrained(
        codeqwen2d5_path,
        device_map="auto",
        trust_remote_code=True
    )

    codeqwen2d5_tokenizer = AutoTokenizer.from_pretrained(
        codeqwen2d5_path,
        trust_remote_code=True
    )

    generated_answers = []
    results = []
    num_empties = 0
    # num_tries = 5
    random_example = df_train.sample(n=1, random_state=RANDOM_STATE)
    print(f"Random example selected:\n{random_example}\n")
    for index, row in df_val.iterrows():
        print(f"Processing question {index}...")
        prompt = get_prompt(
            df_train = df_train,
            question_ids = [random_example.index[0]],
            question = row['question'], 
            choices = row['choices']
        )
        # print(f'Constructed prompt:\n{prompt}')
        inputs = codeqwen2d5_tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)

        outputs = codeqwen2d5.generate(**inputs, max_new_tokens=1)
        answer_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        generated_answer = codeqwen2d5_tokenizer.decode(answer_tokens[0], skip_special_tokens=True).strip()
        # print(f'Gen answer:\n{generated_answer}')
        if generated_answer == "":
            num_empties += 1
            generated_answer = "C" # with a minority of empty answer => answer: C

        generated_answers.append(generated_answer)

        results.append({
            'task_id': row['task_id'],
            'answer': generated_answer
        })

        # num_tries -= 1
        # if num_tries == 0:
        #     break
        # break

    print(f"\nFinished inference, {num_empties} empty answers are set to C.\n")
    # write results to a csv
    df_results = pd.DataFrame(results)
    df_results.to_csv('outputs/codeqwen2d5_14b_3shot.csv', index=False)