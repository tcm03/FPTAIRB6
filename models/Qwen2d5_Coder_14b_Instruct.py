import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
from utils.prompts import get_prompt, get_prompt_chat
from utils.constants import RANDOM_STATE

def run_model5(df_train, df_val):
    model_path = "checkpoints/Qwen2.5-Coder-14B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the instruction-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    generated_answers = []
    results = []
    num_empties = 0
    num_tries = 5

    # Select random examples from the training set for context
    random_example = df_train.sample(n=1, random_state=RANDOM_STATE)
    another_random_example = df_train.sample(n=1, random_state=2*RANDOM_STATE)
    print(f"Random example selected:\n{random_example}\n")
    print(f"Another random example selected:\n{another_random_example}\n")

    for index, row in df_val.iterrows():
        print(f"Processing question {index}...")

        # Construct the prompt using the chat template
        prompt_chat = get_prompt_chat(
            df_train=df_train,
            question_ids=[random_example.index[0], another_random_example.index[0]],
            question=row['question'],
            choices=row['choices']
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant who is proficient at computer science and coding."},
        ] + prompt_chat
        chat_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print(f"\nChat input:\n{chat_input}\n")

        inputs = tokenizer(chat_input, return_tensors="pt").to(device)

        outputs = model.generate(**inputs, max_new_tokens=1)
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]  # Extract only the new tokens
        generated_answer = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

        # print(f'Gen answer:\n{generated_answer}\n')
        if not generated_answer or not generated_answer[-1].isalpha() or (len(generated_answer) >= 2 and generated_answer[-1].isalpha() and generated_answer[-2].isalpha()):
            num_empties += 1
            generated_answer = "C"  # Default answer for empty or invalid responses

        generated_answers.append(generated_answer)
        results.append({
            'task_id': row['task_id'],
            'answer': generated_answer
        })

        # num_tries -= 1
        # if num_tries == 0:
        #     break

    print(f"\nFinished inference, {num_empties} empty answers were set to 'C'.\n")

    # Save results to a CSV file
    df_results = pd.DataFrame(results)
    df_results.to_csv('outputs/qwen2d5_14b_instruct_2shot.csv', index=False)
