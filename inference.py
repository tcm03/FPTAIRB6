import pandas as pd
import ast
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

from models.CodeQwen1d5_7b import run_model1
from models.Qwen2d5_Coder_14b import run_model2
from models.Qwen2d5_Coder_7b import run_model3
from models.Qwen2d5_Coder_7b_Instruct import run_model4

if __name__ == "__main__":
    df_train = pd.read_csv('data/b6_train_data.csv')
    print(f"Raw train set: {df_train.shape[0]}")
    # Ensure the 'choices' column is properly parsed as a list
    df_train['choices'] = df_train['choices'].apply(ast.literal_eval)
    # Filter out rows from df_train where the 'answer' column is empty or null
    df_train = df_train[df_train['answer'].notna() & (df_train['answer'] != '')]
    df_train = df_train.reset_index(drop=True)
    print(f"Filtered train set: {df_train.shape[0]}")
    df_val = pd.read_csv('data/b6_test_data.csv')
    df_val['choices'] = df_val['choices'].apply(ast.literal_eval)

    # run_model1(df_train, df_val)
    # run_model2(df_train, df_val)
    # run_model3(df_train, df_val)
    run_model4(df_train, df_val)