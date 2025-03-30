import pandas as pd
import ast
from models.CodeQwen1d5_7b import run_model1
from models.Qwen2d5_Coder_14b import run_model2
from models.Qwen2d5_Coder_7b import run_model3

if __name__ == "__main__":
    df_train = pd.read_csv('data/b6_train_data.csv')
    # Ensure the 'choices' column is properly parsed as a list
    df_train['choices'] = df_train['choices'].apply(ast.literal_eval)
    df_val = pd.read_csv('data/b6_test_data.csv')
    df_val['choices'] = df_val['choices'].apply(ast.literal_eval)

    # run_model1(df_train, df_val)
    # run_model2(df_train, df_val)
    run_model3(df_train, df_val)