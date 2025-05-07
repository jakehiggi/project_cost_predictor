import pandas as pd
from model_scripts.transform import validate_columns
from model_scripts.pipeline import create_my_pipeline
from model_scripts.tuning import tune_model
from pathlib import Path
import joblib

current_dir = Path(__file__).resolve().parent
csv_path = current_dir / '..' / 'data' / 'project_data.csv'

def build_and_train(csv_path):
    df = pd.read_csv(csv_path)
    df = validate_columns(df)

    df_encoded = pd.get_dummies(df, columns=['Industry'])
    X = df_encoded.drop('Actual_Cost', axis=1)
    y = df_encoded['Actual_Cost']

    pipe = create_my_pipeline()

    param_dist = {
    'clf__n_estimators': [50, 100, 150, 200],
    'clf__max_depth': [3, 4, 5, 6, 7, 8],
    'clf__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'clf__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'clf__gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'clf__reg_alpha': [0, 0.01, 0.1, 1],
    'clf__reg_lambda': [1, 1.5, 2, 3]
 }

    best_model = tune_model(pipe, param_dist, X, y)
    return best_model

if __name__ == "__main__":
    model = build_and_train(csv_path=csv_path)
    print("Best Parameters:", model.best_params_)
    joblib.dump(model, current_dir / 'best_model.pkl')
