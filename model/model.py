import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap
import joblib
from pathlib import Path

current_dir = Path(__file__).resolve(strict=True).parent
csv_path = current_dir / '..' / 'data' / 'project_data.csv'
df = pd.read_csv(csv_path)

# Encode categorical variables (industry types)
df_encoded = pd.get_dummies(df, columns=['Industry'])

# Define X (features) and y (target)
X = df_encoded.drop(['Project_ID', 'Actual_Cost'], axis=1)
y = df_encoded['Actual_Cost']

# Train-test split to validate model performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 3]
}
# Initialize XGBoost model
xgb_model = XGBRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of random parameter combinations tested
    scoring='neg_mean_absolute_error',  # Optimizing for MAE
    cv=5,  # 5-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1  # Use all available CPU cores
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate clearly
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Tuned Model MAE: {mae:.2f}')
print(f'Tuned Model R²: {r2:.2f}')
print("Best Hyperparameters:", random_search.best_params_)

# Save the model
joblib.dump(best_model, 'xgb_actual_cost_model.pkl')

# Interpretability using SHAP values to explain predictions clearly
explainer = shap.Explainer(best_model)
shap_values = explainer(X)
shap.summary_plot(shap_values, X)
