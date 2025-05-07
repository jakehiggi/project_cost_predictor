from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def create_my_pipeline():
    model = XGBClassifier(random_state=42)
    pipeline = Pipeline([
        ('clf', model)
    ])
    return pipeline