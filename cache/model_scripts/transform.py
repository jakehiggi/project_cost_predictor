REQUIRED_COLUMNS = ['Project_ID', 'Industry', 'Initial_Budget', 'Project_Duration', 'Sustainability_Score', 'Client_Satisfaction', 'Resource_Usage_Index', 'Actual_Cost']

def validate_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    return df