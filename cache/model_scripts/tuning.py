from sklearn.model_selection import RandomizedSearchCV

def tune_model(pipeline, param_distributions, X, y, cv=5, n_iter=30, scoring='accuracy'):
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=25,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X, y)
    return search
