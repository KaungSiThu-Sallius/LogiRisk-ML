import pandas as pd
import time
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import optuna
from preprocess import preprocess
from preprocess import train_test_split_encode

dataset = 'DataCoSupplyChainDataset.csv'
df = preprocess(dataset)
X_train, X_test, y_train, y_test = train_test_split_encode(df)

mlflow.set_tracking_uri("http://127.0.0.1:2020/")
time.sleep(15)
mlflow.set_experiment("LogiRisk MLflow using optuna")

def rf_objective(trial):
    with mlflow.start_run(run_name=f"Random Forest trial {trial.number}", nested=True):
        # Hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=1111
        )
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        mlflow.log_params(trial.params)
        mlflow.log_metric("mse", mse)

        return mse
    

with mlflow.start_run(run_name="RF_Optimization"):
    study = optuna.create_study(direction='minimize')
    study.optimize(rf_objective, n_trials=20)

    print(f"Best trial: {study.best_trial.params}")

    mlflow.log_param(study.best_params)
    mlflow.log_metric("best_mese", study.best_value)

    best_rf = RandomForestClassifier(**study.best_params)
    best_rf.fit(X_train, y_train)

    signature = infer_signature(X_test, best_rf.predict(X_test))

    mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="best_rf_model",
        signature=signature,
        input_example=X_test[:3],
        registered_model_name="Random_Forest_Classifier"
    )

print("Optimization for Random Forest Complete.")