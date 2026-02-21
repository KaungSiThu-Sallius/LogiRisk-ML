import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import optuna
from preprocess import preprocess
from preprocess import train_test_split_encode

dataset = 'DataCoSupplyChainDataset.csv'
df = preprocess(dataset)
X_train, X_test, y_train, y_test = train_test_split_encode(df)

mlflow.set_tracking_uri("http://127.0.0.1:5050")

mlflow.set_experiment("LogiRisk MLflow using optuna")

def rf_objective(trial):
    with mlflow.start_run(run_name=f"Random Forest trial {trial.number}", nested=True):
        # Hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=1111
        )
        
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1_weighted')

        mlflow.log_params(trial.params)
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())

        return cv_scores.mean()
    
def xgb_objective(trial):
    with mlflow.start_run(run_name=f"XGB trial {trial.number}", nested=True):
        # Hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1)
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 10)

        xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
        
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='f1_weighted')

        mlflow.log_params(trial.params)
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())

        return cv_scores.mean()
    
with mlflow.start_run(run_name="RF_Optimization"):
    study = optuna.create_study(direction='maximize')
    study.optimize(rf_objective, n_trials=20)

    best_rf = RandomForestClassifier(**study.best_params, random_state=1111)
    best_rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, best_rf.predict(X_train))
    test_acc = accuracy_score(y_test, best_rf.predict(X_test))
    train_f1 = f1_score(y_train, best_rf.predict(X_train), average='weighted')
    test_f1 = f1_score(y_test, best_rf.predict(X_test), average='weighted')

    mlflow.log_params(study.best_params)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("train_f1", train_f1)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("overfit_gap", train_acc - test_acc)

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Overfit Gap    : {train_acc - test_acc:.4f}")

    input_example = X_test.iloc[[0]]
    signature = infer_signature(input_example, best_rf.predict(input_example))

    mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="best_rf_model",
        signature=signature,
        input_example=input_example,
        registered_model_name="Random_Forest_Classifier"
    )
    
with mlflow.start_run(run_name="XGB_Optimization"):
    study = optuna.create_study(direction='maximize')
    study.optimize(xgb_objective, n_trials=20)

    best_xgb = XGBClassifier(**study.best_params, random_state=1111)
    best_xgb.fit(X_train, y_train)

    print(f"Best trial: {study.best_trial.params}")

    X_train_pred = best_xgb.predict(X_train)
    X_test_pred = best_xgb.predict(X_test)

    train_acc = accuracy_score(y_train, X_train_pred)
    test_acc = accuracy_score(y_test, X_test_pred)
    train_f1 = f1_score(y_train, X_train_pred, average='weighted')
    test_f1 = f1_score(y_test, X_test_pred, average='weighted')

    mlflow.log_params(study.best_params)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("train_f1", train_f1)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("overfit_gap", train_acc- test_acc)

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Overfit Gap    : {train_acc - test_acc:.4f}")
    
    input_example = X_test.iloc[[0]]
    signature = infer_signature(input_example, best_xgb.predict(input_example))

    mlflow.xgboost.log_model(
        xgb_model=best_xgb,
        artifact_path="best_xgboost_model",
        signature=signature,
        input_example=input_example,
        registered_model_name="XGB_Classifier"
    )

print("Optimization for Random Forest Complete.")