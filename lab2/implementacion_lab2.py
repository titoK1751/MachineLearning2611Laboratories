from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


TARGET = "temp_max_manana"
RANDOM_STATE = 42


def _preprocessor(dataframe, polynomial_degree=None):
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = dataframe.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if polynomial_degree is not None:
        numeric_steps.append(
            ("polynomial", PolynomialFeatures(degree=polynomial_degree, include_bias=False))
        )
    numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def _model_searches(dataframe):
    plain_preprocessor = _preprocessor(dataframe)
    polynomial_preprocessor = _preprocessor(dataframe, polynomial_degree=2)

    return {
        "LinearRegression": GridSearchCV(
            Pipeline(
                [("preprocessor", plain_preprocessor), ("model", LinearRegression())]
            ),
            {"model__fit_intercept": [True, False]},
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        ),
        "Ridge": GridSearchCV(
            Pipeline([("preprocessor", plain_preprocessor), ("model", Ridge())]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        ),
        "Lasso": GridSearchCV(
            Pipeline(
                [
                    ("preprocessor", plain_preprocessor),
                    ("model", Lasso(max_iter=20000, random_state=RANDOM_STATE)),
                ]
            ),
            {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        ),
        "PolynomialRidge": GridSearchCV(
            Pipeline(
                [("preprocessor", polynomial_preprocessor), ("model", Ridge())]
            ),
            {
                "preprocessor__numeric__polynomial__degree": [1, 2],
                "model__alpha": [0.1, 1.0, 10.0, 100.0],
            },
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        ),
    }


def _metrics(model, features, target):
    predictions = model.predict(features)
    return {
        "rmse": float(np.sqrt(mean_squared_error(target, predictions))),
        "mae": float(mean_absolute_error(target, predictions)),
        "r2": float(r2_score(target, predictions)),
    }


def _validation_curve(features, target):
    rows = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for degree in [1, 2, 3]:
        pipeline = Pipeline(
            [
                ("preprocessor", _preprocessor(features, polynomial_degree=degree)),
                ("model", Ridge(alpha=10.0)),
            ]
        )
        fold_train_errors = []
        fold_validation_errors = []
        for train_indices, validation_indices in splitter.split(features):
            pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
            fold_train_errors.append(
                np.sqrt(
                    mean_squared_error(
                        target.iloc[train_indices],
                        pipeline.predict(features.iloc[train_indices]),
                    )
                )
            )
            fold_validation_errors.append(
                np.sqrt(
                    mean_squared_error(
                        target.iloc[validation_indices],
                        pipeline.predict(features.iloc[validation_indices]),
                    )
                )
            )
        rows.append(
            {
                "degree": degree,
                "train_rmse_mean": float(np.mean(fold_train_errors)),
                "train_rmse_std": float(np.std(fold_train_errors)),
                "validation_rmse_mean": float(np.mean(fold_validation_errors)),
                "validation_rmse_std": float(np.std(fold_validation_errors)),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_confidence_intervals(model, features, target, repetitions=500):
    rng = np.random.default_rng(RANDOM_STATE)
    predictions = model.predict(features)
    metric_values = {"rmse": [], "mae": [], "r2": []}
    sample_size = len(target)

    for _ in range(repetitions):
        indices = rng.integers(0, sample_size, size=sample_size)
        sampled_target = target.iloc[indices]
        sampled_predictions = predictions[indices]
        metric_values["rmse"].append(
            float(np.sqrt(mean_squared_error(sampled_target, sampled_predictions)))
        )
        metric_values["mae"].append(
            float(mean_absolute_error(sampled_target, sampled_predictions))
        )
        metric_values["r2"].append(float(r2_score(sampled_target, sampled_predictions)))

    return {
        metric: {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }
        for metric, values in metric_values.items()
    }


def run_implementation(dataset_path):
    data = pd.read_csv(Path(dataset_path))
    if TARGET not in data.columns:
        raise ValueError(f"No se encontró la variable objetivo '{TARGET}'.")

    data = data.dropna(subset=[TARGET]).reset_index(drop=True)
    features = data.drop(columns=[TARGET, "fecha"], errors="ignore")
    target = pd.to_numeric(data[TARGET])
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    searches = _model_searches(train_features)
    comparison_rows = []
    for model_name, search in searches.items():
        search.fit(train_features, train_target)
        comparison_rows.append(
            {
                "model": model_name,
                "cv_rmse_mean": float(-search.best_score_),
                "cv_rmse_std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "best_params": search.best_params_,
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    best_model_name = comparison.iloc[0]["model"]
    best_model = searches[best_model_name].best_estimator_
    test_metrics = _metrics(best_model, test_features, test_target)

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "comparison": comparison,
        "validation_curve": _validation_curve(train_features, train_target),
        "test_metrics": test_metrics,
        "bootstrap_ci": _bootstrap_confidence_intervals(
            best_model, test_features, test_target
        ),
    }