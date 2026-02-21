import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and testing data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            params = {

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10, 20]
                },

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5]
                },

                "Linear Regression": {},

                "KNeighbors": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5]
                },

                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [100, 200]
                },

                "AdaBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                }
            }

            best_model = None
            best_model_score = float("-inf")
            best_model_name = None

            # 🔥 Hyperparameter tuning loop
            for model_name, model in models.items():

                logging.info(f"Tuning model: {model_name}")

                param_grid = params[model_name]

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )

                grid_search.fit(X_train, y_train)

                tuned_model = grid_search.best_estimator_

                y_pred = tuned_model.predict(X_test)
                score = r2_score(y_test, y_pred)

                logging.info(f"{model_name} R2 Score: {score}")

                if score > best_model_score:
                    best_model_score = score
                    best_model = tuned_model
                    best_model_name = model_name

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(f"Best Model Selected: {best_model_name}")
            logging.info(f"Best Model R2 Score: {best_model_score}")

            # Save best tuned model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)