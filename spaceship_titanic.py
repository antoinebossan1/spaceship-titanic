import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import yaml
from logzero import logger


class TitanicSpaceship:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Load the training data using the configuration file
        self.train_data = pd.read_csv(
            self.config["input"]["train"]["filepath"],
            usecols=self.config["input"]["train"]["usecols"].keys(),
            dtype=self.config["input"]["train"]["usecols"],
        )

        # Load the test data using the configuration file
        self.test_data = pd.read_csv(
            self.config["input"]["test"]["filepath"],
            usecols=self.config["input"]["test"]["usecols"].keys(),
            dtype=self.config["input"]["test"]["usecols"],
        )

    def pre_processing(self):
        self.train_data["train_test"] = 1
        self.test_data["train_test"] = 0
        # Concatenating the training and test data into a single dataframe
        self.all_data = pd.concat([self.train_data, self.test_data], sort=False)

        # Dropping the "PassengerId" and "Name" columns
        self.all_data.drop(["PassengerId", "Name"], axis=1, inplace=True)

        # Splitting the "Cabin" column into 3 columns
        self.all_data[["Cabin_1st", "Cabin_2nd", "Cabin_3rd"]] = self.all_data[
            "Cabin"
        ].str.split("/", expand=True)
        # Dropping the original "Cabin" column
        self.all_data.drop("Cabin", axis=1, inplace=True)

        # Separating the categorical and numerical columns
        categorical_cols = self.all_data.select_dtypes(include=object).columns
        numerical_cols = self.all_data.select_dtypes(exclude=object).columns

        # Fill missing values in categorical columns with the mode
        for col in categorical_cols:
            self.all_data[col].fillna(self.all_data[col].mode()[0], inplace=True)

        # Fill missing values in numerical columns with the mean
        for col in numerical_cols:
            self.all_data[col].fillna(self.all_data[col].mean(), inplace=True)

        # Convert the CryoSleep column to binary values (0 or 1)
        self.all_data["CryoSleep"] = self.all_data["CryoSleep"].map(
            {"False": 0, "True": 1}
        )

        # Create dummies for categorical columns
        all_data = self.all_data.copy()
        for column in all_data.select_dtypes(include=object).columns:
            if column != "Transported":
                dummies = pd.get_dummies(self.all_data[column])
                self.all_data = pd.concat((self.all_data, dummies), axis=1)
                self.all_data.drop(column, axis=1, inplace=True)

        self.train_data = self.all_data[self.all_data["train_test"] == 1].drop(
            "train_test", axis=1
        )
        self.test_data = self.all_data[self.all_data["train_test"] == 0].drop(
            ["train_test", "Transported"], axis=1
        )

        self.X = self.train_data.drop("Transported", axis=1)
        self.y = self.train_data["Transported"]

        self.y = self.y.map({"False": 0, "True": 1})

        # Split the training data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.33, random_state=0
        )

    def fit_and_evaluate_models(self):

        # List of models to be evaluated
        models = [
            RandomForestClassifier(random_state=1),
            SVC(random_state=1),
            KNeighborsClassifier(),
            LogisticRegression(random_state=1),
        ]
        model_names = ["Random Forest", "SVC", "KNN", "Logistic Regression", "XGB"]
        accuracy_scores = {}
        accuracy_scores = {}
        for model, model_name in zip(models, model_names):
            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_val)

            accuracy = accuracy_score(self.y_val, y_pred)

            accuracy_scores[model_name] = accuracy

            print(f"Accuracy of {model_name}: {accuracy}")

        # Find the name of the best model based on the highest accuracy score
        best_model_name = max(accuracy_scores, key=accuracy_scores.get)

        print(f"Best Model: {best_model_name}")

    # Best Model: Random Forest

    def model_tuning(self):
        param_grid = {
            "n_estimators": [100, 500, 1000],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", "log2", None],
        }

        rf_clf = RandomForestClassifier(random_state=1)
        grid_search = GridSearchCV(
            rf_clf, param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=1
        )

        # Fit the grid search to the training data
        grid_search.fit(self.X_train, self.y_train)

        # Get the best parameters found by grid search
        best_params = grid_search.best_params_

        best_rf_clf = RandomForestClassifier(random_state=1, **best_params)

        # Fit the best Random Forest Classifier to the training data
        best_rf_clf.fit(self.X_train, self.y_train)

        # Make predictions on the validation set
        y_pred = best_rf_clf.predict(self.X_val)

        test_acc = accuracy_score(self.y_val, y_pred)
        print("Test accuracy:", test_acc)

        # Make predictions on the test data using the best Random Forest classifier
        self.y_pred = best_rf_clf.predict(self.test_data)

        # Load the passenger IDs from the test file
        df_submission = pd.read_csv(
            self.config["input"]["test"]["filepath"], usecols=["PassengerId"]
        )

        # Add the predicted "Transported" values to the data frame
        df_submission["Transported"] = self.y_pred.astype(bool)

        df_submission.to_csv(self.config["output"], index=False)

        DEBUGVAL__ = df_submission.head()
        logger.debug(f"\n {DEBUGVAL__}")


if __name__ == "__main__":
    spaceship = TitanicSpaceship(
        "/Users/atnbsn/Desktop/VSCode/Machine_Learning/data/spaceship_titanic/spaceship_titanic.yaml"
    )
    spaceship.pre_processing()
    spaceship.fit_and_evaluate_models()
    spaceship.model_tuning()
    spaceship.predict()
