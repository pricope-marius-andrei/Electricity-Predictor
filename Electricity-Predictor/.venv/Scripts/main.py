import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text  # type: ignore
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names")
from sklearn.metrics import mean_squared_error
import math


class Prediction:
    def __init__(self, data, discretization_rules, target_col):
        """
        Initialize the Prediction class.
        Args:
            data (dict): Data dictionary containing features and target variable.
            discretization_rules (dict): Dictionary with column names as keys, and (bins, labels) as values for discretization.
            target_col (str): The target column name for classification.
        """
        self.df = pd.DataFrame(data)
        self.discretization_rules = discretization_rules
        self.target_col = target_col
        self.df_discretized = self.preprocess_data()

    def discretize_column(self, column, bins, labels):
        """
        Discretize a numeric column into buckets.
        Args:
            column (pd.Series): Numeric column to discretize
            bins (list): Bin edges
            labels (list): Labels for each bin
        Returns:
            pd.Series: Discretized column
        """
        return pd.cut(column, bins=bins, labels=labels, include_lowest=True)

    def preprocess_data(self):
        """
        Preprocess data by discretizing numeric columns.
        Returns:
            pd.DataFrame: Discretized data
        """
        df_discretized = self.df.copy()
        for col, (bins, labels) in self.discretization_rules.items():
            df_discretized[col] = self.discretize_column(self.df[col], bins, labels)
        df_discretized[self.target_col] = df_discretized[self.target_col].astype("category")
        return df_discretized

    def naive_bayes_classification(self, train_df, test_row):
        """
        Perform Naive Bayes classification on a discretized dataset.
        Args:
            train_df (pd.DataFrame): Training dataset (discretized)
            test_row (pd.Series): A single test instance (discretized)
        Returns:
            str: Predicted class label for the test instance
        """
        test_row_aligned = test_row.copy()
        for col in train_df.columns:
            if col != self.target_col:
                test_row_aligned[col] = pd.Categorical(
                    [test_row[col]],
                    categories=train_df[col].cat.categories
                )[0]

        priors = train_df[self.target_col].value_counts(normalize=True).to_dict()
        likelihoods = {}

        for cls in train_df[self.target_col].cat.categories:
            cls_df = train_df[train_df[self.target_col] == cls]
            cls_likelihood = 1.0
            for col in train_df.columns:
                if col != self.target_col:
                    value = test_row_aligned[col]
                    value_prob = (
                        cls_df[col].value_counts(normalize=True).get(value, 0)
                    )
                    cls_likelihood *= value_prob
            likelihoods[cls] = cls_likelihood

        posteriors = {cls: priors[cls] * likelihoods[cls] for cls in priors}
        total_prob = sum(posteriors.values())
        normalized_posteriors = {cls: prob / total_prob for cls, prob in posteriors.items()}

        predicted_class = max(normalized_posteriors, key=normalized_posteriors.get)
        return predicted_class

    def id3_classification(self, train_df, test_instance):
        """
        Train an ID3 decision tree classifier and classify a test instance.
        Args:
            train_df (pd.DataFrame): Training dataset (discretized)
            test_instance (pd.Series): Test instance (discretized)
        Returns:
            str: Predicted class label for the test instance
        """
        X_train = train_df.drop(columns=[self.target_col]).apply(lambda x: x.cat.codes)
        y_train = train_df[self.target_col].cat.codes

        model = DecisionTreeClassifier(criterion='entropy', max_depth=None)
        model.fit(X_train, y_train)

        test_instance_encoded = pd.DataFrame([test_instance]).apply(
            lambda x: pd.Categorical(x, categories=train_df[x.name].cat.categories).codes
        ).iloc[0]

        predicted_class_idx = model.predict([test_instance_encoded])[0]
        class_labels = train_df[self.target_col].cat.categories
        predicted_class = class_labels[predicted_class_idx]

        return predicted_class

    def performance_test_mae(self, model_func, test_df, test_data, class_mapping):
        """
        Test the performance of the classification model using MAE.
        Args:
            model_func (function): Classification function to test (e.g., naive_bayes_classification or id3_classification).
            test_df (pd.DataFrame): Test dataset.
            test_data (pd.DataFrame): Ground truth for comparison.
            class_mapping (dict): Mapping of categorical values to numeric representations.
        Returns:
            float: MAE score of the model on the test data.
        """
        predictions = []
        ground_truth = []

        for _, row in test_data.iterrows():
            test_instance = row.drop(self.target_col)
            prediction = model_func(test_df, test_instance)
            predictions.append(class_mapping[prediction])
            ground_truth.append(class_mapping[row[self.target_col]])

        # Calculate MAE using numeric representations
        mae = mean_absolute_error(ground_truth, predictions)
        return mae

    def performance_test_rmse(self, model_func, test_df, test_data, class_mapping):
        """
        Test the performance of the classification model using RMSE.
        Args:
            model_func (function): Classification function to test (e.g., naive_bayes_classification or id3_classification).
            test_df (pd.DataFrame): Test dataset.
            test_data (pd.DataFrame): Ground truth for comparison.
            class_mapping (dict): Mapping of categorical values to numeric representations.
        Returns:
            float: RMSE score of the model on the test data.
        """
        predictions = []
        ground_truth = []

        for _, row in test_data.iterrows():
            test_instance = row.drop(self.target_col)
            prediction = model_func(test_df, test_instance)
            predictions.append(class_mapping[prediction])
            ground_truth.append(class_mapping[row[self.target_col]])

        # Calculate RMSE using numeric representations
        mse = mean_squared_error(ground_truth, predictions)
        rmse = math.sqrt(mse)
        return rmse


def main():
    data = {
        "Consum": [5402, 6753, 8223, 6281, 4808, 5334, 6011, 5847, 7622, 5659, 5374, 7055, 5784],
        "Productie": [5384, 7059, 7574, 6690, 6337, 5045, 5126, 5449, 6073, 5116, 5156, 6308, 5086],
        "Intermitent": [1913, 3012, 2014, 1709, 2301, 1833, 840, 1951, 2123, 2057, 2367, 3903, 2182],
        "Constant": [3471, 4047, 5560, 4981, 4036, 3212, 4286, 3498, 3950, 3458, 2789, 2405, 2904],
        "Sold": [18, -306, 649, -409, -1529, 289, 885, 398, 1549, 144, 218, 747, 698]
    }

    rules = {
        "Consum": ([4500, 5500, 6500, 7500, 8500], ["Mic", "Mediu", "Mare", "Foarte Mare"]),
        "Productie": ([5000, 6000, 7000, 8000], ["Mic", "Mediu", "Mare"]),
        "Intermitent": ([800, 2000, 3200, 4400], ["Mic", "Mediu", "Mare"]),
        "Constant": ([2200, 3400, 4600, 5800], ["Mic", "Mediu", "Mare"]),
        "Sold": ([-np.inf, -500, 500, 1500, np.inf], ["Negativ Mic", "Echilibrat", "Pozitiv Mic", "Pozitiv Mare"])
    }

    class_mapping = {
        "Negativ Mic": 1,
        "Echilibrat": 2,
        "Pozitiv Mic": 3,
        "Pozitiv Mare": 4
    }

    target_col = "Sold"
    prediction_model = Prediction(data, rules, target_col)

    # Example test instance
    test_instance = pd.Series({"Consum": "Foarte Mare", "Productie": "Mare", "Intermitent": "Mare", "Constant": "Mic"})

    train_df, test_df = train_test_split(prediction_model.df_discretized, test_size=0.3, random_state=42)

    # Naive Bayes prediction
    nb_result = prediction_model.naive_bayes_classification(train_df, test_instance)
    print("Naive Bayes Result:", nb_result)

    # ID3 prediction
    id3_result = prediction_model.id3_classification(train_df, test_instance)
    print("ID3 Result:", id3_result)
    print('\n')

    # Performance testing MAE
    accuracy_nb = prediction_model.performance_test_mae(prediction_model.naive_bayes_classification,
                                                        train_df, test_df, class_mapping)
    print(f"Naive Bayes MAE: {accuracy_nb:.2f}")

    accuracy_id3 = prediction_model.performance_test_mae(prediction_model.id3_classification,
                                                         prediction_model.df_discretized, test_df, class_mapping)
    print(f"ID3 MAE: {accuracy_id3:.2f}")

    # Performance testing RMSE
    rmse_nb = prediction_model.performance_test_rmse(prediction_model.naive_bayes_classification,
                                                     train_df, test_df, class_mapping)
    print(f"Naive Bayes RMSE: {rmse_nb:.2f}")

    rmse_id3 = prediction_model.performance_test_rmse(prediction_model.id3_classification,
                                                      prediction_model.df_discretized, test_df, class_mapping)
    print(f"ID3 RMSE: {rmse_id3:.2f}")


if __name__ == "__main__":
    main()
