import os
import gc
import joblib
import numpy as np
import shap
import lime
import lime.lime_tabular

# Import Phase 2 config
import config


class SepsisExplainer:
    """
    Explainable AI manager for the Sepsis Prediction system.
    Provides local explanations using SHAP and LIME.
    """

    def __init__(self, use_background_data=False):

        # 1. Load preprocessor
        print(f"Loading preprocessor from {config.PREPROCESSOR_PATH}...")

        if not os.path.exists(config.PREPROCESSOR_PATH):
            raise FileNotFoundError(
                f"Preprocessor not found at {config.PREPROCESSOR_PATH}"
            )

        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)

        # 2. Load model metadata
        print(f"Loading best model info from {config.BEST_MODEL_INFO_PATH}...")

        if not os.path.exists(config.BEST_MODEL_INFO_PATH):
            raise FileNotFoundError(
                f"Best model info not found at {config.BEST_MODEL_INFO_PATH}"
            )

        best_info = joblib.load(config.BEST_MODEL_INFO_PATH)

        self.model_name = best_info["metadata"]["model_name"]
        self.tuned_threshold = float(
            best_info["metadata"]["tuned_threshold"]
        )

        # 3. Load XGBoost model
        model_path = os.path.join(
            config.MODELS_DIR,
            "xgboost.joblib"
        )

        print(f"Loading XGBoost model from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"XGBoost model not found at {model_path}"
            )

        self.model = joblib.load(model_path)

        # 4. Load feature names
        feature_names_path = os.path.join(
            config.PREPROCESSING_DIR,
            "feature_names.joblib"
        )

        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(
                f"Feature names file not found at {feature_names_path}"
            )

        self.feature_names = joblib.load(feature_names_path)

        # 5. Initialize SHAP without background data
        # This avoids loading a separate SHAP background dataset.
        print("Initializing SHAP TreeExplainer...")

        self.background_data = None
        self.shap_explainer = shap.TreeExplainer(self.model)

        # 6. Initialize LIME with a smaller training sample
        self.lime_explainer = None

        if os.path.exists(config.X_TRAIN_PATH):

            print("Loading training data for LIME...")

            X_train = joblib.load(config.X_TRAIN_PATH)

            # Convert to NumPy if the saved object is a DataFrame.
            if hasattr(X_train, "to_numpy"):
                X_train = X_train.to_numpy()

            X_train = np.asarray(X_train)

            if X_train.ndim != 2:
                raise ValueError(
                    "Training data must be a 2D array."
                )

            if X_train.shape[1] != len(self.feature_names):
                raise ValueError(
                    "Feature count in training data does not match "
                    "the feature names."
                )

            if X_train.shape[0] > 0:

                np.random.seed(config.RANDOM_SEED)

                sample_size = min(100, X_train.shape[0])

                indices = np.random.choice(
                    X_train.shape[0],
                    sample_size,
                    replace=False
                )

                lime_training_data = X_train[indices]

                self.lime_explainer = (
                    lime.lime_tabular.LimeTabularExplainer(
                        training_data=lime_training_data,
                        feature_names=list(self.feature_names),
                        class_names=["No Sepsis", "Sepsis"],
                        mode="classification",
                        random_state=config.RANDOM_SEED
                    )
                )

                del lime_training_data
                del indices

            del X_train
            gc.collect()

        else:
            print(
                "Warning: Training data not found. "
                "LIME explanations will be unavailable."
            )

        print("SepsisExplainer successfully initialized!")

    def _predict_probabilities(self, X):
        """Return probabilities for both classes."""

        probabilities = np.asarray(
            self.model.predict_proba(X)
        )

        if probabilities.ndim == 1:
            p = probabilities
            return np.column_stack((1 - p, p))

        if probabilities.shape[1] == 1:
            p = probabilities[:, 0]
            return np.column_stack((1 - p, p))

        return probabilities

    def explain_patient_hour(self, preprocessed_row):
        """
        Generate prediction, SHAP and LIME explanations
        for one preprocessed patient observation.
        """

        row_1d = np.asarray(
            preprocessed_row
        ).squeeze()

        if row_1d.ndim != 1:
            raise ValueError(
                "preprocessed_row must represent a single observation."
            )

        if len(row_1d) != len(self.feature_names):
            raise ValueError(
                "Input feature count does not match feature names."
            )

        row_2d = row_1d.reshape(1, -1)

        # Prediction
        probabilities = self._predict_probabilities(row_2d)

        prob = float(probabilities[0, 1])

        pred_class = int(
            prob >= self.tuned_threshold
        )

        # SHAP explanation
        shap_values = self.shap_explainer.shap_values(row_2d)

        if isinstance(shap_values, list):
            shap_values = (
                shap_values[1]
                if len(shap_values) == 2
                else shap_values[0]
            )

        shap_values = np.asarray(shap_values).squeeze()

        if shap_values.ndim != 1:
            raise ValueError(
                "Unexpected SHAP output shape."
            )

        if len(shap_values) != len(self.feature_names):
            raise ValueError(
                "SHAP output feature count does not match feature names."
            )

        # SHAP base value
        base_value = self.shap_explainer.expected_value

        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.asarray(base_value).squeeze()

            if base_value.ndim > 0:
                base_value = (
                    base_value[1]
                    if len(base_value) == 2
                    else base_value[0]
                )

        base_value = float(base_value)

        shap_contributions = []

        for name, value, impact in zip(
            self.feature_names,
            row_1d,
            shap_values
        ):
            shap_contributions.append({
                "feature": str(name),
                "value": float(value),
                "impact": float(impact),
                "importance": float(abs(impact))
            })

        shap_contributions.sort(
            key=lambda item: item["importance"],
            reverse=True
        )

        # LIME explanation
        lime_contributions = []
        lime_intercept = 0.0

        if self.lime_explainer is not None:

            explanation = self.lime_explainer.explain_instance(
                data_row=row_1d,
                predict_fn=self._predict_probabilities,
                labels=(1,),
                num_features=min(10, len(self.feature_names)),
                num_samples=500
            )

            lime_intercept = float(
                explanation.intercept.get(1, 0.0)
            )

            local_exp = explanation.local_exp.get(1, [])

            for feature_index, weight in local_exp:

                if not 0 <= feature_index < len(self.feature_names):
                    continue

                lime_contributions.append({
                    "feature": str(
                        self.feature_names[feature_index]
                    ),
                    "value": float(row_1d[feature_index]),
                    "impact": float(weight),
                    "importance": float(abs(weight))
                })

            lime_contributions.sort(
                key=lambda item: item["importance"],
                reverse=True
            )

        return {
            "prediction": {
                "probability": prob,
                "prediction_class": pred_class,
                "tuned_threshold": self.tuned_threshold
            },
            "shap": {
                "base_value": base_value,
                "prediction_value": float(
                    base_value + np.sum(shap_values)
                ),
                "contributions": shap_contributions
            },
            "lime": {
                "intercept": lime_intercept,
                "contributions": lime_contributions
            }
        }

    def explain_instance_shap(self, preprocessed_row):
        """
        Return a SHAP Explanation object for one observation.
        Useful for waterfall plots.
        """

        row_2d = np.asarray(
            preprocessed_row
        ).reshape(1, -1)

        explanation = self.shap_explainer(row_2d)

        if len(explanation.shape) == 3:
            explanation = explanation[:, :, 1]

        return explanation[0]

    def explain_instance_lime_obj(self, preprocessed_row):
        """
        Return a LIME explanation object for one observation.
        """

        if self.lime_explainer is None:
            raise RuntimeError(
                "LIME is unavailable because training data was not found."
            )

        row_1d = np.asarray(
            preprocessed_row
        ).squeeze()

        if row_1d.ndim != 1:
            raise ValueError(
                "preprocessed_row must represent one observation."
            )

        return self.lime_explainer.explain_instance(
            data_row=row_1d,
            predict_fn=self._predict_probabilities,
            labels=(1,),
            num_features=min(10, len(self.feature_names)),
            num_samples=500
        )
