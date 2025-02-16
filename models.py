from sklearn.linear_model import LinearRegression, Ridge
import torch.nn.functional as F
import statsmodels.api as sm
import torch.nn as nn
import numpy as np
import torch as T
import os
from scipy import stats
from sklearn.feature_selection import f_regression

def weights_init_(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)




from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import cross_val_predict

# Model class
class Regression_Model():
    def __init__(self,pipeline, model_name='Undefined'):
        if isinstance(pipeline, tuple):
            self.model = make_pipeline(*pipeline)
        else:
            self.model = make_pipeline(pipeline)

        self.model_name = model_name


    def fit(self, inputs, outputs):
        self.model.fit(inputs, outputs)

    def adjusted_r2(self, r2, n, p):
        if n <= p + 1:  # Avoid division by zero or negative denominator
            print("Warning: Not enough samples to compute adjusted R^2 properly.")
            return None  # Or return R² itself

        r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

        # Ensure R² adjusted is within valid range [0, 1]
        return max(0, min(r2_adj, 1))

    def predicted_r2_cv(self, features, output):
        predictions = cross_val_predict(self.model, features, output, cv=5)
        ssr = mean_squared_error(output, predictions) * len(output)  # Sum of squared residuals
        tss = np.sum((output - np.mean(output)) ** 2)  # Total sum of squares
        return 1 - (ssr / tss)

    def predicted_r2_model_based(self, test_features, test_outputs):
        """
        Computes the Predicted R² (Q²) using the trained model
        by predicting on the test set and comparing with actual values.
        """
        # Get predictions using the trained model
        predictions = self.model.predict(test_features)

        # Compute SSR (Sum of Squared Residuals)
        ssr = np.sum((test_outputs - predictions) ** 2)

        # Compute TSS (Total Sum of Squares)
        tss = np.sum((test_outputs - np.mean(test_outputs)) ** 2)

        # Compute R² predicted
        r2_pred = 1 - (ssr / tss)

        return r2_pred

    def get_metrics(self, data_dict, verbose=True):
        # Train R^2
        model_r2_train = self.model.score(
            data_dict['train_features'], data_dict['train_outputs'], multioutput='variance_weighted'
        )

        # Test R^2 (Generalization Check)
        model_r2_general = self.model.score(
            data_dict['features'], data_dict['outputs'], multioutput='variance_weighted'
        )

        # Adjusted R^2 (on training set)
        model_r2_adj = self.adjusted_r2(model_r2_train, *data_dict['train_features'].shape)

        # Predicted R^2 (on test set)
        model_r2_pred_cv = self.predicted_r2_cv(data_dict['features'], data_dict['outputs'])

        # Predicted R^2 using Model-Based Method
        model_r2_pred_mb = self.predicted_r2_model_based(data_dict['test_features'], data_dict['test_outputs'])

        if verbose:
            print(f'------ {self.model_name} ------')
            print(f"Train R^2 score: {model_r2_train}")
            print(f"Test R^2 score: {model_r2_general}")
            print(f"Adjusted R^2 score: {model_r2_adj}")
            print(f"Predicted R^2 CV score: {model_r2_pred_cv}")
            print(f"Predicted R^2 MB score: {model_r2_pred_mb}\n")

        return model_r2_general, model_r2_train, model_r2_adj, model_r2_pred_mb

    def get_equation(self):
        """
        Gets the equation representation of the regression model.
        Handles both single-output and multi-output cases.
        """
        poly_step = None
        reg_step = None

        for step_name, step in self.model.named_steps.items():
            if isinstance(step, PolynomialFeatures):
                poly_step = step
            elif isinstance(step, (LinearRegression, Ridge)):
                reg_step = step

        if poly_step is None or reg_step is None:
            print("Error: Could not find PolynomialFeatures or Regression step in the pipeline.")
            return

        coefficients = reg_step.coef_  # Shape: (n_features,) or (n_outputs, n_features)
        intercept = reg_step.intercept_  # Shape: scalar or (n_outputs,)
        feature_names = poly_step.get_feature_names_out()

        # Handle both single-output and multi-output cases
        if coefficients.ndim == 1:
            # Single output case
            print("\nRegression Equation:")
            equation = f"y = {intercept:.6f} "
            for name, coef in zip(feature_names, coefficients):
                if coef != 0:  # Only include non-zero terms
                    sign = "+" if coef > 0 else "-"
                    equation += f"{sign} {abs(coef):.6f}*{name} "
            print(equation)
        else:
            # Multi-output case
            for i in range(coefficients.shape[0]):
                print(f"\nEquation for Output {i + 1}:")
                equation = f"y_{i + 1} = {intercept[i]:.6f} "
                for name, coef in zip(feature_names, coefficients[i]):
                    if coef != 0:  # Only include non-zero terms
                        sign = "+" if coef > 0 else "-"
                        equation += f"{sign} {abs(coef):.6f}*{name} "
                print(equation)
