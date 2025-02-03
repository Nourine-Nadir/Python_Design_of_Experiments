# Import libraries
from tabnanny import verbose
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import cross_val_predict

# Model class
class Model():
    def __init__(self,pipeline, model_name='Undefined'):
        if isinstance(pipeline, tuple):
            self.model = make_pipeline(*pipeline)
        else:
            self.model = make_pipeline(pipeline)

        self.model_name = model_name

    def fit(self, inputs, outputs):
        self.model.fit(inputs, outputs)

    def adjusted_r2(self, r2, n, p):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def predicted_r2(self, features, output):
        predictions = cross_val_predict(self.model, features, output, cv=4)
        # print(predictions)
        ssr = mean_squared_error(output, predictions) * len(output)  # Sum of squared residuals
        tss = np.sum((output - np.mean(output)) ** 2)  # Total sum of squares
        return 1 - (ssr / tss)

    def get_metrics(self,data_dict, verbose=True):

        model_r2 = self.model.score(data_dict['test_features'], data_dict['test_outputs'], multioutput='variance_weighted')
        model_r2_adj = self.adjusted_r2(model_r2, *data_dict['features'].shape)
        model_r2_pred = self.predicted_r2(data_dict['features'], data_dict['outputs'])
        if verbose :
            print(f'------ {self.model_name} ------')
            print(f"R^2 score: {model_r2}")
            print(f"R^2 adjusted score: {model_r2_adj}")
            print(f"R^2 pred score: {model_r2_pred}\n")

    def get_equation(self):
        poly_step = None
        linreg_step = None

        for step_name, step in self.model.named_steps.items():
            if isinstance(step, PolynomialFeatures):
                poly_step = step
            reg_step = step

        if poly_step is None or reg_step is None:
            print("Error: Could not find PolynomialFeatures or LinearRegression in the pipeline.")
            return

        coefficients = reg_step.coef_  # Shape: (n_outputs, n_features_expanded)
        intercepts = reg_step.intercept_  # Shape: (n_outputs,)
        feature_names = poly_step.get_feature_names_out()

        for i in range(coefficients.shape[0]):  # Loop over outputs
            print(f"\nEquation for Output {i + 1}:")
            equation = f"y_{i + 1} = {intercepts[i]:.3f} "
            for name, coef in zip(feature_names, coefficients[i]):
                equation += f"+ ({coef:.3f} * {name}) "
            print(equation)