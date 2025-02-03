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
        return  model_r2, model_r2_adj, model_r2_pred

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



class FC_net(nn.Module):
    def __init__(self,
                 lr,
                 input_shape,
                 fc1_dims,
                 fc2_dims,
                 n_output):
        super(FC_net, self).__init__()
        self.fc1 = nn.Linear(input_shape, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.fc4 = nn.Linear(fc2_dims, n_output)

        self.apply(weights_init_)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.99 ** epoch, 1e-2)
        )

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(f'device used: {self.device}')
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.FloatTensor(x).to(self.device)
        elif isinstance(x, T.Tensor):
            x = x.to(self.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def save_model(self, PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)

        T.save(self.state_dict(), PATH)

    def load_model(self, PATH, map_location=None):
        if map_location is None:
            map_location = self.device  # Use the model's current device
        self.load_state_dict(T.load(PATH, map_location=map_location, weights_only=True))
