from sympy.physics.units import degree

from utils import *
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import  train_test_split
import numpy as np
from models import Regression_Model



engine = Engine()
df = engine.read_sheet('Step1.xlsx',)
data = np.array(df)

features = np.array(data[:,1:-2])
outputs = np.array(data[:,-1:])

train_features, test_features, train_outputs, test_outputs = train_test_split(
    features, outputs, test_size=0.2, random_state=23
)

data_dict = {
    'features' : features,
    'outputs' : outputs,
    'train_features' : train_features,
    'train_outputs' : train_outputs,
    'test_features' : test_features,
    'test_outputs' : test_outputs,

}
anova_results = anova_table(data_dict['features'], data_dict['outputs'], degree=2)
print(anova_results.head(10))

models = [
    # Regression_Model(pipeline=(PolynomialFeatures(degree=1), LinearRegression()), model_name="Linear"),
    # Regression_Model(pipeline=(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), LinearRegression()),
    #                  model_name="2FI",),
    Regression_Model(pipeline=(PolynomialFeatures(degree=2,include_bias=False), LinearRegression()),
                     model_name="Quadratic",),
    # Regression_Model(pipeline=(PolynomialFeatures(degree=3, include_bias=False), LinearRegression()),
    #                  model_name="Cubic",)
]
results = []

for model in models:
    print('----------------------------------',model.model_name,'----------------------------------')
    model.fit(data_dict['train_features'], data_dict['train_outputs'])
    r2, r2_adj, r2_pred = model.get_metrics(data_dict)
    model.get_equation()

    results.append([model.model_name, r2, r2_adj, r2_pred])

# Print the table
print(
    "\n| **Model** | **R²** | **Adjusted R²** | **Predicted R²** |")
print("| --- | --- | --- | --- | --- | --- |")
for row in results:
    print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]}")
# print(f'features {features.sh

# from scipy.optimize import minimize
#
# Extract coefficients and feature names
# coefficients = poly2.named_steps['linearregression'].coef_
# intercept = poly2.named_steps['linearregression'].intercept_
# feature_names = poly2.named_steps['polynomialfeatures'].get_feature_names_out()
#
# # Combine terms into a dictionary for clarity
# quadratic_formula = {name: coef for name, coef in zip(feature_names, coefficients)}
# quadratic_formula['intercept'] = intercept
#
# print("Quadratic Formula:")
# for term, coef in quadratic_formula.items():
#     print(f"{term}: {coef}")
#
# # Define the quadratic function based on the coefficients
# def quadratic_function(features):
#     expanded_features = poly2.named_steps['polynomialfeatures'].transform([features])
#     return -poly2.named_steps['linearregression'].predict(expanded_features)[0]  # Negative for maximization
#
# # Find the maximum value using optimization
# initial_guess = [50, 500, 3]  # Initial guess for the optimization
# result = minimize(quadratic_function,
#                   initial_guess,
#                   method='L-BFGS-B',
#                   bounds=[(0, 1),
#                           (0, 1.5),
#                           (0, 70),
#                           ])
#
# # Get the maximum value and corresponding features
# max_features = result.x
# max_value = -result.fun
#
# print("\nMaximum Value:")
# print(f"Features: {max_features}")
# print(f"Maximum Output: {max_value}")
#
