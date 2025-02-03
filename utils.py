import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


class Engine():
    def __init__(self):
        self.k = 0


    def read_sheet(self, file_path=None, sheetName=0):

        dict_data = pd.read_excel(file_path, sheet_name=sheetName)
        df_data = pd.DataFrame(dict_data)
        return df_data
    def get_inputs(self,df,index):
        return  df.iloc[:,index]

    # Function to compute adjusted R^2
    def adjusted_r2(self, r2, n, p):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def predicted_r2(self, model, features, output):
        predictions = cross_val_predict(model, features, output, cv=5)
        # print(predictions)
        ssr = mean_squared_error(output, predictions) * len(output)  # Sum of squared residuals
        tss = np.sum((output - np.mean(output)) ** 2)  # Total sum of squares
        return 1 - (ssr / tss)


import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import PolynomialFeatures


def anova_table(features, outputs, degree=2):
    # Générer les termes polynomiaux
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    features_poly = poly.fit_transform(features)

    # Nettoyer les noms des variables (remplacer espaces et puissances)
    feature_names = poly.get_feature_names_out()
    feature_names = [name.replace(" ", "_").replace("^", "__") for name in feature_names]
    print(feature_names)
    # Créer un DataFrame
    df = pd.DataFrame(features_poly, columns=feature_names)
    df["Output"] = outputs.ravel()
    print(df['Output'].shape)
    print(df['Output'].dtype)  # Vérifie si c'est un type numérique (float64, int64)
    df["Output"] = pd.to_numeric(df["Output"], errors='coerce')  # Convertit les valeurs non numériques en NaN

    # Construire la formule (sans `^`)
    formula = "Output ~ " + " + ".join(feature_names)

    # Ajuster le modèle
    model = ols(formula, data=df).fit()

    # Calculer l'ANOVA
    anova_results = sm.stats.anova_lm(model, typ=1)
    anova_results = anova_results.round(2)

    # Now, print the table with P-value, F-value, and Significance
    # anova_results["P-value"] = anova_results["PR(>F)"]
    anova_results.rename(columns={'PR(>F)': 'P-value', 'F':'F-value'}, inplace=True)
    # Ajouter la colonne de signification
    anova_results["Significance"] = anova_results["P-value"].apply(lambda p: "S" if p < 0.05 else "NS")

    return anova_results

