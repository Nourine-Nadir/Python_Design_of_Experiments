import itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.formula.api import ols
from sklearn.preprocessing import PolynomialFeatures


def read_sheet(file_path=None, sheetName=0):
    try:
        dict_data = pd.read_excel(file_path, sheet_name=sheetName)
        df_data = pd.DataFrame(dict_data)
        return df_data
    except ValueError as e:
        print(e)


def anova_table(features, outputs, degree=2):
    # Generate polynomial terms
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    features_poly = poly.fit_transform(features)

    # Clean variable names (replace spaces and powers)
    feature_names = poly.get_feature_names_out()
    # Replace power symbol with __ to avoid invalid syntax
    feature_names = [name.replace(" ", "_").replace("^", "__") for name in feature_names]

    # Create DataFrame
    df = pd.DataFrame(features_poly, columns=feature_names)
    df["Output"] = outputs.ravel()
    df["Output"] = pd.to_numeric(df["Output"], errors='coerce')

    # Build formula
    formula = "Output ~ " + " + ".join(feature_names)

    # Fit model
    model = ols(formula, data=df).fit()

    # Calculate ANOVA
    anova_results = sm.stats.anova_lm(model, typ=1)

    # Store residual information before dropping
    residual_ss = anova_results.loc['Residual', 'sum_sq']
    residual_df = anova_results.loc['Residual', 'df']
    residual_ms = residual_ss / residual_df

    # Calculate Pure Error and Lack of Fit
    y_pred = model.fittedvalues
    residuals = outputs.ravel() - y_pred

    # Group data by unique x values to calculate pure error
    df_with_pred = df.copy()
    df_with_pred['Predicted'] = y_pred
    df_with_pred['Residuals'] = residuals

    # Calculate means for each unique combination of feature values
    grouped = df_with_pred.groupby(feature_names)['Output']

    # Pure Error calculations - fixed to return scalar values
    pure_error_ss = sum(((group - group.mean()) ** 2).sum() for name, group in grouped)
    pure_error_df = sum(len(group) - 1 for name, group in grouped)
    pure_error_ms = pure_error_ss / pure_error_df if pure_error_df > 0 else 0

    # Lack of Fit calculations
    lack_of_fit_ss = residual_ss - pure_error_ss
    lack_of_fit_df = residual_df - pure_error_df
    lack_of_fit_ms = lack_of_fit_ss / lack_of_fit_df if lack_of_fit_df > 0 else 0

    # F-value and p-value for Lack of Fit
    if pure_error_ms > 0:
        lof_f_value = lack_of_fit_ms / pure_error_ms
        lof_p_value = 1 - stats.f.cdf(lof_f_value, lack_of_fit_df, pure_error_df)
    else:
        lof_f_value = np.nan
        lof_p_value = np.nan

    # Create rows for Residual, Pure Error and Lack of Fit
    residual_row = pd.DataFrame({
        'sum_sq': residual_ss,
        'df': residual_df,
        'mean_sq': residual_ms,
        'F': np.nan,
        'PR(>F)': np.nan,
    }, index=['Residual'])

    pure_error_row = pd.DataFrame({
        'sum_sq': pure_error_ss,
        'df': pure_error_df,
        'mean_sq': pure_error_ms,
        'F': np.nan,
        'PR(>F)': np.nan,
    }, index=['Pure Error'])

    lack_of_fit_row = pd.DataFrame({
        'sum_sq': lack_of_fit_ss,
        'df': lack_of_fit_df,
        'mean_sq': lack_of_fit_ms,
        'F': lof_f_value,
        'PR(>F)': lof_p_value,
    }, index=['Lack of Fit'])

    # Calculate model statistics
    model_ssr = anova_results['sum_sq'].iloc[:-1].sum()
    model_df = anova_results['df'].iloc[:-1].sum()
    model_ms = model_ssr / model_df
    model_f_value = model_ms / residual_ms
    model_p_value = 1 - stats.f.cdf(model_f_value, model_df, residual_df)

    # Create model row
    model_row = pd.DataFrame({
        'sum_sq': model_ssr,
        'df': model_df,
        'mean_sq': model_ms,
        'F': model_f_value,
        'PR(>F)': model_p_value,
    }, index=['Model'])

    # Remove original Residual row and add new decomposed rows
    anova_results = anova_results.drop('Residual')

    # Combine all components
    anova_results = pd.concat([
        model_row,
        anova_results,
        residual_row,
        lack_of_fit_row,
        pure_error_row
    ])

    # Rename columns and add significance
    if 'PR(>F)' in anova_results.columns:
        anova_results.rename(columns={'PR(>F)': 'P-value', 'F': 'F-value'}, inplace=True)
    anova_results["Significance"] = anova_results["P-value"].apply(
        lambda p: "S" if (not np.isnan(p) and p < 0.05) else "NS"
    )

    # Calculate other parameters
    pred_range = np.max(y_pred) - np.min(y_pred)
    rmse = np.sqrt(residual_ss / residual_df)  # Using stored residual values
    adequate_precision = pred_range / rmse
    mean_output = np.mean(y_pred)
    cv = (rmse / mean_output) * 100

    other_params = {
        'Adequate_Precision': round(adequate_precision, 4),
        'Std_dev': round(rmse, 4),
        'Coefficient of Variation (%)': round(cv, 4)
    }
    anova_results = anova_results.round(4)

    # Save results
    anova_results.to_csv('Anova.xlsx')

    return anova_results, other_params


def quadratic_function_with_penalty(features, model):
    def objective(x):
        expanded_features = model.model.named_steps['polynomialfeatures'].transform([x])
        predicted_value = model.model.named_steps['linearregression'].predict(expanded_features)[0]

        # Penalty if output exceeds 0.4
        penalty = 10000 * max(0, predicted_value - 0.4)
        return -(predicted_value - penalty)

    return objective


def maximize(model, features, initial_guess):
    # Create the objective function with the model
    objective_function = quadratic_function_with_penalty(features, model)

    result = minimize(objective_function,
                      initial_guess,
                      method='L-BFGS-B',
                      bounds=[(0.5, 1.5),
                              (0.05, 0.15),
                              (50, 70),
                              (10, 120)])

    max_features = result.x
    max_value = -result.fun

    print("\nConstrained Maximum Value:")
    print(f"Features: {max_features}")
    print(f"Maximum Output (≤0.4): {max_value}")

    return max_value


class Visualizer():
    def __init__(self, model, data_dict):
        self.model = model
        self.data_dict = data_dict
        self.predictions = self.model.model.predict(self.data_dict['features']).flatten()
        self.actual = self.data_dict['outputs'].flatten()

        # Calculate residuals
        self.residuals = self.actual - self.predictions
        self.studentized_residuals = self.residuals / np.std(self.residuals)

        self.run_numbers = np.arange(1, len(self.actual) + 1)

    def pred_vs_actual(self):
        scatter = plt.scatter(self.actual, self.predictions,
                              c=abs(self.residuals),
                              cmap="magma",
                              edgecolor="black")
        plt.colorbar(scatter, label="Absolute Residuals")  # Add colorbar

        plt.plot([min(self.actual), max(self.actual)], [min(self.actual), max(self.actual)], color="black",
                 linestyle="--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs. Actual")
        filename = f"figs/Pred vs. Actual.png"  # Create a descriptive filename
        plt.savefig(filename, dpi=500, bbox_inches='tight')  # Save as PNG with 1000 DPI

        plt.show()

    def normal_plot_of_residuals(self):
        # Calculate the normal probabilities
        n = len(self.studentized_residuals)
        probabilities = np.arange(1, n + 1) / (n + 1) * 100  # Convert to percentage

        # Sort the residuals
        sorted_residuals = np.sort(self.studentized_residuals)

        # Add reference line - corrected unpacking
        slope, intercept, r_value, p_value, std_err = stats.linregress(sorted_residuals, probabilities)
        # Calculate the fitted probabilities (i.e., the values along the reference line)
        fitted_probabilities = slope * sorted_residuals + intercept

        # Calculate the residuals (distance from the fitted line)
        distances_from_line = np.abs(probabilities - fitted_probabilities)

        # Create scatter plot with magma colors

        scatter = plt.scatter(sorted_residuals, probabilities,
                              c=distances_from_line,
                              cmap="magma",
                              edgecolor="black",
                              zorder=5
                              )
        plt.colorbar(scatter, label='Distance from the line')

        plt.plot(sorted_residuals, slope * sorted_residuals + intercept, "k--")
        plt.title("Normal Plot of Residuals")
        plt.xlabel("Internally Studentized Residuals")
        plt.ylabel("Normal % Probability")

        plt.xlim(-2, 2)
        plt.ylim(0, 100)
        filename = f"figs/Normal plot of Residuals.png"  # Create a descriptive filename
        plt.savefig(filename, dpi=500, bbox_inches='tight')  # Save as PNG with 1000 DPI

        plt.show()

    def residuals_vs_pred(self):
        # Plot 1: Residuals vs. Predicted
        distance_from_zero = np.abs(self.studentized_residuals)

        scatter = plt.scatter(self.predictions, self.studentized_residuals,
                              c=distance_from_zero,
                              cmap="magma",
                              edgecolor="black")
        plt.colorbar(scatter, label="Studentized Residuals")  # Add colorbar

        plt.axhline(0, color="black", linestyle="--")
        plt.axhline(3, color="red", linestyle="-")
        plt.axhline(-3, color="red", linestyle="-")
        plt.xlabel("Predicted")
        plt.ylabel("Internally Studentized Residuals")
        plt.title("Residuals vs. Predicted")
        filename = f"figs/Residuals vs. Pred.png"  # Create a descriptive filename
        plt.savefig(filename, dpi=500, bbox_inches='tight')  # Save as PNG with 1000 DPI

        plt.show()

    def residuals_vs_run(self):
        distance_from_zero = np.abs(self.studentized_residuals)

        # Plot the line first (behind the points)
        plt.plot(self.run_numbers, self.studentized_residuals, 'k-', linewidth=0.5, alpha=0.5)
        # Create scatter plot with magma coloring on top
        scatter = plt.scatter(self.run_numbers, self.studentized_residuals,
                              c=distance_from_zero,
                              cmap="magma",
                              edgecolor="black", s=50)
        plt.colorbar(scatter, label="Studentized Residuals")  # Add colorbar

        plt.axhline(0, color="black", linestyle="--")
        plt.axhline(3, color="red", linestyle="-")
        plt.axhline(-3, color="red", linestyle="-")
        plt.xlabel("Run Number")
        plt.ylabel("Internally Studentized Residuals")
        plt.title("Residuals vs. Run")

        plt.tight_layout()
        filename = f"figs/Residuals vs. Run.png"  # Create a descriptive filename
        plt.savefig(filename, dpi=500, bbox_inches='tight')  # Save as PNG with 1000 DPI

        plt.show()

    def box_cox(self):
        positive_data = self.data_dict['outputs'].flatten()

        if np.any(positive_data <= 0):
            print("Warning: Non-positive values found and removed for Box-Cox analysis")
            positive_data = positive_data[positive_data > 0]

        # Create lambda range
        lambdas = np.linspace(-3, 3, 200)  # Increased number of points for smoother curve

        # Calculate log-likelihood for each lambda
        boxcox_llf = np.array([stats.boxcox_llf(lmbda, positive_data) for lmbda in lambdas])
        print(f"Sample size: {len(positive_data)}")
        print(f"Data range: [{np.min(positive_data):.3f}, {np.max(positive_data):.3f}]")
        print(f"Data mean: {np.mean(positive_data):.3f}")
        print(f"Data std: {np.std(positive_data):.3f}")
        print(f"Log-likelihood range: [{np.min(boxcox_llf):.3f}, {np.max(boxcox_llf):.3f}]")
        # Find optimal lambda
        _, opt_lambda = stats.boxcox(positive_data)
        if isinstance(opt_lambda, np.ndarray):
            opt_lambda = float(opt_lambda)  # Convert to scalar if array

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(lambdas, boxcox_llf, color='black', linewidth=2)
        plt.axvline(opt_lambda, color='green', linestyle='--',
                    label=f'Optimal λ = {opt_lambda:.3f}')

        # Add confidence interval (95%)
        max_llf = np.max(boxcox_llf)
        ci_threshold = max_llf - 0.5 * stats.chi2.ppf(0.5, df=1)
        plt.axhline(ci_threshold, color='red', linestyle=':', alpha=0.5)

        # Highlight confidence interval range
        valid_lambdas = lambdas[boxcox_llf >= ci_threshold]
        plt.axvspan(min(valid_lambdas), max(valid_lambdas),
                    alpha=0.2, color='green',
                    label=f'50% CI: [{min(valid_lambdas):.2f}, {max(valid_lambdas):.2f}]')

        # Enhance plot appearance
        plt.grid(True, alpha=0.3)
        plt.xlabel("Lambda (λ)", fontsize=12)
        plt.ylabel("Log-Likelihood", fontsize=12)
        plt.title("Box-Cox Transformation Analysis", fontsize=14)
        plt.legend()
        plt.tight_layout()
        filename = f"figs/Box-Cox Transformation.png"  # Create a descriptive filename
        plt.savefig(filename, dpi=500, bbox_inches='tight')  # Save as PNG with 1000 DPI

        plt.show()

        # Print interpretation
        transformations = {
            -2: "Inverse square (1/y²)",
            -1: "Inverse (1/y)",
            -0.5: "Inverse square root (1/√y)",
            0: "Natural log (ln(y))",
            0.5: "Square root (√y)",
            1: "No transformation needed",
            2: "Square (y²)"
        }

        print(f"\nOptimal λ = {opt_lambda:.3f}")
        print(f"50% Confidence Interval: [{min(valid_lambdas):.2f}, {max(valid_lambdas):.2f}]")

        # Find closest common transformation
        common_lambdas = np.array(list(transformations.keys()))
        closest_lambda = common_lambdas[np.abs(common_lambdas - opt_lambda).argmin()]
        print(f"\nClosest common transformation: λ = {closest_lambda}")
        print(f"Suggested transformation: {transformations[closest_lambda]}")

    def response_contour_plot(self, feature_names: list = None):
        if not feature_names:
            print('\nResponse contour couldn\'t be done \nPlease pass features names ..')
            return
        font_properties = {'family': 'Ubuntu', 'size': 20}
        # Get feature values
        X_data = self.data_dict['features']
        num_features = X_data.shape[1]

        # Iterate over all feature pairs
        feature_combinations = list(itertools.combinations(range(num_features), 2))

        for feat_x, feat_y in feature_combinations:
            # Extract min and max values for the chosen features
            x_min, x_max = X_data[:, feat_x].min(), X_data[:, feat_x].max()
            y_min, y_max = X_data[:, feat_y].min(), X_data[:, feat_y].max()

            # Create a mesh grid for the selected features
            x_values = np.linspace(x_min, x_max, 100)
            y_values = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_values, y_values)

            # Flatten the grid for predictions
            grid_points = np.c_[X.ravel(), Y.ravel()]

            # Fix other features at their mean values
            fixed_values = np.mean(X_data, axis=0)  # Compute means for all features
            fixed_values = np.tile(fixed_values, (grid_points.shape[0], 1))  # Repeat for all grid points

            # Replace the selected feature columns with grid values
            fixed_values[:, feat_x] = grid_points[:, 0]
            fixed_values[:, feat_y] = grid_points[:, 1]

            # Predict output values
            predicted_outputs = self.model.model.predict(fixed_values)
            predicted_outputs = predicted_outputs.reshape(X.shape)

            # Create a new figure for each plot
            plt.figure(figsize=(12, 10))

            # Plot the contour
            contour = plt.contourf(X, Y, predicted_outputs,
                                   cmap="magma", levels=50)
            plt.colorbar(contour)

            # Set axis labels and title
            plt.xlabel(feature_names[feat_x], **font_properties)
            plt.ylabel(feature_names[feat_y], **font_properties)
            plt.title(f"SPR intensity", family='ubuntu', size=20)
            filename = f"figs/RC {feature_names[feat_x]}_vs_{feature_names[feat_y]}.png"  # Create a descriptive filename
            plt.savefig(filename, dpi=500, bbox_inches='tight')  # Save as PNG with 1000 DPI
            # Show the plot
            plt.show()

    def response_surface_plot(self, feature_names: list = None):
        if not feature_names:
            print('\nResponse contour couldn\'t be done \nPlease pass features names ..')
            return
        # Get feature values
        X_data = self.data_dict['features']
        num_features = X_data.shape[1]

        # Set font properties
        plt.rcParams['font.weight'] = 'bold'
        font_properties = {'family': 'Ubuntu', 'weight': 'bold', 'size': 16}

        # Iterate over all feature pairs
        feature_combinations = list(itertools.combinations(range(num_features), 2))

        for feat_x, feat_y in feature_combinations:
            # Extract min and max values for the chosen features
            x_min, x_max = X_data[:, feat_x].min(), X_data[:, feat_x].max()
            y_min, y_max = X_data[:, feat_y].min(), X_data[:, feat_y].max()

            # Create a mesh grid for the selected features
            x_values = np.linspace(x_min, x_max, 20)
            y_values = np.linspace(y_min, y_max, 20)
            X, Y = np.meshgrid(x_values, y_values)

            # Flatten the grid for predictions
            grid_points = np.c_[X.ravel(), Y.ravel()]

            # Fix other features at their mean values
            fixed_values = np.mean(X_data, axis=0)
            fixed_values = np.tile(fixed_values, (grid_points.shape[0], 1))

            # Replace the selected feature columns with grid values
            fixed_values[:, feat_x] = grid_points[:, 0]
            fixed_values[:, feat_y] = grid_points[:, 1]

            # Predict output values
            predicted_outputs = self.model.model.predict(fixed_values)
            predicted_outputs = predicted_outputs.reshape(X.shape)

            # Create a new figure for each 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the surface
            surface = ax.plot_surface(X, Y, predicted_outputs, cmap="magma", alpha=0.95)

            # Add ground projection (contour plot)
            offset = predicted_outputs.min() - (predicted_outputs.max() - predicted_outputs.min()) * 0.1
            contour = ax.contour(X, Y, predicted_outputs,
                                 zdir='z',
                                 offset=offset,
                                 levels=15,
                                 cmap="magma",
                                 alpha=0.7)

            # Add a filled contour on the ground
            ground = ax.contourf(X, Y, predicted_outputs,
                                 zdir='z',
                                 offset=offset,
                                 levels=15,
                                 cmap="magma",
                                 alpha=0.7)

            ax.view_init(elev=15, azim=220)

            # Adjust z-axis limits to accommodate the ground projection
            ax.set_zlim(offset, predicted_outputs.max())

            # Add a color bar with bold label
            cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
            cbar.ax.tick_params(labelsize=12)
            for label in cbar.ax.get_yticklabels():
                label.set_weight('bold')

            # Set labels with larger, bold font
            ax.set_xlabel(feature_names[feat_x], labelpad=18, **font_properties)
            ax.set_ylabel(feature_names[feat_y], labelpad=18, **font_properties)

            ax.set_zlabel("SPR intensity", labelpad=15, **font_properties)

            # Make tick labels bold and larger
            ax.tick_params(axis='both', which='major', labelsize=12)
            for tick in ax.get_xticklabels():
                tick.set_weight('bold')
            for tick in ax.get_yticklabels():
                tick.set_weight('bold')
            for tick in ax.get_zticklabels():
                tick.set_weight('bold')

            if feature_names[feat_x] == "Synthesis temperature (°C)":
                desired_ticks = [50, 55, 60, 65, 70]
                ax.set_xticks(desired_ticks)
                ax.set_xticklabels(desired_ticks, size=13, weight='bold')
            elif feature_names[feat_y] == "Synthesis temperature (°C)":
                desired_ticks = [50, 55, 60, 65, 70]
                ax.set_yticks(desired_ticks)
                ax.set_yticklabels(desired_ticks, size=13, weight='bold')

            filename = f"figs/RSM {feature_names[feat_x]}_VS_{feature_names[feat_y]}.png"
            plt.savefig(filename, dpi=500, bbox_inches='tight')

            plt.show()
