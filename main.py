import numpy as np
from parser import Parser
from models import Regression_Model
from args_config import PARSER_CONFIG
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from utils import read_sheet, anova_table, maximize, Visualizer

# Initialize parser object and get arguments
try:
    parser = Parser(prog='Design Of Experiments',
                    description='Tools for design of experiments purposes')
    args = parser.get_args(
        PARSER_CONFIG
    )
except ValueError as e:
    print(e)

if __name__ == '__main__':

    try:
        # Extract data as Pandas Dataframe
        df = read_sheet(args.data_path)
        # Convert data to numpy array
        data = np.array(df)
    except FileNotFoundError as e:
        print(f'Error loading the data because the file was not found ! ')


    # Split inputs and outputs from the array
    features = np.array(data[:,:-1])
    outputs = np.array(data[:,-1])

    train_features, test_features, train_outputs, test_outputs = train_test_split(
        features, outputs, test_size=0.2, random_state=7

    )
    # Ensure the singularity of experiments (because there is 3 Box-Behnken center experiments)
    # print(test_outputs)

    # Store all the data in a dictionary
    data_dict = {
        'features' : features,
        'outputs' : outputs,
        'train_features' : train_features,
        'train_outputs' : train_outputs,
        'test_features' : test_features,
        'test_outputs' : test_outputs,

    }
    if args.train_models:
        models = {
            'linear' : Regression_Model(pipeline=(PolynomialFeatures(degree=1, include_bias=False),
                                                  LinearRegression()),
                                        model_name="Linear"),

            # 'ridge': Regression_Model(pipeline=(PolynomialFeatures(degree=1, include_bias=False),
            #                                      Ridge()),
            #                            model_name="Ridge"),

            '2FI' : Regression_Model(pipeline=(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                                               LinearRegression()),
                                     model_name="2FI", ),

            'quadratic' :Regression_Model(pipeline=(PolynomialFeatures(degree=2, include_bias=False),
                                                    LinearRegression()),
                                          model_name="Quadratic", ),

            'cubic' :Regression_Model(pipeline=(PolynomialFeatures(degree=3, include_bias=False),
                                                LinearRegression()),
                                      model_name="Cubic", )
        }
        results = []
        for name, model in models.items():
            print('----------------------------------', model.model_name, '----------------------------------')

            # Train the model
            model.fit(data_dict['train_features'], data_dict['train_outputs'])

            # Compute metrics
            r2_general, r2_train, r2_adj, r2_pred = model.get_metrics(data_dict)

            # Get equation (if applicable)
            model.get_equation()

            # Store results
            results.append({'model_name' :model.model_name,
                            'r2_general':r2_general,
                            'r2_train':r2_train,
                            'r2_adj':r2_adj,
                            'r2_pred':r2_pred})

        # Print the comparison table
        print("\n| **Model** |  **General R²** | **Train R²** | **Adjusted R²** | **Predicted R²** |")
        print("| --- | --- | --- | --- | --- |")
        for item in results:
            print(f"| {item['model_name']} | {item['r2_general']:.4f} | {item['r2_train']:.4f} | {item['r2_adj']:.4f} | {item['r2_pred']:.4f} |")

    if args.anova:
        # Get ANOVA results and other interpretable parameters
        # Degree 2 chosen because Quadratic was the best performing
        anova_results, other_params = anova_table(data_dict['features'], data_dict['outputs'], degree=2)
        print(anova_results)
        print(f"Other params : {other_params}")


    if args.max_opt:
        try :
            initial_guess = [1.5, 0.15, 60, 96]  # Initial guess for the optimization
            # Quadratic is the best performing model
            maximize(models['quadratic'], features, initial_guess)
        except NameError as e :
            print('Please train the models before using the next tools (Max_opt, Visualizer)')

    try :
        # Instantiate  a Visualizer object
        viz = Visualizer(models['quadratic'], data_dict)
    except NameError as e :
        print(e, '\tPlease train your models before visualizations')

    if args.main_visualizations:
        # Plot different visualization
        viz.pred_vs_actual()
        viz.normal_plot_of_residuals()
        viz.residuals_vs_pred()
        viz.residuals_vs_run()

    if args.box_cox_visualizations:
        # Box-Cox plot
        viz.box_cox()

    # Define feature names (Modify according to your dataset)
    feature_names = ["Silver nitrate concentration (mM)",
                     "MVLPAE-to-silver-nitrate ratio",
                     "Synthesis temperature (°C)",
                     "Synthesis time (min)"]

    # 2D and 3D features interactions plot
    if args.response_contour_visualizations:
        # RC
        viz.response_contour_plot(feature_names)

    if args.response_surface_visualizations:
        # RSM
        viz.response_surface_plot(feature_names)
