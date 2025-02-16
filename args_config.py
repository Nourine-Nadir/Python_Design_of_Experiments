PARSER_CONFIG = {
    'data_file':{
    'flags' : ('-dfp', '--data_path'),
    'help': 'The data file path (excel)',
    'type': str,
    'default':'AgNP.xlsx'
    },

    'train_models':{
        'flags' : ('-tr', '--train_models'),
        'help': 'Enable training models',
        'action' : 'store_true',
        'default': True
    },

    'anova':{
            'flags' : ('-anv', '--anova'),
            'help': 'Compute Anova and plot its table',
            'action' : 'store_true',
            'default': False
        },

    'maximize_optimization':{
        'flags' : ('-max', '--max_opt'),
        'help': 'Enable training models',
        'action' : 'store_true',
        'default': False
        },

    'main_visualizations':{
        'flags' : ('-main_viz', '--main_visualizations'),
        'help': 'Plot main Visualizations (pred_actual, normal_plot_of_res, res_pred, res_run)',
        'action' : 'store_true',
        'default': False
        },

    'box_cox':{
        'flags' : ('-box_cox', '--box_cox_visualizations'),
        'help': 'Compute and Plot box-cox visualization',
        'action' : 'store_true',
        'default': False
        },

    'response_contour_plot':{
        'flags' : ('-rc', '--response_contour_visualizations'),
        'help': 'Compute and Plot response contour visualization',
        'action' : 'store_true',
        'default': False
        },

    'response_surface_plot':{
        'flags' : ('-rsm', '--response_surface_visualizations'),
        'help': 'Compute and Plot response surface visualization',
        'action' : 'store_true',
        'default': False
        },

}