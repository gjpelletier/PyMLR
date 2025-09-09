# -*- coding: utf-8 -*-

__version__ = "1.2.186"

def check_X_y(X,y, enable_categorical=False):

    '''
    Check the X and y inputs used in regression 
    '''
    
    import pandas as pd
    import numpy as np
    import sys

    # start with copies of X and y to avoid changing the original
    X = X.copy()
    y = y.copy()

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        ctrl = (X.index == y.index).all()
        if not ctrl:
            print('Check X and y: they need to have the same index values!','\n')
            sys.exit()

    if isinstance(X, pd.DataFrame):
        ctrl = X.isna().sum().sum()==0
        if not ctrl:
            print('Check X: it needs to have no nan values!','\n')
            sys.exit()
        ctrl = not np.isinf(X.select_dtypes(include=[float])).any().any()
        if not ctrl:
            print('Check X: it needs to have no inf values!','\n')
            sys.exit()
        ctrl = X.columns.is_unique
        if not ctrl:
            print('Check X: X needs to have unique column names for every column!','\n')
            sys.exit()
    
    if isinstance(y, pd.Series):
        ctrl = y.isna().sum().sum()==0
        if not ctrl:
            print('Check y: it needs to have no nan values!','\n')
            sys.exit()
        ctrl = not np.isinf(y.values).any()
        if not ctrl:
            print('Check y: it needs to have no inf values!','\n')
            sys.exit()

    if isinstance(X, np.ndarray):
        ctrl = np.isnan(X).sum().sum()==0
        if not ctrl:
            print('Check X: it needs to have no nan values!','\n')
            sys.exit()
        ctrl = not np.any([np.isinf(val) if isinstance(val, (int, float)) else False for val in X])
        if not ctrl:
            print('Check X: it needs to have no inf values!','\n')
            sys.exit()
    
    if isinstance(y, np.ndarray):
        ctrl = np.isnan(y).sum().sum()==0
        if not ctrl:
            print('Check y: it needs to have no nan values!','\n')
            sys.exit()
        ctrl = not np.any([np.isinf(val) if isinstance(val, (int, float)) else False for val in y])
        if not ctrl:
            print('Check y: it needs to have no inf values!','\n')
            sys.exit()

    # convert X and y to pandas dataframe and series if not already
    # if isinstance(X, np.ndarray):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        X.columns = ['X' + str(i) for i in X.columns]       
    # if isinstance(y, np.ndarray):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        y.name = 'y'
    
    if enable_categorical:
        cat_dtype_cols = X.select_dtypes(include=['category']).columns.tolist()
        non_cat_dtype_cols = [
            col for col, dtype in X.dtypes.items() 
            if not isinstance(dtype, pd.CategoricalDtype)
        ]
        ctrl = np.isreal(X[non_cat_dtype_cols]).all()
    else:    
        ctrl = np.isreal(X).all()
    if not ctrl:
        if enable_categorical:
            print('Check non-categorical X cols: they need to be all real numbers!','\n')
        else:
            print('Check X: it needs be all real numbers!','\n')
        sys.exit()

    ctrl = X.ndim==2
    if not ctrl:
        print('Check X: it needs be 2-D!','\n')
        sys.exit()
    ctrl = np.isreal(y).all()
    if not ctrl:
        print('Check y: it needs be all real numbers!','\n')
        sys.exit()
    ctrl = y.ndim==1
    if not ctrl:
        print('Check y: it needs be 1-D!','\n')
        sys.exit()
    ctrl = X.shape[0] == y.shape[0]
    if not ctrl:
        print('Check X and y: X and y need to have the same number of rows!','\n')
        sys.exit()

    return X, y

def check_X(X, enable_categorical=False):

    '''
    Check the X input used in regression 
    '''
    
    import pandas as pd
    import numpy as np
    import sys

    # start with copies of X and y to avoid changing the original
    X = X.copy()

    if isinstance(X, pd.DataFrame):
        ctrl = X.isna().sum().sum()==0
        if not ctrl:
            print('Check X: it needs to have no nan values!','\n')
            sys.exit()
        ctrl = not np.isinf(X.select_dtypes(include=[float])).any().any()
        if not ctrl:
            print('Check X: it needs to have no inf values!','\n')
            sys.exit()
        ctrl = X.columns.is_unique
        if not ctrl:
            print('Check X: X needs to have unique column names for every column!','\n')
            sys.exit()
    
    if isinstance(X, np.ndarray):
        ctrl = np.isnan(X).sum().sum()==0
        if not ctrl:
            print('Check X: it needs to have no nan values!','\n')
            sys.exit()
        ctrl = not np.any([np.isinf(val) if isinstance(val, (int, float)) else False for val in X])
        if not ctrl:
            print('Check X: it needs to have no inf values!','\n')
            sys.exit()
    
    # convert X to pandas dataframe if not already
    # if isinstance(X, np.ndarray):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        X.columns = ['X' + str(i) for i in X.columns]       

    if enable_categorical:
        cat_dtype_cols = X.select_dtypes(include=['category']).columns.tolist()
        non_cat_dtype_cols = [
            col for col, dtype in X.dtypes.items() 
            if not isinstance(dtype, pd.CategoricalDtype)
        ]
        ctrl = np.isreal(X[non_cat_dtype_cols]).all()
    else:
        ctrl = np.isreal(X).all()
    if not ctrl:
        if enable_categorical:
            print('Check non-categorical X cols: they need to be all real numbers!','\n')
        else:
            print('Check X: it needs be all real numbers!','\n')
        sys.exit()

    ctrl = X.ndim==2
    if not ctrl:
        print('Check X: it needs be 2-D!','\n')
        sys.exit()
    
    return X

def cross_val_scoring():
    '''
    returns result dictionary of all string names of 
    regression and classification scorers 
    for sklearn cross_val_score
    '''

    from sklearn.metrics import get_scorer_names
    
    # Get all available scorers
    all_scorers = get_scorer_names()
    
    # Classification scorers (based on common usage and metric semantics)
    classification_scoring = sorted([
        scorer for scorer in all_scorers
        if scorer in {
            "accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_micro", "f1_weighted",
            "precision", "precision_macro", "precision_micro", "precision_weighted",
            "recall", "recall_macro", "recall_micro", "recall_weighted",
            "roc_auc", "roc_auc_ovr", "roc_auc_ovo", "average_precision",
            "neg_log_loss", "neg_brier_score"
        }
    ])
    
    # Regression scorers (based on regression semantics)
    regression_scoring = sorted([
        scorer for scorer in all_scorers
        if scorer in {
            "r2", "neg_mean_squared_error", "neg_root_mean_squared_error",
            "neg_mean_absolute_error", "neg_median_absolute_error",
            "neg_mean_squared_log_error", "explained_variance"
        }
    ])

    result = {
        'classification_scoring': classification_scoring,
        'regression_scoring': regression_scoring,
    }        
    
    return result 
   
def preprocess_train(df, **kwargs):
    """
    Detects categorical (numeric and non-numeric) columns, applies one-hot encoding,
    scales continuous numeric columns, and safely handles cases with missing types.
    All categorical features are cast to float.

    Args:
        df (pd.DataFrame): Training data 
            (if df is not a dataframe it will be converted to a dataframe)
        kwargs: optional keyword arguments
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)
            use_encoder: True (default) or False
            use_scaler: True (default) or False

    Returns:
        dict: {
            'df': original data, converted to dataframe if needed
            'df_processed': Preprocessed DataFrame,
            'columns_original': list of column names of original dataframe
            'columns_processed': list of column names of processed dataframe            
            'encoder': Fitted OneHotEncoder or None,
            'scaler': Fitted Scaler or None,
            'use_encoder': True or False, whether to one-hot encode non_bool_cats
            'use_scaler': True or False, whether to scale continupus_cols
            'continuous_cols': list of continuous numeric columns,
            'categorical_numeric': list of categorical numeric columns,
            'non_numeric_cats': list of non_numeric categorical columns,
            'bool_cols': list of boolean columns,
            'categorical_cols': list of all categorical columns,
            'non_bool_cats': list of categorical columns that are not boolean,
            'datetime_cols': list of datetime columns,
            'category_mappings': Mapping of categories or {},
            'threshold_skew_pos': same as input threshold_skew_pos,
            'threshold_skew_neg': same as inpt threshold_skew_neg,
            'skew_df': dataframe of skew of each feature in continuous_col,
            'skewed_pos_cols': list of candidate features for positive unskewing, 
            'skewed_neg_cols': list of candidate features for negative unskewing,
            'unskew_pos': True if candidate features for positive unskewing were unskewed,
            'unskew_neg': True if candidate features for negative unskewing were unskewed
        }
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
    from PyMLR import check_X
    import scipy.stats as stats
    from datetime import datetime
    import warnings

    # Define default values of input data arguments
    defaults = {
        'enable_categorical': False, 
        'use_scaler': True, 
        'use_encoder': True, 
        'threshold_cat': 12,
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5        
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")
    
    # extract control variables from data dictionary
    use_encoder = data['use_encoder']
    use_scaler = data['use_scaler']
    threshold_cat = data['threshold_cat']
    scale = data['scale']
    unskew_pos = data['unskew_pos']
    threshold_skew_pos = data['threshold_skew_pos']
    unskew_neg = data['unskew_neg']
    threshold_skew_neg = data['threshold_skew_neg']
    enable_categorical = data['enable_categorical']
    
    # Start with a copy to avoid changing the original df
    df = df.copy()
    df_orig = df.copy()

    # check df and convert to dataframe if not already
    # df = check_X(df)
    df = check_X(df, enable_categorical=enable_categorical)
    
    # # identify columns that are any typed or coercible date or time
    def get_all_datetime_like_columns(df):
        # 1. Columns with datetime-like dtypes
        typed = df.select_dtypes(include=['datetime', 'datetimetz', 'timedelta']).columns.tolist()    
        # 2. Columns of type object or string that appear coercible
        coercible = []
        for col in df.select_dtypes(include=['object', 'string']).columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                try:
                    converted = pd.to_datetime(df[col], errors='coerce')
                    if not converted.isna().all():
                        coercible.append(col)
                except Exception:
                    continue  # Skip completely incompatible columns
        return list(dict.fromkeys(typed + coercible))  # Preserve order and uniqueness

    # lists of columns of dtype categorical and not categorical
    cat_dtype_cols = df.select_dtypes(include=['category']).columns.tolist()
    non_cat_dtype_cols = [
        col for col, dtype in df.dtypes.items() 
        if not isinstance(dtype, pd.CategoricalDtype)
    ]

    # preprocess only the non-cat dtypes if enable_categorical
    if enable_categorical:
        df = df[non_cat_dtype_cols].copy()
    
    datetime_cols = get_all_datetime_like_columns(df)

    # identify boolean columns and covert to int
    bool_cols = df.select_dtypes(include='bool').columns.tolist()

    # Identify columns that are not boolean but contain only 0 and 1
    columns_with_0_and_1 = [
        col for col in df.columns 
        if not pd.api.types.is_bool_dtype(df[col]) and set(df[col].unique()) == {0, 1}
    ]

    # combine columns_with_0_and_1 into bool_cols so they will be treated the same 
    bool_cols = bool_cols + columns_with_0_and_1

    df[bool_cols] = df[bool_cols].astype(int)

    # identify numeric columns
    numerical_cols = df.select_dtypes(include='number').columns.tolist()

    # # identify columns that are not any type of number, date, or time
    # non_numeric_cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # non_numeric_cats = df.select_dtypes(
    #     exclude=['number', 'datetime', 'datetimetz', 'timedelta']).columns.tolist()    
    non_numeric_cats = df.select_dtypes(exclude=['number']).columns.tolist()    
    # remove items from non_numeric_cats that are in datetime_cols
    non_numeric_cats = [item for item in non_numeric_cats if item not in datetime_cols]

    categorical_numeric = [col for col in numerical_cols if df[col].nunique() <= threshold_cat and col not in bool_cols]
    continuous_cols = [col for col in numerical_cols if col not in categorical_numeric and col not in bool_cols]

    all_cat_cols = categorical_numeric + non_numeric_cats + bool_cols

    non_bool_cats = categorical_numeric + non_numeric_cats

    # -------- Transforming skewed continuous_cols before scaling --------

    # dataframe of skew of each feature
    skew_df = pd.DataFrame(df[continuous_cols].columns, columns=['feature'])
    skew_df['skew'] = skew_df['feature'].apply(lambda feature: stats.skew(df[feature]))
    skew_df['skew_pos'] = skew_df['skew'].apply(lambda x: True if x >= threshold_skew_pos else False)
    skew_df['skew_neg'] = skew_df['skew'].apply(lambda x: True if x <= threshold_skew_neg else False)

    # function to loop through cols of dataframe and test condition
    get_columns_by_condition = lambda df, condition: [col for col in df.columns if condition(df[col])]

    # skewed_pos_cols is a list of continuous cols that can be log1p transformed if positive skew
    col_test = skew_df['feature'][skew_df['skew']>=threshold_skew_pos].to_list()
    df_test = df[col_test]
    condition = lambda col: col.min() >= 0    # can only use log1p if >= 0
    skewed_pos_cols = get_columns_by_condition(df_test, condition)
    
    # skewed_neg_cols is a list of continuous cols that can be sqrt transformed if negative skew
    col_test = skew_df['feature'][skew_df['skew']<=threshold_skew_neg].to_list()
    df_test = df[col_test]
    condition = lambda col: col.min() >= 0    # can only use sqrt if >= 0
    skewed_neg_cols = get_columns_by_condition(df_test, condition)
    
    # log1p-transform positively skewed features in df if unskew_pos==True
    if unskew_pos:
        df[skewed_pos_cols] = np.log1p(df[skewed_pos_cols])

    # sqrt-transform negatively skewed features in df if unskew_neg==True
    if unskew_neg:
        df[skewed_neg_cols] = np.sqrt(df[skewed_neg_cols])
    
    # -------- One-hot encoding --------
    
    # One-hot encoding
    if use_encoder and non_bool_cats:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_array = encoder.fit_transform(df[non_bool_cats])
        encoded_df = pd.DataFrame(encoded_array,
                                  columns=encoder.get_feature_names_out(non_bool_cats),
                                  index=df.index).astype(float)
        category_mappings = {
            col: encoder.categories_[i].tolist()
            for i, col in enumerate(non_bool_cats)
        }
    else:
        encoder, encoded_df, category_mappings = None, pd.DataFrame(index=df.index), {}

    # -------- Scaling --------

    # Scaling
    if use_scaler and continuous_cols:
        scaler = StandardScaler() if scale == 'standard' else MinMaxScaler()
        scaled_array = scaler.fit_transform(df[continuous_cols])
        scaled_df = pd.DataFrame(scaled_array, columns=continuous_cols, index=df.index).astype(float)
    else:
        scaler, scaled_df = None, pd.DataFrame(index=df.index)

    # Merge all transformed features

    # drop_cols = non_bool_cats + continuous_cols
    # df_processed = df.drop(columns=drop_cols, errors='ignore')
    # df_processed = df_processed.join([encoded_df, scaled_df])

    df_processed = df.copy()
    if use_encoder:
        df_processed = df_processed.drop(columns=non_bool_cats, errors='ignore')
        df_processed = df_processed.join([encoded_df])
    if use_scaler:
        df_processed = df_processed.drop(columns=continuous_cols, errors='ignore')
        df_processed = df_processed.join([scaled_df])

    # set all_cols except datetime_cols to float
    all_cols = df_processed.columns.to_list()
    float_cols = [item for item in all_cols if item not in datetime_cols]
    df_processed[float_cols] = df_processed[float_cols].astype(float)

    # columnn-wise concat of original cat_dtype_cols with preprocessed non-categorical dtypes
    if enable_categorical:
        df_processed = pd.concat([df_orig[cat_dtype_cols], df_processed], axis=1)
    
    return {
        'df': df_orig,
        'df_processed': df_processed,
        'columns_original': df_orig.columns.to_list(),
        'columns_processed': df_processed.columns.to_list(),
        'enable_categorical': enable_categorical,
        'cat_dtype_cols': cat_dtype_cols,
        'non_cat_dtype_cols': non_cat_dtype_cols,
        'encoder': encoder,
        'scaler': scaler,
        'use_encoder': use_encoder,
        'use_scaler': use_scaler,
        'continuous_cols': continuous_cols,
        'categorical_numeric': categorical_numeric,
        'non_numeric_cats': non_numeric_cats,
        'bool_cols': bool_cols,
        'categorical_cols': all_cat_cols,
        'non_bool_cats': non_bool_cats,
        'datetime_cols': datetime_cols,
        'category_mappings': category_mappings,
        'unskew_pos': unskew_pos,
        'unskew_neg': unskew_neg,
        'threshold_skew_pos': threshold_skew_pos,
        'threshold_skew_neg': threshold_skew_neg,
        'skew_df': skew_df,
        'skewed_pos_cols': skewed_pos_cols, 
        'skewed_neg_cols': skewed_neg_cols 
    }

def preprocess_test(df_test, preprocess_result):
    """
    Transforms the test DataFrame using artifacts from preprocess_train.
    Handles missing columns and unknown categories safely.

    Args:
        df_test (pd.DataFrame): Input test DataFrame 
            if df_test is not a dataframe it will be converted to dataframe
        preprocess_result (dict): Output dictionary from preprocess_train

    Returns:
        pd.DataFrame: Preprocessed test DataFrame
    """
    import pandas as pd
    import numpy as np
    import sys
    from PyMLR import check_X
    import warnings
    warnings.filterwarnings('ignore')

    if preprocess_result != None:
        encoder = preprocess_result['encoder']
        scaler = preprocess_result['scaler']
        use_encoder = preprocess_result['use_encoder']
        use_scaler = preprocess_result['use_scaler']
        categorical_cols = preprocess_result['categorical_cols']
        non_bool_cats = preprocess_result['non_bool_cats']
        continuous_cols = preprocess_result['continuous_cols']
        datetime_cols = preprocess_result['datetime_cols']
        # unskewing of skewed continuous_cols
        threshold_skew_pos = preprocess_result['threshold_skew_pos'] 
        threshold_skew_neg = preprocess_result['threshold_skew_neg'] 
        skewed_pos_cols = preprocess_result['skewed_pos_cols']  
        skewed_neg_cols = preprocess_result['skewed_neg_cols']  
        unskew_pos = preprocess_result['unskew_pos'] 
        unskew_neg = preprocess_result['unskew_neg']     
        enable_categorical = preprocess_result['enable_categorical']
        cat_dtype_cols = preprocess_result['cat_dtype_cols']
        non_cat_dtype_cols = preprocess_result['non_cat_dtype_cols']
    else:
        print('Exited preprocess_test because preprocess_result=None','\n')
        sys.exit()
    
    # copy df_test to prevent altering the original
    df_test = df_test.copy()
    df_test_orig = df_test.copy()

    # check that df_test is a dataframe and convert to dataframe if needed
    df_test = check_X(df_test, enable_categorical=enable_categorical)

    # preprocess only the non-cat dtypes if enable_categorical
    if enable_categorical:
        df_test = df_test[non_cat_dtype_cols].copy()
    
    # -------- Transforming skewed continuous_cols before scaling --------

    # log1p-transform positively skewed features in df if unskew_pos==True

    if unskew_pos:
        n_err = np.sum(df_test[skewed_pos_cols].values < 0)
        if n_err > 0:
            print(f'WARNING! - {n_err} negative values in skewed_pos_cols were transformed to nan')
        df_test[skewed_pos_cols] = np.log1p(df_test[skewed_pos_cols])

    # sqrt-transform negatively skewed features in df if unskew_neg==True
    if unskew_neg:
        n_err = np.sum(df_test[skewed_neg_cols].values < 0)
        if n_err > 0:
            print(f'WARNING! - {n_err} negative values in skewed_neg_cols were transformed to nan')
        df_test[skewed_neg_cols] = np.sqrt(df_test[skewed_neg_cols])
        
    # -------- One-hot encoding --------

    for col in non_bool_cats:
        if df_test.get(col, pd.Series(dtype=object)).dtype == bool:
            df_test[col] = df_test[col].astype(int)
    
    # Encode categoricals
    if use_encoder and encoder is not None and non_bool_cats:
        df_cat = pd.DataFrame(index=df_test.index)
        for col in non_bool_cats:
            df_cat[col] = df_test[col] if col in df_test.columns else np.nan

        encoded_array = encoder.transform(df_cat[non_bool_cats])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(non_bool_cats),
            index=df_test.index
        ).astype(float)
    else:
        encoded_df = pd.DataFrame(index=df_test.index)

    # -------- Scaling --------
    
    # Scale continuous
    if use_scaler and scaler is not None and continuous_cols:
        df_num = pd.DataFrame(index=df_test.index)
        for col in continuous_cols:
            df_num[col] = df_test[col] if col in df_test.columns else 0.0

        scaled_array = scaler.transform(df_num[continuous_cols])
        scaled_df = pd.DataFrame(scaled_array, columns=continuous_cols, index=df_test.index).astype(float)
    else:
        scaled_df = pd.DataFrame(index=df_test.index)

    # drop_cols = set(non_bool_cats + continuous_cols)
    # remaining = df_test.drop(columns=[col for col in drop_cols if col in df_test.columns], errors='ignore')
    # df_processed = remaining.join([encoded_df, scaled_df])
    df_processed = df_test.copy()
    if use_encoder:
        df_processed = df_processed.drop(columns=non_bool_cats, errors='ignore')
        df_processed = df_processed.join([encoded_df])
    if use_scaler:
        df_processed = df_processed.drop(columns=continuous_cols, errors='ignore')
        df_processed = df_processed.join([scaled_df])

    # # set all_cols except datetime_cols to float
    # df_processed = df_processed.astype(float)
    all_cols = df_processed.columns.to_list()
    float_cols = [item for item in all_cols if item not in datetime_cols]
    df_processed[float_cols] = df_processed[float_cols].astype(float)

    # columnn-wise concat of original cat_dtype_cols with preprocessed non-categorical dtypes
    if enable_categorical:
        df_processed = pd.concat([df_test_orig[cat_dtype_cols], df_processed], axis=1)
    
    # Restore warnings to normal
    warnings.filterwarnings("default")

    return df_processed

def show_dtypes(df):
    '''
    Show the dtype and number of unique values for each column of a dataframe
    Arg:
        df: dataframe to be examined
    Returns transposed dataframe of each feature, dtype, and number of unique values    
    '''
    import pandas as pd
    
    # Display dtype and number of unique values for each column
    # result = df.apply(lambda col: pd.Series({'dtype': col.dtype, 'unique_values': col.nunique()}))
    result = df.apply(lambda col: pd.Series({'dtype': col.dtype, 
                                             'nunique': col.nunique(),
                                             'isna_sum': col.isna().sum(),
                                             # 'zero_sum': (col == 0).sum(),
                                             # 'one_sum': (col == 1).sum()
                                            }))
    
    with pd.option_context('display.max_rows', None):
        print(result.T)

    return result.T
 
def show_optuna(study, random_state=42):

    '''
    Show the results of the optimized optuna study

    input:
    study= optimized optuna study
    random_state= optional seed for reproducibility of param_importances

    output:
    display and save plots of optuna study results
    result dictionary of the following:
    'best_params': dictionary of final best optimized hyperparameter values
    'trial_scores': list of score for each trial iteration
    'best_scores': list of best score as of each trial iteration
    'param_importances': dictionary of hyperparameters and normalized importances values (sum to 1)
    '''
    
    import optuna
    from optuna.importance import get_param_importances
    from optuna.importance import FanovaImportanceEvaluator
    import matplotlib.pyplot as plt
    import warnings

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    warnings.filterwarnings('ignore')

    # Show the best parameters
    print("Best parameters:")
    print('')
    for key, value in study.best_params.items():
        # print(f"{key:<10}: {value:<10}")
        print(f"{key}: {value}")
    print('')
    print("Best score:", study.best_value)
    print('')

    # Generate optimization history plot
    trials = study.trials
    trial_values = [trial.value for trial in trials]
    best_values = [max(trial_values[: i + 1]) for i in range(len(trial_values))]  # type: ignore
    # fig, ax = plt.subplots(figsize=(7, 5))
    plt.close('all')  # Clear any previous figures
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('gainsboro')
    ax.set_title("Optimization History")
    ax.plot(best_values, label='Best Value', color='tab:red', zorder=3)
    ax.plot(trial_values, marker="o", linestyle='none', label='Trial Value', color='tab:blue', zorder=2)
    ax.grid(color='white', linestyle='-', linewidth=0.5, zorder=0)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Score')
    ax.legend()    
    plt.savefig('optuna_optimization_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    # Generate hyperparameter importance plot
    # # Note: get_param_importances without evaluator argument are not deterministic and differ at each call
    # param_importances = get_param_importances(study)
    # # Note: evaluator from FanovaImportanceEvaluator with specifed seed provides reproducible get_param_importances
    evaluator = FanovaImportanceEvaluator(seed=random_state)
    param_importances = optuna.importance.get_param_importances(study, evaluator=evaluator)
    data = param_importances
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=False))
    categories = list(data.keys())
    values = list(data.values())
    # fig = plt.figure();
    # fig, axs = plt.subplots(figsize=(7, 5))
    plt.close('all')  # Clear any previous figures
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('gainsboro')  # Set the background color to gray
    plt.barh(categories, values, color='tab:blue', zorder=3)
    ax.grid(color='white', linestyle='-', linewidth=0.5, zorder=0)  # White grid lines
    for index, value in enumerate(values):
        plt.text(value + .001, index, (str(round(value,2)) if value > 0.01 else '<0.01'), va='center', fontsize=10)  # Adjust position with `+2`
    plt.xlabel('Relative Importance')
    plt.ylabel('Hyperparameters')
    plt.title('Hyperparameter Importances')
    plt.tight_layout()
    plt.savefig('optuna_parameter_importance.png', 
                dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close(fig)
    
    # Generate contour plot of the two most important parameters (shows parameter interactions)
    keys = [str(key) for key, value in param_importances.items() if isinstance(value, (int, float))]    
    if len(keys) >= 2:
        first_key = keys[0]
        second_key = keys[1]
        optuna.visualization.matplotlib.plot_contour(study, params=[first_key, second_key])
        plt.title(f"{first_key} vs. {second_key}")
        plt.savefig(f"optuna_{first_key}_vs_{second_key}.png", 
                    dpi=300, bbox_inches='tight') 
        plt.show()
    
    # Restore warnings to normal
    warnings.filterwarnings("default")

    result = {
        'best_params': study.best_params.items(),
        'trial_scores': trial_values,
        'best_scores': best_values,
        'param_importances': param_importances
    }
    
    return result

def show_coef(fitted_model, X):
    '''
    Show the intercept and coefs of a fitted sklearn model that has intercept and coefs
    '''
    import numpy as np
    import pandas as pd
    import sys
    ctrl = fitted_model.coef_.size==len(X.columns)
    if not ctrl:
        print('Error: Mismatch between len(X.columns) and model.coef_.size!','\n')
        sys.exit()
    if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
            and fitted_model.coef_.size==len(X.columns)):
        intercept = fitted_model.intercept_
        coefficients = fitted_model.coef_
        # dataframe of model parameters, intercept and coefficients, including zero coefs
        n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
        popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
        for i in range(n_param):
            if i == 0:
                popt[0][i] = 'Intercept'
                popt[1][i] = fitted_model.intercept_
            else:
                popt[0][i] = X.columns[i-1]
                popt[1][i] = fitted_model.coef_[i-1]
        popt = pd.DataFrame(popt).T
        popt.columns = ['Feature', 'Parameter']
        # Table of intercept and coef
        popt_table = pd.DataFrame({
                "Feature": popt['Feature'],
                "Parameter": popt['Parameter']
            })
        popt_table.set_index('Feature',inplace=True)
        print(popt_table.to_markdown(index=True))
    else:
        print('The fitted model does not have intercept and/or coefficients')
        popt_table = None
    return popt_table

def show_vif(X):
    '''
    Show Variance Inflation Factors of X
    '''
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_ = X.copy()
    X__ = sm.add_constant(X_)    # Add a constant for the intercept
    vif = pd.DataFrame()
    vif['Feature'] = X__.columns
    vif["VIF"] = [variance_inflation_factor(X__.values, i)
                        for i in range(len(X__.columns))]
    vif.set_index('Feature',inplace=True)
    vif.index.name = 'Feature'
    print(vif.to_markdown(index=True))
    return vif

def test_model(
        model, X, y, preprocess_result=None, selected_features=None):

    """
    Plots Actual vs Predicted and Residuals vs Predicted 
    for fitted sklearn linear regression models 
    and provide goodness of fit statistics
    (replaces plot_linear_results_test)

    Args:
    model= fitted sklearn linear regression model object
    X = dataframe of the candidate independent variables 
    y = series of the dependent variable (one column of data)
    preprocess_result = results of preprocess_train
    selected_features = optimized selected features
    Returns: dict of the following:
        metrics: goodness of fit metrics
        stats: dataframe of fit metrics
        y_pred: predicted y given X
        fig= figure for the residuals plot
    """
 
    from PyMLR import check_X_y, preprocess_test, fitness_metrics
    import pandas as pd
    import numpy as np
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import warnings
    import sys

    # copy X and y to avoid altering originals
    X = X.copy()
    y = y.copy()

    # check X and y and put into dataframe if needed
    X, y = check_X_y(X, y)
    
    if preprocess_result!=None:
        X = preprocess_test(X, preprocess_result)
        
    if selected_features==None:
        selected_features = X.columns.to_list()

    y_pred = model.predict(X[selected_features])    

    # Goodness of fit statistics
    metrics = fitness_metrics(
        model, 
        X[selected_features], y)
    stats = pd.DataFrame([metrics]).T
    stats.index.name = 'Statistic'
    stats.columns = ['Regressor']

    result = {}
    result['metrics'] = metrics
    result['stats'] = stats
    result['y_pred'] = model.predict(X[selected_features])

    print('')
    print("Goodness of fit to testing data in result['metrics']:")
    print('')
    print(result['stats'].to_markdown(index=True))
    print('')
    
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred,
        kind="actual_vs_predicted",
        ax=axs[0]
    )
    axs[0].set_title("Actual vs. Predicted")
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred,
        kind="residual_vs_predicted",
        ax=axs[1]
    )
    axs[1].set_title("Residuals vs. Predicted")
    rmse = np.sqrt(np.mean((y-y_pred)**2))
    fig.suptitle(
        f"Predictions compared with actual values and residuals (RMSE={rmse:.3f})")
    plt.tight_layout()

    result['fig'] = fig
    
    return result
    
def plot_predictions_from_test(
    model, X, y, 
    standardize=True, scaler=None, 
    pca_transform=False, pca=None, n_components=None):

    """
    DEPRECATED
    
    Plots Actual vs Predicted and Residuals vs Predicted 
    for fitted sklearn linear regression models 

    Args:
    model= fitted sklearn linear regression model object
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)
    standardize= True (default) or False whether to standardize X
    scaler= None (default) or 
        the scaler object that was fit to training X
    pca_transform= True (default) or False whether to PCA transform X
    pca= None (default) or 
        the PCA object that was fit to training X
    n_components= number of components to use to fit PCA transformer
        if pca_transform=True and pca=None
    Returns:
        fig= figure for the plot
    """
 
    from PyMLR import check_X_y
    import pandas as pd
    import numpy as np
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import warnings
    import sys

    # copy X and y to avoid altering originals
    X = X.copy()
    y = y.copy()

    # check X and y and put into dataframe if needed
    X, y = check_X_y(X, y)
    
    if standardize and scaler == None:
        # create a new scaler 
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        # Convert scaled arrays into pandas dataframes with same column names as X
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        # Copy index from unscaled to scaled dataframes
        X_scaled.index = X.index
        # Replace X with the standardized X for regression
        X = X_scaled.copy()
    elif standardize and scaler != None:
        # use the input scaler
        X_scaled = scaler.transform(X)
        # Convert scaled arrays into pandas dataframes with same column names as X
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        # Copy index from unscaled to scaled dataframes
        X_scaled.index = X.index
        # Replace X with the standardized X for regression
        X = X_scaled.copy()

    if pca_transform and pca == None:
        # fit new PCA transformer
        n_components = min(X.shape[0],X.shape[1])   # lesser of n_samples and n_features
        pca = PCA(n_components=n_components).fit(X)
        X = pca.transform(X)        
        n_components = pca.n_components_
        X = pd.DataFrame(pca.transform(X), columns= [f"PC_{i+1}" for i in range(n_components)])
        X.index = y.index    
    if pca_transform and pca != None:
        # use input PCA transformer
        n_components = pca.n_components_
        X = pd.DataFrame(pca.transform(X), columns= [f"PC_{i+1}" for i in range(n_components)])
        X.index = y.index    
        
    y_pred = model.predict(X)    
    
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred,
        kind="actual_vs_predicted",
        ax=axs[0]
    )
    axs[0].set_title("Actual vs. Predicted")
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred,
        kind="residual_vs_predicted",
        ax=axs[1]
    )
    axs[1].set_title("Residuals vs. Predicted")
    rmse = np.sqrt(np.mean((y-y_pred)**2))
    fig.suptitle(
        f"Predictions compared with actual values and residuals (RMSE={rmse:.3f})")
    plt.tight_layout()

    return fig
     
def fitness_metrics(model, X, y):
    '''
    Extracts multiple evaluation metrics 
    from a trained scikit-learn linear regression model
    given fitted model, X, and y

    Parameters:
        model: A fitted sklearn linear regression model object
        X: Features used to fit the model
        y: Response variable used to fit the model
    
    Returns:
        dict of relevant sklearn metrics
    '''
    
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
        explained_variance_score,
        max_error,
        mean_absolute_percentage_error,
        mean_squared_log_error
    )
    import numpy as np
    
    y_pred = model.predict(X)
    metrics = {}

    # Safe defaults
    metrics['R-squared'] = r2_score(y, y_pred)
    metrics['MSE'] = mean_squared_error(y, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['Explained Variance'] = explained_variance_score(y, y_pred)
    metrics['MAE'] = mean_absolute_error(y, y_pred)
    metrics['Max Error'] = max_error(y, y_pred)

    # Handle MAPE with care
    try:
        metrics['MAPE'] = mean_absolute_percentage_error(y, y_pred)
    except ValueError as e:
        # metrics['MAPE'] = f'Error: {e}'
        metrics['MAPE'] = None

    # Handle MSLE with care (requires non-negative values)
    try:
        metrics['MSLE'] = mean_squared_log_error(y, y_pred)
    except ValueError as e:
        # metrics['MSLE'] = f'Error: {e}'
        metrics['MSLE'] = None

    metrics['n_samples'] = X.shape[0]
    
    return metrics

def pseudo_r2(model, X, y):
    """
    Calculate McFadden's pseudo-R² 
    for a fitted scikit-learn LogisticRegression model
    given fitted model, X, and y
    Works with binary and multinomial classification.
    
    Parameters:
        model: A fitted sklearn.linear_model.LogisticRegression object
        X: Features used to fit the model
        y: True binary labels
    
    Returns:
        McFadden's pseudo-R²
    """
    import numpy as np
    from sklearn.metrics import log_loss

    probas = model.predict_proba(X)
    ll_full = -log_loss(y, probas, normalize=False)
    
    # Build null model prediction: use empirical class distribution
    classes, class_counts = np.unique(y, return_counts=True)
    class_probs = class_counts / len(y)
    probas_null = np.tile(class_probs, (len(y), 1))
    ll_null = -log_loss(y, probas_null, normalize=False)
    
    return 1 - ll_full / ll_null

def fitness_metrics_logistic(model, X, y, brier=True):
    """
    Extracts multiple evaluation metrics 
    from a trained LogisticRegression model
    given fitted model, X, and y
    Works with binary and multinomial classification.

    Parameters:
        model: A fitted sklearn.linear_model.LogisticRegression object
        X: Features used to fit the model
        y: True binary labels
    
    Returns:
        dict of relevant sklearn metrics
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, log_loss, brier_score_loss,
        f1_score, precision_score, recall_score)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    average_method = 'binary' if len(np.unique(y)) == 2 else 'macro'

    metrics = {
        "mcfadden_pseudo_r2": pseudo_r2(model, X, y),
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred, average=average_method),
        "precision": precision_score(y, y_pred, average=average_method),
        "recall": recall_score(y, y_pred, average=average_method),
        "log_loss": log_loss(y, y_proba)
    }

    # Brier score only valid for binary: take prob class 1
    if brier and len(np.unique(y)) == 2  and np.max(y_proba)<=1:
        metrics["brier_score"] = brier_score_loss(y, y_proba[:, 1])

    metrics['n_classes'] = len(np.unique(y))
    metrics['n_samples'] = X.shape[0]
    
    return metrics

def detect_dummy_variables(df, sep=None):
    """
    Detects dummy variables in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        sep (str, optional): Separator used in column names if dummy variables were created with pd.get_dummies. Defaults to None.

    Returns:
        bool: True if dummy variables are likely present, False otherwise.
    """
    import pandas as pd

    for col in df.columns:
        if df[col].nunique() == 2 and df[col].isin([0, 1]).all():
            return True

    if sep is not None:
        try:
            pd.from_dummies(df, sep=sep)
            return True
        except ValueError:
            pass

    return False

def detect_gpu():
    '''
    Check if the computer has an nvidia gpu
    returns boolean use_gpu= True or False to indicate if the computer has a gpu or not
    '''
    import subprocess
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # print("GPU detected: NVIDIA GPU is available.")
            # print(result.stdout)
            use_gpu = True
        else:
            # print("No NVIDIA GPU detected or `nvidia-smi` not installed.")
            use_gpu = False
    except FileNotFoundError:
        # print("`nvidia-smi` command not found. Ensure NVIDIA drivers are installed.")
        use_gpu = False
        print("Auto-detect gpu failed, try using keyword argument gpu=False")
    return use_gpu

def nnn(x):

    """
    PURPOSE
    Count the number of non-nan values in the numpy array x
    USAGE
    result = nnn(x)
    INPUT
    x = any numpy array of any dimension
    OUTPUT
    result = number of non-nan values in the array x
    """
    
    import numpy as np

    result = np.count_nonzero(not np.isnan(x))
    
    return result

def stepwise(X, y, **kwargs):

    """
    Python function for stepwise linear regression to minimize AIC or BIC
    and eliminate non-signficant predictors

    by
    Greg Pelletier
    gjpelletier@gmail.com
    17-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        criterion= 'aic' (default) or 'bic' where
            'aic': use the Akaike Information Criterion to score the model
            'bic': use the Bayesian Information Criterion to score the model
            'r2': use the adjusted r-squared to score the model
            'p_coef': use p-values of coefficients to select features
                using p_coef  as criterion automatically uses backward direction
        verbose= 'on' (default) or 'off'
        direction= 'forward' (default), 'backward', or 'all' where
            'forward' (default): 
                1) Start with no predictors in the model
                2) Add the predictor that results in the lowest AIC
                3) Keep adding predictors as long as it reduces AIC
            'backward':
                1) Fit a model with all predictors.
                2) Remove the predictor that results in the lowest AIC
                3) Keep removing predictors as long as it reduces AIC
            'all': find the best model of all possibe subsets of predictors
                Note: 'all' requires no more than 20 columns in X
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        drop_insig= 'on' (default) or 'off'
            'on': drop predictors with p-values below threshold p-value (default) 
            'off': keep all predictors regardless of p-value
        p_threshold= threshold p-value to eliminate predictors (default 0.05)                
        allow_dummies= True or False (default)                
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        model_object, model_outputs 
            model_object is the final fitted model returned by statsmodels.api OLS
            model_outputs is a dictionary of the following outputs:
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'selected_features' are the final selected features
                - 'step_features' are the features and fitness score at each step
                    (if 'direction'=='forward' or 'direction'=='backward'), 
                    or the best 10 subsets of features (if 'direction'=='all'),
                    including the AIC, BIC, and adjusted r-squared
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'popt': Constant (intercept) and coefficients for the 
                    best fit models from each of the four methods
                - 'pcov': Covariance matrix of features 
                - 'vif': Variance Inlfation Factors of selected_features
                - 'stats': Regression statistics for each model
                - 'summary': statsmodels model.summary() of the best fitted model

    NOTE
    Do any necessary/optional cleaning of data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique 
    column names for each column.

    EXAMPLE 1 - use the default AIC as the criterion with forward stepping:
    best_features, best_model = stepwise(X, y)

    EXAMPLE 2 - use the option of BIC as the criterion with forward stepping:
    best_features, best_model = stepwise(X, y, criterion='BIC')

    EXAMPLE 3 - use the option of BIC as the criterion with backward stepping:
    best_features, best_model = stepwise(X, y, criterion='BIC', direction='backward')

    EXAMPLE 4 - use the option of BIC as the criterion and search all possible models:
    best_features, best_model = stepwise(X, y, criterion='BIC', direction='all')

    """

    from PyMLR import detect_dummy_variables, check_X_y, fitness_metrics
    from PyMLR import preprocess_train, preprocess_test
    import statsmodels.api as sm
    from itertools import combinations
    import pandas as pd
    import numpy as np
    import sys
    from sklearn.preprocessing import StandardScaler
    import time
    import matplotlib.pyplot as plt
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.metrics import PredictionErrorDisplay
    import warnings

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'criterion': 'AIC',
        'verbose': 'on',
        'direction': 'forward',
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'allow_dummies': False,
        'drop_insig': 'on',
        'p_threshold': 0.05,
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        

        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")
    
    p_threshold = data['p_threshold']
    if data['criterion'] == 'aic':
        data['criterion'] = 'AIC'
    if data['criterion'] == 'bic':
        data['criterion'] = 'BIC'
    if data['criterion'] == 'AIC':
        crit = 'AIC'
    elif data['criterion'] == 'BIC':
        crit = 'BIC'
    elif data['criterion'] == 'r2':
        crit = 'rsq_adj'
    if data['criterion'] == 'p_coef':
        data['direction'] = 'backward'

    # check for input errors
    ctrl = detect_dummy_variables(X)
    if ctrl and not data['allow_dummies']:
        print('Check X: X appears to have dummy variables. Use allow_dummies=True, or use Lasso, Ridge, or ElasticNet','\n')
        sys.exit()

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    if data['direction'] == 'all':
        ctrl = X.shape[1]<=20
        if not ctrl:
            print('X needs to have <= 20 columns to use all directions! Try forward or backward stepping instead!','\n')
            sys.exit()

    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Set start time for calculating run time
    if data['direction'] == 'all':
        nsubsets = 2**X.shape[1]
        runtime = (nsubsets / (2**16)) * (120/60)   # guess runtime assuming 120 sec for 16 candidate features
        if X.shape[1] > 15:
            print("Fitting models for all "+str(nsubsets)+
                " subsets of features, this may take about {:.0f} minutes, please wait ...".format(runtime))
        else:
            print("Fitting models for all "+str(nsubsets)+
                " subsets of features, this may take up to a minute, please wait ...")
            
    else:
        print('Fitting Stepwise models, please wait ...')
    if data['verbose'] == 'on':
        print('')
    start_time = time.time()
    # model_outputs = {}
    step_features = {}
            
    if data['direction'] == 'forward':

        # Forward selection to minimize AIC or BIC
        selected_features = []
        remaining_features = list(X.columns)

        # best_score = float('inf')

        istep = 0
        while remaining_features:
            score_with_candidates = []        
            
            # start with only a constant in the model
            if istep == 0:
                X_const = np.ones((len(y), 1))  # column of ones for the intercept
                X_const = pd.DataFrame(X_const,columns=['constant'])
                X_const.index = X.index
                model = sm.OLS(y, X_const).fit()

                # output dataframe of score at each step
                step_features = {'Step': 0, 'AIC': model.aic, 'BIC': model.bic, 
                    'rsq_adj': 0.0, 'Features': [[]]}
                step_features = pd.DataFrame(step_features)
                
                if data['criterion'] == 'AIC':
                    candidate = ['']
                    score_with_candidates.append((model.aic, candidate))
                    best_score = model.aic
                elif data['criterion'] == 'BIC':
                    candidate = ['']
                    score_with_candidates.append((model.bic, candidate))
                    best_score = model.bic
                elif data['criterion'] == 'r2':
                    candidate = ['']
                    score_with_candidates.append((1-model.rsquared_adj, candidate))
                    best_score = 1-model.rsquared_adj
                                       
            for candidate in remaining_features:
                model = sm.OLS(y, sm.add_constant(X[selected_features + [candidate]])).fit()
                if data['criterion'] == 'AIC':
                    score_with_candidates.append((model.aic, candidate))
                elif data['criterion'] == 'BIC':
                    score_with_candidates.append((model.bic, candidate))
                elif data['criterion'] == 'r2':
                    score_with_candidates.append((1-model.rsquared_adj, candidate))
            score_with_candidates.sort()  # Sort by criterion
            best_new_score, best_candidate = score_with_candidates[0]        
            if best_new_score < best_score:
                best_score = best_new_score
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

                # add new row to output dataframe
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                
                if data['criterion'] == 'AIC':
                    score = model.aic
                elif data['criterion'] == 'BIC':
                    score = model.bic
                elif data['criterion'] == 'r2':
                    score = model.rsquared_adj
                if (data['verbose'] == 'on' and
                        (remaining_features == [] and data['drop_insig'] == 'off')):
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    print(step_features.to_markdown(index=False))
                    print('\nForward step '+str(istep)+", "+crit+"= {:.2f}".format(score))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        

            else:            
                remaining_features.remove(best_candidate)
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                                
                if (data['verbose'] == 'on' and 
                        (remaining_features != [] and data['drop_insig'] == 'off')):
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    print(step_features.to_markdown(index=False))
                    print('\nFinal forward model before removing insignficant features if any:')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                break            

        if data['drop_insig'] == 'on':
    
            # Backward elimination of features with p < p_threshold
            while selected_features:
    
                # Backward elimination of non-signficant predictors
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                p_values = model.pvalues.iloc[1:]  # Ignore intercept
                max_p_value = p_values.max()
        
                if max_p_value > p_threshold:
                    worst_feature = p_values.idxmax()
                    selected_features.remove(worst_feature)
                else:
                    
                    # add new row to output dataframe
                    new_row = {'Step': istep+1, 'AIC': model.aic, 'BIC': model.bic, 
                               'rsq_adj': model.rsquared_adj, 
                               'Features': np.array(selected_features)}
                    step_features = pd.concat([step_features, 
                            pd.DataFrame([new_row])], ignore_index=True)

                    if data['verbose'] == 'on':
                        print("Model skill and features at each step in model_outputs['step_features']:\n")
                        print(step_features.to_markdown(index=False))
                        print('\nFinal forward model after removing insignficant features if any:')
                        print('Best features: ', selected_features,'\n')
                        print(model.summary())
                    break
    
    if data['direction'] == 'backward' and data['criterion'] != 'p_coef':

        # Backward selection to minimize AIC or BIC
        selected_features = list(X.columns)
        remaining_features = []
        istep = 0
        while len(selected_features) > 0:
            score_with_candidates = []        
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

            # start output dataframe of score at each step
            if istep == 0:
                step_features = {'Step': 0, 'AIC': model.aic, 'BIC': model.bic, 
                    'rsq_adj': model.rsquared_adj, 'Features': [np.array(selected_features)]}
                step_features = pd.DataFrame(step_features)
            
            if data['criterion'] == 'AIC':
                best_score = model.aic
            elif data['criterion'] == 'BIC':
                best_score = model.bic
            elif data['criterion'] == 'r2':
                best_score = 1-model.rsquared_adj
            # for candidate in remaining_features:
            for candidate in selected_features:
                # model = sm.OLS(y, sm.add_constant(X[selected_features - [candidate]])).fit()
                test_features = selected_features.copy()
                test_features.remove(candidate)
                model = sm.OLS(y, sm.add_constant(X[test_features])).fit()
                if data['criterion'] == 'AIC':
                    score_with_candidates.append((model.aic, candidate))
                elif data['criterion'] == 'BIC':
                    score_with_candidates.append((model.bic, candidate))
                elif data['criterion'] == 'r2':
                    score_with_candidates.append((1-model.rsquared_adj, candidate))
            score_with_candidates.sort()  # Sort by criterion
            best_new_score, best_candidate = score_with_candidates[0]        
            if best_new_score < best_score:
                best_score = best_new_score
                remaining_features.append(best_candidate)
                selected_features.remove(best_candidate)
                istep += 1
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

                # add new row to output dataframe
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                            
                if data['criterion'] == 'AIC':
                    score = model.aic
                elif data['criterion'] == 'BIC':
                    score = model.bic
                elif data['criterion'] == 'r2':
                    score = model.rsquared_adj
                if (data['verbose'] == 'on' and
                        (selected_features == [] and data['drop_insig'] == 'off')):
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    print(step_features.to_markdown(index=False))
                    print('\nBacksard step '+str(istep)+", "+crit+"= {:.2f}".format(score))
                    print('Features added: ', selected_features,'\n')
                    print(model.summary())        

            else:            
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
               
                if data['verbose'] == 'on' and data['drop_insig'] == 'off':
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    print(step_features.to_markdown(index=False))
                    print('\nFinal backward model before removing insignficant features if any:')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                break            

        if data['drop_insig'] == 'on':
    
            while selected_features:
                # Backward elimination of non-signficant predictors
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                p_values = model.pvalues.iloc[1:]  # Ignore intercept
                max_p_value = p_values.max()
                if max_p_value > p_threshold:
                    worst_feature = p_values.idxmax()
                    selected_features.remove(worst_feature)
                else:
                    model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

                    # add new row to output dataframe
                    new_row = {'Step': istep+1, 'AIC': model.aic, 'BIC': model.bic, 
                               'rsq_adj': model.rsquared_adj, 
                               'Features': np.array(selected_features)}
                    step_features = pd.concat([step_features, 
                            pd.DataFrame([new_row])], ignore_index=True)
                    
                    print("Model skill and features at each step in model_outputs['step_features']:\n")
                    # print(model_outputs['step_features'].to_markdown(index=False))
                    print(step_features.to_markdown(index=False))
                    print('\nFinal backward model after removing insignficant features if any:')
                    print('Best features: ', selected_features,'\n')
                    print(model.summary())
                    break

    if data['direction'] == 'backward' and data['criterion'] == 'p_coef':

        # Backward selection to keep only features with p_coef <= p_threshold
        selected_features = list(X.columns)
        remaining_features = []
        istep = 0
        while selected_features:
            # Backward elimination of non-signficant predictors
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

            # start output dataframe of score at each step
            if istep == 0:
                step_features = {'Step': 0, 'AIC': model.aic, 'BIC': model.bic, 
                    'rsq_adj': model.rsquared_adj, 'Features': [np.array(selected_features)]}
                step_features = pd.DataFrame(step_features)
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
            else:
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                
            p_values = model.pvalues.iloc[1:]  # Ignore intercept
            max_p_value = p_values.max()
            istep += 1
            if max_p_value > p_threshold:
                worst_feature = p_values.idxmax()
                selected_features.remove(worst_feature)
            else:
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

                # add new row to output dataframe
                new_row = {'Step': istep, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                
                print("Model skill and features at each step in model_outputs['step_features']:\n")
                print(step_features.to_markdown(index=False))
                print('\nFinal backward model after removing insignficant features if any:')
                print('Best features: ', selected_features,'\n')
                print(model.summary())
                break
        
    if data['direction'] == 'all':

        # make a list of lists of all possible combinations of features
        list_combinations = []
        for n in range(len(list(X.columns)) + 1):
            list_combinations += list(combinations(list(X.columns), n))

        # loop through all possible combinations and sort by AIC or BIC of each combination
        score_with_candidates = []        
        for i in range(len(list_combinations)):
            selected_features = list(map(str,list_combinations[i]))
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

            if i == 0:
                # output dataframe of score at each step
                step_features = {'Rank': i, 'AIC': model.aic, 'BIC': model.bic, 
                    'rsq_adj': model.rsquared_adj, 'Features': selected_features}
                step_features = pd.DataFrame(step_features)
            else:
                # add new row to output dataframe
                new_row = {'Rank': i, 'AIC': model.aic, 'BIC': model.bic, 
                           'rsq_adj': model.rsquared_adj, 
                           'Features': np.array(selected_features)}
                step_features = pd.concat([step_features, 
                        pd.DataFrame([new_row])], ignore_index=True)
                            
            if data['criterion'] == 'AIC':
                score_with_candidates.append((model.aic, selected_features))
            elif data['criterion'] == 'BIC':
                score_with_candidates.append((model.bic, selected_features))
            elif data['criterion'] == 'r2':
                score_with_candidates.append((1-model.rsquared_adj, selected_features))
        score_with_candidates.sort()  # Sort by criterion
        best_score, selected_features = score_with_candidates[0]        
        model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

        if data['drop_insig'] == 'off':            
            # sort step_features by criterion
            if data['criterion'] == 'AIC':
                step_features = step_features.sort_values(by='AIC')
            elif data['criterion'] == 'BIC':
                step_features = step_features.sort_values(by='BIC')            
            elif data['criterion'] == 'r2':
                step_features = step_features.sort_values(by='rsq_adj', ascending=False)            
            ranks = np.arange(0, step_features.shape[0])
            step_features['Rank'] = ranks        
            # save best 10 subsets of features in step_features
            nhead = min(step_features.shape[0],10)
            step_features = step_features.head(nhead)
        
        if data['verbose'] == 'on' and data['drop_insig'] == 'off':            
            print("Best "+str(nhead)+" subsets of features in model_outputs['step_features']:\n")
            print(step_features.head(nhead).to_markdown(index=False))
            print('\nBest of all possible models before removing insignficant features if any:')
            print('Best features: ', selected_features,'\n')
            print(model.summary())
 
        if data['drop_insig'] == 'on':
    
            while selected_features:
                # Backward elimination of non-signficant predictors
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                p_values = model.pvalues.iloc[1:]  # Ignore intercept
                max_p_value = p_values.max()
                if max_p_value > p_threshold:
                    worst_feature = p_values.idxmax()
                    selected_features.remove(worst_feature)
                else:
                    model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()

                    # add new row to output dataframe
                    new_row = {'Rank': i+1, 'AIC': model.aic, 'BIC': model.bic, 
                               'rsq_adj': model.rsquared_adj, 
                               'Features': np.array(selected_features)}
                    step_features = pd.concat([step_features, 
                            pd.DataFrame([new_row])], ignore_index=True)

                    # sort step_features by criterion
                    if data['criterion'] == 'AIC':
                        step_features = step_features.sort_values(by='AIC')
                    elif data['criterion'] == 'BIC':
                        step_features = step_features.sort_values(by='BIC')            
                    elif data['criterion'] == 'r2':
                        step_features = step_features.sort_values(by='rsq_adj', ascending=False)            
                    ranks = np.arange(0, step_features.shape[0])
                    step_features['Rank'] = ranks        
                    # save best 10 subsets of features in step_features
                    nhead = min(step_features.shape[0],10)
                    step_features = step_features.head(nhead)
                    
                    if data['verbose'] == 'on':
                        print("Best "+str(nhead)+" subsets of features in model_outputs['step_features']:\n")
                        print(step_features.head(nhead).to_markdown(index=False))
                        print('\nBest of all possible models after removing insignficant features if any:')
                        print('Best features: ', selected_features,'\n')
                        print(model.summary())
                    break
            
    # Variance Inflation Factors of selected_features
    # Add a constant for the intercept
    X_ = sm.add_constant(X[selected_features])    
    vif = pd.DataFrame()
    vif['Feature'] = X_.columns.to_list()
    vif["VIF"] = [variance_inflation_factor(X_.values, i)
                        for i in range(len(X_.columns))]
    vif.set_index('Feature',inplace=True)
    if data['verbose'] == 'on':
        print("\nVariance Inflation Factors of selected_features:")
        print("Note: VIF>5 indicates excessive collinearity\n")
        print(vif.to_markdown(index=True))

    # dataframe of model parameters, intercept and coefficients, including zero coefs if any
    n_param = model.params.size               # number of parameters including intercept
    popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
    for i in range(n_param):
        popt[0][i] = model.model.exog_names[i]
        popt[1][i] = model.params[i]
    popt = pd.DataFrame(popt).T
    popt.columns = ['Feature', 'param']
    popt.set_index('Feature',inplace=True)
    
    model_object = model
    # model_output = {}
    # scaler = StandardScaler().fit(X)
    # model_outputs['scaler'] = scaler
    # model_outputs['standardize'] = data['standardize']
    model_outputs['selected_features'] = selected_features
    model_outputs['step_features'] = step_features
    model_outputs['y_pred'] = model.predict(sm.add_constant(X[selected_features]))
    # model_outputs['residuals'] = model_outputs['y_pred'] - y
    model_outputs['residuals'] = y - model_outputs['y_pred']
    model_outputs['popt'] = popt

    # # Get the covariance matrix of parameters including intercept
    # # results = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
    # # cov_matrix = results.cov_params()
    # cov_matrix = model.cov_params()
    # # Exclude the intercept (assumes the intercept is the first parameter)
    # cov_matrix_excluding_intercept = cov_matrix.iloc[1:, 1:]
    X_ = sm.add_constant(X[selected_features])    # Add a constant for the intercept
    pcov = pd.DataFrame(np.cov(X_, rowvar=False), index=X_.columns)
    pcov.columns = X_.columns
    
    model_outputs['pcov'] = pcov
    model_outputs['vif'] = vif

    # Summary statitistics
    list_name = ['r-squared','adjusted r-squared',
        'n_samples','df residuals','df model',
        'F-statistic','Prob (F-statistic)',
        'RMSE',
        'Log-Likelihood','AIC','BIC']
    list_stats = [model.rsquared, model.rsquared_adj,
        len(y), model.df_resid, model.df_model, 
        model.fvalue, model.f_pvalue, 
        np.sqrt(np.mean(model_outputs['residuals']**2)),  
        model.llf,model.aic,model.bic]
    stats = pd.DataFrame(
        {
            "Statistic": list_name,
            "Value": list_stats
        }
        )
    stats.set_index('Statistic',inplace=True)
    model_outputs['stats'] = stats
    model_outputs['summary'] = model.summary()
    
    # plot residuals
    if data['verbose'] == 'on':
        '''
        y_pred = model_outputs['y_pred']
        res = model_outputs['residuals']
        rmse = np.sqrt(np.mean(res**2))
        plt.figure()
        plt.scatter(y_pred, res)
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residual plot for training error, RMSE={:.2f}".format(rmse))
        plt.xlabel('y_pred')
        plt.ylabel('residual')
        plt.savefig("Stepwise_residuals.png", dpi=300)
        ''' 
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        rmse = np.sqrt(np.mean(model_outputs['residuals']**2))
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={rmse:.3f})")
        plt.tight_layout()
        # plt.show()
        plt.savefig("Stepwise_predictions.png", dpi=300)
        
    # Print the run time
    fit_time = time.time() - start_time
    if data['verbose'] == 'on':
        print('')
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")
    
    return model_object, model_outputs

def stats_given_model(X,y,model):

    """
    Calculate linear regression summary statistics 
    from input and output of X, y, and fitted sklearn linear_model  

    by
    Greg Pelletier
    gjpelletier@gmail.com
    12-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = pandas dataframe of the observed independent variables 
        that were used to fit the model
    y = pandas dataframe of the observed dependent variable 
        that was used to fit the model
    model = output model object from sklearn.linear_model 
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    # from sklearn.linear_model import LassoLarsIC
    # from sklearn.linear_model import LassoCV
    import sys

    from PyMLR import check_X_y
    X, y = check_X_y(X,y)
        
    # Calculate regression summary stats
    y_pred = model.predict(X)                   # best fit of the predicted y values
    # residuals = y_pred - y
    residuals = y - y_pred
    n_samples = np.size(y)

    # dataframe of model parameters, intercept and coefficients, including zero coefs
    n_param = 1 + model.coef_.size               # number of parameters including intercept
    popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
    for i in range(n_param):
        if i == 0:
            popt[0][i] = 'const'
            popt[1][i] = model.intercept_
        else:
            popt[0][i] = X.columns[i-1]
            popt[1][i] = model.coef_[i-1]
    popt = pd.DataFrame(popt).T
    popt.columns = ['Feature', 'param']

    n_param = np.count_nonzero(popt['param'])     # number of non-zero param (incl intcpt)
    df = n_samples - n_param
    SSE = np.sum(residuals ** 2)                # sum of squares (residual error)
    MSE = SSE / df                              # mean square (residual error)
    syx = np.sqrt(MSE)                          # standard error of the estimate
    RMSE = np.sqrt(SSE/n_samples)                    # root mean squared error
    SST = np.sum(y **2) - np.sum(y) **2 / n_samples  # sum of squares (total)
    SSR = SST - SSE                             # sum of squares (regression model)
    MSR = SSR / (n_param-1)                      # mean square (regression model)
    Fstat = MSR / MSE                           # F statistic
    dfn = n_param - 1                            # df numerator for F-test
    dfd = df                                    # df denomenator for F-test
    pvalue = 1-stats.f.cdf(Fstat, dfn, dfd)     # p-value of F-test
    rsquared = SSR / SST                                    # ordinary r-squared                                                    # ordinary rsquared
    adj_rsquared = 1-(1-rsquared)*(n_samples-1)/(n_samples-n_param-1)  # adjusted rsquared

    # Calculate Log-Likelihood (LL), AIC, and BIC
    sigma_squared = np.sum(residuals**2) / n_samples  # Variance estimate
    sigma = np.sqrt(sigma_squared)
    log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi) + np.log(sigma_squared) + 1)
    aic = -2 * log_likelihood + 2 * n_param
    bic = -2 * log_likelihood + n_param * np.log(n_samples)

    # Put residuals and y_pred into pandas dataframes to preserve the index of X and y
    df_y_pred = pd.DataFrame(y_pred)
    df_y_pred.index = y.index
    df_y_pred.columns = ['y_pred']    
    df_y_pred = df_y_pred['y_pred']
    df_residuals = pd.DataFrame(residuals)
    df_residuals.index = y.index
    df_residuals.columns = ['residuals']    
    df_residuals = df_residuals['residuals']
        
    # put the results into a dictionary
    result = {
            'X': X,
            'y': y,
            'y_pred': df_y_pred,
            'residuals': df_residuals,
            'model': model,
            'popt': popt,
            'n_samples': n_samples,
            'n_param': n_param,
            'df': df,
            'SST': SST,
            'SSR': SSR,
            'SSE': SSE,
            'MSR': MSR,
            'MSE': MSE,
            'syx': syx,
            'RMSE': RMSE,
            'Fstat': Fstat,
            'dfn': dfn,
            'dfd': dfd,
            'pvalue': pvalue,
            'rsquared': rsquared,
            'adj_rsquared': adj_rsquared,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic        
            }

    return result

def stats_given_y_pred(X,y,y_pred):

    """
    Calculate linear regression summary statistics 
    given X, y, and y_pred from fitted model 
    assuming n_param = number of columns of X

    by
    Greg Pelletier
    gjpelletier@gmail.com
    30-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = pandas dataframe of the observed independent variables 
        that were used to fit the model
    y = pandas dataframe of the observed dependent variable 
        that was used to fit the model
    y_pred = predicted y from fitted model (array or dataframe)
    """

    import numpy as np
    import pandas as pd
    from scipy import stats
    import sys
    
    from PyMLR import check_X_y
    X, y = check_X_y(X,y)

    ctrl = len(y) == len(y_pred)
    if not ctrl:
        print('Check y_pred: it needs to be the same length as y!','\n')
        sys.exit()
        
    n_param = X.shape[1]
    n_samples = len(y)
    # residuals = y_pred - y
    residuals = y - y_pred
    df = n_samples - n_param
    SSE = np.sum(residuals ** 2)                # sum of squares (residual error)
    MSE = SSE / df                              # mean square (residual error)
    syx = np.sqrt(MSE)                          # standard error of the estimate
    RMSE = np.sqrt(SSE/n_samples)                    # root mean squared error
    SST = np.sum(y **2) - np.sum(y) **2 / n_samples  # sum of squares (total)
    SSR = SST - SSE                             # sum of squares (regression model)
    MSR = SSR / (n_param-1)                      # mean square (regression model)
    Fstat = MSR / MSE                           # F statistic
    dfn = n_param - 1                            # df numerator for F-test
    dfd = df                                    # df denomenator for F-test
    pvalue = 1-stats.f.cdf(Fstat, dfn, dfd)     # p-value of F-test
    rsquared = SSR / SST                                    # ordinary r-squared
    adj_rsquared = 1-(1-rsquared)*(n_samples-1)/(n_samples-n_param-1)  # adjusted rsquared
    sigma_squared = np.sum(residuals**2) / n_samples  # Variance estimate
    sigma = np.sqrt(sigma_squared)
    log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi) + np.log(sigma_squared) + 1)
    aic = -2 * log_likelihood + 2 * n_param
    bic = -2 * log_likelihood + n_param * np.log(n_samples)

    # put y_pred and residuals into dataframes
    if not isinstance(y_pred, pd.DataFrame):
        df_y_pred = pd.DataFrame(y_pred)
        df_y_pred.index = y.index
        df_y_pred.columns = ['y_pred']    
        df_y_pred = df_y_pred['y_pred']
    else:
        df_y_pred = y_pred
    df_residuals = pd.DataFrame(residuals)
    df_residuals.index = y.index
    df_residuals.columns = ['residuals']    
    df_residuals = df_residuals['residuals']
    
    # put the results into a dictionary
    result = {
            'X': X,
            'y': y,
            'y_pred': df_y_pred,
            'residuals': df_residuals,
            'n_samples': n_samples,
            'n_param': n_param,
            'df': df,
            'SST': SST,
            'SSR': SSR,
            'SSE': SSE,
            'MSR': MSR,
            'MSE': MSE,
            'syx': syx,
            'RMSE': RMSE,
            'Fstat': Fstat,
            'dfn': dfn,
            'dfd': dfd,
            'pvalue': pvalue,
            'rsquared': rsquared,
            'adj_rsquared': adj_rsquared,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic        
            }

    return result

def lasso(X, y, **kwargs):

    """
    Python function for Lasso linear regression 
    using k-fold cross-validation (CV) or to minimize AIC or BIC

    by
    Greg Pelletier
    gjpelletier@gmail.com
    17-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        nfolds= number of folds to use for cross-validation (CV)
            with k-fold LassoCV or LassoLarsCV (default nfolds=20)
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        alpha_min= minimum value of range of alphas to evaluate (default=1e-3)
        alpha_max= maximum value of range of alphas to evaluate (default=1e3)
        n_alpha= number of log-spaced alphas to evaluate (default=100)
        verbose= 'on' (default), 'off', or 1=show stats and residuals plot
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    It is generally recommended to use a largest possible number of folds 
    for LassoCV and LassoLarsCV to ensure more accurate model selection. 
    The only disadvantage of a large number of folds is the increase 
    computational time. The lasso function allows you to specify 
    the number of folds using the nfolds argument. 
    Using a larger number can lead to better performance. 
    For optimal results, consider experimenting 
    with different fold sizes to find the best balance 
    between performance and speed.

    RETURNS
        model_objects, model_outputs
            model_objects are the fitted model objects from 
                sklearn.linear_model LassoCV, LassoLarsCV, and LassoLarsIC
                of the final best models using the following four methods: 
                - LassoCV: k-fold CV coordinate descent
                - LassoLarsCV: k-fold CV least angle regression
                - LassoLarsAIC: LassoLarsIC using AIC
                - LassoLarsBIC: LasspLarsIC using BIC
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'alpha_vs_coef': model coefficients for each X variable
                    as a function of alpha using Lasso
                - 'alpha_vs_AIC_BIC': AIC and BIC as a function of alpha 
                    using LassoLarsIC
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'popt': Constant (intercept) and coefficients for the 
                    best fit models from each of the four methods
                - 'popt_table': Constant (intercept) and coefficients
                    of best fit of all four methods in one table
                - 'pcov': Covariance matrix of features 
                - 'vif': Variance Inlfation Factors of features of each method
                - 'vif_table': Variance Inlfation Factors of features of 
                    all four methods in one table
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = lasso(X, y)

    """

    from PyMLR import stats_given_model, detect_dummy_variables
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    import time
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import LassoLarsCV
    from sklearn.linear_model import LassoLarsIC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import PredictionErrorDisplay
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    
   
    # Define default values of input data arguments
    defaults = {
        'nfolds': 20,
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'alpha_min': 1.0e-3,
        'alpha_max': 1.0e3,
        'n_alpha': 100,
        'verbose': 'on'
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']
            
    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]


    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    ctrl = data['alpha_min'] > 0 
    if not ctrl:
        print('Check input of alpha_min, it must be greater than zero!','\n')
        sys.exit()
    ctrl = data['alpha_max'] > data['alpha_min'] 
    if not ctrl:
        print('Check input of alpha_max, it must be greater than alpha_min!','\n')
        sys.exit()
    ctrl = data['n_alpha'] > 1 
    if not ctrl:
        print('Check inputs of n_alpha, it must be greater than 1!','\n')
        sys.exit()

    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting Lasso regression models, please wait ...')
    if data['verbose'] == 'on' or data['verbose'] == 1:
        print('')

    # Calculate the role of alpha vs coefficient values
    alpha_min = np.log10(data['alpha_min'])
    alpha_max = np.log10(data['alpha_max'])    
    n_alpha = data['n_alpha']    
    alphas = 10**np.linspace(alpha_min,alpha_max,n_alpha)
    # alphas = 10**np.linspace(-3,3,100)
    lasso = Lasso(max_iter=10000)
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X, y)
        coefs.append(lasso.coef_)
    alpha_vs_coef = pd.DataFrame({
        'alpha': alphas,
        'coef': coefs
        }).set_index("alpha")
    model_outputs['alpha_vs_coef'] = alpha_vs_coef

    # LassoCV k-fold cross validation via coordinate descent
    model_cv = LassoCV(cv=data['nfolds'], random_state=0, max_iter=10000).fit(X, y)
    model_objects['LassoCV'] = model_cv
    alpha_cv = model_cv.alpha_

    # LassoLarsCV k-fold cross validation via least angle regression
    model_lars_cv = LassoLarsCV(cv=data['nfolds'], max_iter=10000).fit(X, y)
    model_objects['LassoLarsCV'] = model_lars_cv
    alpha_lars_cv = model_lars_cv.alpha_

    # LassoLarsIC minimizing AIC
    model_aic = LassoLarsIC(criterion="aic", max_iter=10000).fit(X, y)
    model_objects['LassoLarsAIC'] = model_aic
    alpha_aic = model_aic.alpha_

    # LassoLarsIC minimizing BIC
    model_bic = LassoLarsIC(criterion="bic", max_iter=10000).fit(X, y)
    model_objects['LassoLarsBIC'] = model_bic
    alpha_bic = model_bic.alpha_

    # results of alphas to minimize AIC and BIC
    alpha_vs_AIC_BIC = pd.DataFrame(
        {
            "alpha": model_aic.alphas_,
            "AIC": model_aic.criterion_,
            "BIC": model_bic.criterion_,
        }
        ).set_index("alpha")
    model_outputs['alpha_vs_AIC_BIC'] = alpha_vs_AIC_BIC

    # Lasso Plot the results of lasso coef as function of alpha
    if data['verbose'] == 'on' and data['verbose'] != 1:
        ax = plt.gca()
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.axis('tight')
        plt.xlabel(r"$\alpha$")
        if X.shape[1] < 20:
            plt.legend(X.columns)
        plt.ylabel('Coefficients')
        plt.title(r'Lasso regression coefficients as a function of $\alpha$');
        plt.savefig("Lasso_alpha_vs_coef.png", dpi=300)

    # LassoCV Plot the MSE vs alpha for each fold
    if data['verbose'] == 'on' and data['verbose'] != 1:
        lasso = model_cv
        plt.figure()
        plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
        plt.plot(
            lasso.alphas_,
            lasso.mse_path_.mean(axis=-1),
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(lasso.alpha_, linestyle="--", color="black", 
                    label="CV selected alpha={:.3e}".format(model_cv.alpha_))        
        # ymin, ymax = 2300, 3800
        # plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean Square Error")
        plt.legend()
        _ = plt.title(
            "LassoCV - Mean Square Error on each fold: coordinate descent"
        )
        plt.savefig("LassoCV_alpha_vs_MSE.png", dpi=300)

    # LassoLarsCV Plot the MSE vs alpha for each fold
    if data['verbose'] == 'on' and data['verbose'] != 1 and X.shape[1] < 250:
        lasso = model_lars_cv
        plt.figure()
        plt.semilogx(lasso.cv_alphas_, lasso.mse_path_, ":")
        plt.semilogx(
            lasso.cv_alphas_,
            lasso.mse_path_.mean(axis=-1),
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(lasso.alpha_, linestyle="--", color="black", 
                    label="LarsCV selected alpha={:.3e}".format(model_lars_cv.alpha_))

        # plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean Square Error")
        plt.legend()
        _ = plt.title(f"LassoLarsCV - Mean Square Error on each fold: LARS")
        plt.savefig("LassoLarsCV_alpha_vs_MSE.png", dpi=300)

    # LassoLarsIC Plot of alphas to minimize AIC and BIC
    if data['verbose'] == 'on' and data['verbose'] != 1:
        results = alpha_vs_AIC_BIC
        ax = results.plot()
        ax.vlines(
            alpha_aic,
            results["AIC"].min(),
            results["AIC"].max(),
            label="AIC selected alpha={:.3e}".format(model_aic.alpha_),
            linestyles="--",
            color="tab:blue",
        )
        ax.vlines(
            alpha_bic,
            results["BIC"].min(),
            results["BIC"].max(),
            label="BIC selected alpha={:.3e}".format(model_bic.alpha_),
            linestyle="--",
            color="tab:orange",
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Information Criterion (AIC or BIC)")
        ax.set_xscale("log")
        ax.legend()
        _ = ax.set_title(
            "LassoLarsIC - Information Criterion for model selection"
        )
        plt.savefig("LassoLarsIC_alpha_vs_AIC_BIC.png", dpi=300)

    # LassoLarsIC Plot sequence of alphas to minimize AIC and BIC
    if data['verbose'] == 'on' and data['verbose'] != 1:
        plt.figure()
        aic_criterion = model_aic.criterion_
        bic_criterion = model_bic.criterion_
        index_alpha_path_aic = np.flatnonzero(model_aic.alphas_ == model_aic.alpha_)[0]
        index_alpha_path_bic = np.flatnonzero(model_bic.alphas_ == model_bic.alpha_)[0]
        # print('check index alpha: ',index_alpha_path_aic == index_alpha_path_bic)
        plt.plot(aic_criterion, color="tab:blue", marker="o", label="AIC criterion")
        plt.plot(bic_criterion, color="tab:orange", marker="o", label="BIC criterion")
        # vline for alpha for aic
        plt.vlines(
            index_alpha_path_aic,
            aic_criterion.min(),
            aic_criterion.max(),
            color="tab:blue",
            linestyle="--",
            label="AIC selected alpha={:.3e}".format(model_aic.alpha_),
        )
        # vline for alpha for bic
        plt.vlines(
            index_alpha_path_bic,
            aic_criterion.min(),
            aic_criterion.max(),
            color="tab:orange",
            linestyle="--",
            label="BIC selected alpha={:.3e}".format(model_bic.alpha_),
        )
        plt.legend()
        plt.ylabel("Information Criterion (AIC or BIC)")
        plt.xlabel("Lasso model sequence")
        _ = plt.title("LassoLarsIC - Model sequence of AIC and BIC")
        plt.savefig("LassoLarsIC_sequence_of_AIC_BIC.png", dpi=300)

    # Calculate regression stats
    stats_cv = stats_given_model(X, y, model_cv)
    stats_lars_cv = stats_given_model(X, y, model_lars_cv)
    stats_aic = stats_given_model(X, y, model_aic)
    stats_bic = stats_given_model(X, y, model_bic)

    # residual plot for training error
    if data['verbose'] == 'on' or data['verbose'] == 1:
        # plot predictions vs actual
        y_pred_cv = stats_cv['y_pred']
        y_pred_lars_cv = stats_lars_cv['y_pred']
        y_pred_aic = stats_aic['y_pred']
        y_pred_bic = stats_bic['y_pred']
        res_cv = stats_cv['residuals']
        res_lars_cv = stats_lars_cv['residuals']
        res_aic = stats_aic['residuals']
        res_bic = stats_bic['residuals']
        rmse_cv = stats_cv['RMSE']
        rmse_lars_cv = stats_lars_cv['RMSE']
        rmse_aic = stats_aic['RMSE']
        rmse_bic = stats_bic['RMSE']
        plt.figure()
        plt.scatter(y_pred_cv, y, s=40, label=('LassoCV (RMSE={:.2f})'.format(rmse_cv)))
        plt.scatter(y_pred_lars_cv, y, s=15, label=('LassoLarsCV (RMSE={:.2f})'.format(rmse_lars_cv)))
        plt.scatter(y_pred_aic, y, s=10, label=('LassoLarsAIC (RMSE={:.2f})'.format(rmse_aic)))
        plt.scatter(y_pred_bic, y, s=5, label=('LassoLarsBIC (RMSE={:.2f})'.format(rmse_bic)))

        # plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        y45 = np.linspace(min(y), max(y), 100)  # Adjust range as needed
        x45 = y45  # 45-degree line: y = x
        plt.plot(x45, y45, color="k")

        plt.title("Actual vs. Predicted")
        plt.legend();
        plt.xlabel('y_pred')
        plt.ylabel('y')
        plt.savefig("Lasso_predictions_vs_actual.png", dpi=300)
        # plot predictions vs residuals
        y_pred_cv = stats_cv['y_pred']
        y_pred_lars_cv = stats_lars_cv['y_pred']
        y_pred_aic = stats_aic['y_pred']
        y_pred_bic = stats_bic['y_pred']
        res_cv = stats_cv['residuals']
        res_lars_cv = stats_lars_cv['residuals']
        res_aic = stats_aic['residuals']
        res_bic = stats_bic['residuals']
        rmse_cv = stats_cv['RMSE']
        rmse_lars_cv = stats_lars_cv['RMSE']
        rmse_aic = stats_aic['RMSE']
        rmse_bic = stats_bic['RMSE']
        plt.figure()
        plt.scatter(y_pred_cv, (res_cv), s=40, label=('LassoCV (RMSE={:.2f})'.format(rmse_cv)))
        plt.scatter(y_pred_lars_cv, (res_lars_cv), s=15, label=('LassoLarsCV (RMSE={:.2f})'.format(rmse_lars_cv)))
        plt.scatter(y_pred_aic, (res_aic), s=10, label=('LassoLarsAIC (RMSE={:.2f})'.format(rmse_aic)))
        plt.scatter(y_pred_bic, (res_bic), s=5, label=('LassoLarsBIC (RMSE={:.2f})'.format(rmse_bic)))
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residuals vs. Predicted")
        plt.legend();
        plt.xlabel('y_pred')
        plt.ylabel('residual')
        plt.savefig("Lasso_predictions_vs_residuals.png", dpi=300)

    # Find the AIC and BIC of the LassoLarsAIC and LassoLarsBIC models
    min_index_aic = model_outputs['alpha_vs_AIC_BIC']['AIC'].idxmin()
    min_index_bic = model_outputs['alpha_vs_AIC_BIC']['BIC'].idxmin()
    AIC_for_LassoLarsAIC = model_outputs['alpha_vs_AIC_BIC']['AIC'][min_index_aic]
    BIC_for_LassoLarsAIC = model_outputs['alpha_vs_AIC_BIC']['BIC'][min_index_aic]
    AIC_for_LassoLarsBIC = model_outputs['alpha_vs_AIC_BIC']['AIC'][min_index_bic]
    BIC_for_LassoLarsBIC = model_outputs['alpha_vs_AIC_BIC']['BIC'][min_index_bic]

    # Make the model_outputs dataframes
    list1_name = ['alpha','r-squared','adjusted r-squared',
                        'n_samples','df residuals','df model',
                        'F-statistic','Prob (F-statistic)','RMSE',
                        'Log-Likelihood','AIC','BIC']
    list2_name = list(stats_cv['popt']['Feature'])
    list3_name = list1_name + list2_name

    list1_cv = [model_cv.alpha_, stats_cv["rsquared"], stats_cv["adj_rsquared"],
                       stats_cv["n_samples"], stats_cv["df"], stats_cv["dfn"], 
                       stats_cv["Fstat"], stats_cv["pvalue"], stats_cv["RMSE"],  
                       stats_cv["log_likelihood"],stats_cv["aic"],stats_cv["bic"]]
    list2_cv = list(stats_cv['popt']['param'])
    list3_cv = list1_cv + list2_cv

    list1_lars_cv = [model_lars_cv.alpha_, stats_lars_cv["rsquared"], stats_lars_cv["adj_rsquared"], 
                       stats_lars_cv["n_samples"], stats_lars_cv["df"], stats_lars_cv["dfn"], 
                       stats_lars_cv["Fstat"], stats_lars_cv["pvalue"], stats_lars_cv["RMSE"], 
                       stats_lars_cv["log_likelihood"],stats_lars_cv["aic"],stats_lars_cv["bic"]]
    list2_lars_cv = list(stats_lars_cv['popt']['param'])
    list3_lars_cv = list1_lars_cv + list2_lars_cv

    list1_aic = [model_aic.alpha_, stats_aic["rsquared"], stats_aic["adj_rsquared"], 
                       stats_aic["n_samples"], stats_aic["df"], stats_aic["dfn"], 
                       stats_aic["Fstat"], stats_aic["pvalue"], stats_aic["RMSE"], 
                       stats_aic["log_likelihood"],AIC_for_LassoLarsAIC,BIC_for_LassoLarsAIC]
    list2_aic = list(stats_aic['popt']['param'])
    list3_aic = list1_aic + list2_aic

    list1_bic = [model_bic.alpha_, stats_bic["rsquared"], stats_bic["adj_rsquared"], 
                       stats_bic["n_samples"], stats_bic["df"], stats_bic["dfn"], 
                       stats_bic["Fstat"], stats_bic["pvalue"], stats_bic["RMSE"], 
                       stats_bic["log_likelihood"],AIC_for_LassoLarsBIC,BIC_for_LassoLarsBIC]
    list2_bic = list(stats_bic['popt']['param'])
    list3_bic = list1_bic + list2_bic

    y_pred = pd.DataFrame(
        {
            "LassoCV": stats_cv['y_pred'],
            "LassoLarsCV": stats_lars_cv['y_pred'],
            "LassoLarsAIC": stats_aic['y_pred'],
            "LassoLarsBIC": stats_bic['y_pred']
        }
        )
    y_pred.index = y.index
    model_outputs['y_pred'] = y_pred

    residuals = pd.DataFrame(
        {
            "LassoCV": stats_cv['residuals'],
            "LassoLarsCV": stats_lars_cv['residuals'],
            "LassoLarsAIC": stats_aic['residuals'],
            "LassoLarsBIC": stats_bic['residuals']
        }
        )
    residuals.index = y.index
    model_outputs['residuals'] = residuals

    # Table of all popt incl coef=0
    popt_table = pd.DataFrame(
        {
            "Feature": list2_name,
            "LassoCV": list2_cv,
            "LassoLarsCV": list2_lars_cv,
            "LassoLarsAIC": list2_aic,
            "LassoLarsBIC": list2_bic
        }
        )
    popt_table.set_index('Feature',inplace=True)
    model_outputs['popt_table'] = popt_table
    
    # Calculate the covariance matrix of the features
    # popt, pcov, and vif of only the selected features (excl coef=0)
    popt_all = {}
    pcov_all = {}
    vif_all = {}
    # vif = pd.DataFrame()
    col = X.columns
    # LassoCV
    model_ = model_objects['LassoCV']
    popt = stats_cv['popt'].copy()
    X_ = X.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not X_has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        pcov_all['LassoCV'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = [variance_inflation_factor(X__.values, i)
                            for i in range(len(X__.columns))]
        vif.set_index('Feature',inplace=True)
        vif_all["LassoCV"] = vif
    popt.set_index('Feature',inplace=True)
    popt_all['LassoCV'] = popt
    # LassoLarsCV
    model_ = model_objects['LassoLarsCV']
    popt = stats_lars_cv['popt'].copy()
    X_ = X.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not X_has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        pcov_all['LassoLarsCV'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = [variance_inflation_factor(X__.values, i)
                            for i in range(len(X__.columns))]
        vif.set_index('Feature',inplace=True)
        vif_all["LassoLarsCV"] = vif
    popt.set_index('Feature',inplace=True)
    popt_all['LassoLarsCV'] = popt
    # LassoLarsAIC
    model_ = model_objects['LassoLarsAIC']
    popt = stats_aic['popt'].copy()
    X_ = X.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not X_has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        pcov_all['LassoLarsAIC'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = [variance_inflation_factor(X__.values, i)
                            for i in range(len(X__.columns))]
        vif.set_index('Feature',inplace=True)
        vif_all["LassoLarsAIC"] = vif
    popt.set_index('Feature',inplace=True)
    popt_all['LassoLarsAIC'] = popt
    # LassoLarsBIC
    model_ = model_objects['LassoLarsBIC']
    popt = stats_bic['popt'].copy()
    X_ = X.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not X_has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        pcov_all['LassoLarsBIC'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = [variance_inflation_factor(X__.values, i)
                            for i in range(len(X__.columns))]
        vif.set_index('Feature',inplace=True)
        vif_all["LassoLarsBIC"] = vif
    popt.set_index('Feature',inplace=True)
    popt_all['LassoLarsBIC'] = popt
    # save pcov, vif, popt
    if not X_has_dummies:
        # save vif and pcov
        model_outputs['vif'] = vif_all
        model_outputs['pcov'] = pcov_all
    model_outputs['popt'] = popt_all

    if not X_has_dummies:
        # Make big VIF table of all models in one table
        # get row indicdes of non-zero coef values in each model col
        idx = popt_table.apply(lambda col: col[col != 0].index.tolist())
        # initialize vif_table same as popt_table but with nan values
        vif_table = pd.DataFrame(np.nan, index=popt_table.index, columns=popt_table.columns)
        # Put in the VIF values in each model column
        # LassoCV
        vif = model_outputs['vif']['LassoCV']['VIF'].values
        vif_table.loc[idx['LassoCV'], "LassoCV"] = vif
        # LassoLarsCV
        vif = model_outputs['vif']['LassoLarsCV']['VIF'].values
        vif_table.loc[idx['LassoLarsCV'], "LassoLarsCV"] = vif
        # LassoLarsAIC
        vif = model_outputs['vif']['LassoLarsAIC']['VIF'].values
        vif_table.loc[idx['LassoLarsAIC'], "LassoLarsAIC"] = vif
        # LassoLarsBIC
        vif = model_outputs['vif']['LassoLarsBIC']['VIF'].values
        vif_table.loc[idx['LassoLarsBIC'], "LassoLarsBIC"] = vif
        model_outputs['vif_table'] = vif_table
    
    stats = pd.DataFrame(
        {
            "Statistic": list1_name,
            "LassoCV": list1_cv,
            "LassoLarsCV": list1_lars_cv,
            "LassoLarsAIC": list1_aic,
            "LassoLarsBIC": list1_bic
        }
        )
    stats.set_index('Statistic',inplace=True)
    model_outputs['stats'] = stats
    
    # Print model_outputs
    if data['verbose'] == 'on' or data['verbose'] == 1:
        print("Lasso regression statistics of best models in model_outputs['stats']:")
        print('')
        print(model_outputs['stats'].to_markdown(index=True))
        print('')
        if data['verbose'] != 1:
            print("Coefficients of best models in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
            if not X_has_dummies:
                print("Variance Inflation Factors model_outputs['vif']:")
                print("Note: VIF>5 indicates excessive collinearity")
                print('')
                print(model_outputs['vif_table'].to_markdown(index=True))
                print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return model_objects, model_outputs

def vif_ridge(X, pen_factors, is_corr=False):

    """
    Variance Inflation Factor for Ridge regression 

    adapted from statsmodels function vif_ridge by Josef Perktold https://gist.github.com/josef-pkt
    source: https://github.com/statsmodels/statsmodels/issues/1669
    source: https://stackoverflow.com/questions/23660120/variance-inflation-factor-in-ridge-regression-in-python
    author: https://stackoverflow.com/users/333700/josef
    Josef is statsmodels maintainer and developer, semi-retired from scipy.stats maintainance

    assumes penalization is on standardized feature variables
    assumes alpha is scaled by n_samples in calc of penalty factors if using sklearn Ridge (see note below)
    data should not include a constant

    Parameters
    ----------
    X : array_like with dimension n_samples x n_features
        correlation matrix if is_corr=True or standardized feature data if is_corr is False (default).
    pen_factors : iterable array of of regularization penalty factors with dimension n_alpha 
        If you are using sklearn Ridge for the analysis, then:
            pen_factor = alphas / n_samples
        If you are using statsmodels OLS .fit_regularized(L1_wt=0, then:
            pen_factor = alphas
        where alphas is the iterable array of alpha inputs to sklearn or statsmodels
        (see explanation in note below for difference between sklearn and statsmodels)
    is_corr : bool (default False)
        Boolean to indicate how corr_x is interpreted, see corr_x

    Returns
    -------
    vif : ndarray
        variance inflation factors for parameters in columns and 
        ridge penalization factors in rows

    could be optimized for repeated calculations

    Note about scaling of alpha in statsmodels vs sklearn 
    -------
    An analysis by Paul Zivich (https://sph.unc.edu/adv_profile/paul-zivich/) explains 
    how to get the same results of ridge regression from statsmodels and sklearn. 
    The difference is that sklearn's Ridge function scales the input of the 'alpha' 
    regularization term during excecution as alpha / n where n is the number of observations, 
    compared with statsmodels which does not apply this scaling of the regularization 
    parameter during execution. You can have the ridge implementations match 
    if you re-scale the sklearn input alpha = alpha / n for statsmodels. 
    Note that this rescaling of alpha only applies to ridge regression. 
    The sklearn and statsmodels results for Lasso regression using exactly 
    the same alpha values for input without rescaling.
    
    Here is a link to the original post of this analysis by Paul Zivich:
    
    https://stackoverflow.com/questions/72260808/mismatch-between-statsmodels-and-sklearn-ridge-regression
    
    -------
    Example use of vif_ridge using sklearn for the analysis:
    
    from sklearn.datasets import load_diabetes
    from PyMLR import vif_ridge
    import numpy as np
    import pandas as pd
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    n_samples = X.shape[0]
    alphas = np.array([0.01,0.1,1,10,100])
    pen_factors = alphas / n_samples 
    vifs = pd.DataFrame(vif_ridge(X, pen_factors))
    vifs.columns = X.columns
    vifs.index = alphas
    vifs = vifs.rename_axis("alpha")
    print(vifs)
    
    Output table of VIF vs alpha for each column of X:
                 age       sex       bmi        bp         s1         s2  \
    alpha                                                                  
    0.01    1.217226  1.277974  1.509267  1.459302  58.892664  38.997322   
    0.10    1.216506  1.277102  1.507746  1.458177  56.210859  37.300175   
    1.00    1.209504  1.268643  1.493706  1.447171  37.168552  25.227946   
    10.00   1.145913  1.192754  1.384258  1.347925   4.766889   4.250315   
    100.00  0.724450  0.724378  0.781325  0.767933   0.313543   0.465363   

                   s3        s4         s5        s6  
    alpha                                             
    0.01    15.338432  8.881508  10.032564  1.484506  
    0.10    14.786345  8.798110   9.656775  1.483458  
    1.00    10.827196  8.107172   6.979474  1.473147  
    10.00    3.300027  4.946247   2.222185  1.378038  
    100.00   0.615300  0.690887   0.758717  0.791855      
    """

    import numpy as np
    
    X = np.asarray(X)
    if not is_corr:
        # corr = np.corrcoef(X, rowvar=0, bias=True)    # bias is deprecated and has no effect
        corr = np.corrcoef(X, rowvar=0)
    else:
        corr = X

    eye = np.eye(corr.shape[1])
    res = []
    for k in pen_factors:
        minv = np.linalg.inv(corr + k * eye)
        vif = minv.dot(corr).dot(minv)
        res.append(np.diag(vif))

    return np.asarray(res)

def ridge(X, y, **kwargs):

    """
    Python function for Ridge linear regression 
    
    by
    Greg Pelletier
    gjpelletier@gmail.com
    21-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        alpha_min= minimum value of range of alphas to evaluate (default=1e-3)
        alpha_max= maximum value of range of alphas to evaluate (default=1e3)
        n_alpha= number of log-spaced alphas to evaluate (default=100)
        vif_target= VIF target for use with RidgeVIF (default=1.0)
        verbose= 'on' (default), 'off', or 1=show stats and residuals plot
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        model_objects, model_outputs
            model_objects are the fitted model objects from 
                sklearn.linear_model Ridge or RidgeCV
                of the final best models using the following four methods: 
                - RidgeCV: sklearn RidgeCV 
                - RidgeVIF: sklearn Ridge using target VIF to find best alpha
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'alpha_vs_coef': model coefficients for each X variable
                    as a function of alpha using Ridge
                - 'alpha_vs_penalty': penalty factors
                    as a function of alpha using Ridge
                - 'best_alpha_vif': alpha at the VIF closest to the target VIF value from RidgeVIF
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the methods
                - 'popt': Constant (intercept) and coefficients for the 
                    best fit models from each of the methods
                - 'popt_table': Constant (intercept) and coefficients
                    of best fit of all methods in one table
                - 'pcov': Covariance matrix of features 
                - 'vif': Variance Inlfation Factors of features of each method
                - 'vif_table': Variance Inlfation Factors of features of 
                    all methods in one table
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = ridge(X, y)

    """

    from PyMLR import stats_given_model, vif_ridge, detect_dummy_variables
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    import time
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import PredictionErrorDisplay
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import warnings
    import sys
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    
   
    # Define default values of input data arguments
    defaults = {
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'alpha_min': 1.0e-3,
        'alpha_max': 1.0e3,
        'n_alpha': 100,
        'vif_target': 1.0,
        'verbose': 'on'
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # check for input errors
    has_dummies = detect_dummy_variables(X)
    # if ctrl:
    #     print('Check X: Ridge can not handle dummies. Try using lasso if X has dummies.','\n')
    #     sys.exit()

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']
            
    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    ctrl = data['alpha_min'] > 0 
    if not ctrl:
        print('Check inputs of alpha_min, it must be greater than zero!','\n')
        sys.exit()
    ctrl = data['alpha_max'] > data['alpha_min'] 
    if not ctrl:
        print('Check inputs of alpha_max, it must be greater than alpha_min!','\n')
        sys.exit()
    ctrl = data['n_alpha'] > 1 
    if not ctrl:
        print('Check inputs of n_alpha, it must be greater than 1!','\n')
        sys.exit()
        
    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting Ridge regression models, please wait ...')
    if data['verbose'] == 'on' or data['verbose'] == 1:
        print('')

    # Calculate the role of alpha vs coefficient values
    alpha_min = np.log10(data['alpha_min'])
    alpha_max = np.log10(data['alpha_max'])    
    n_alpha = data['n_alpha']    
    alphas = 10**np.linspace(alpha_min,alpha_max,n_alpha)
    # alphas = np.logspace(data['alpha_min'],data['alpha_max'],data['n_alpha'])
    # ridge = Ridge(max_iter=15000)
    ridge = Ridge()
    coefs = []
    pen_factors = []
    n_samples, n_columns = X.shape   # sklearn ridge scales alpha by n_samples
    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)

    if not has_dummies:
        pen_factors = alphas / n_samples   # use this line if using sklearn Ridge
        alpha_vs_coef = pd.DataFrame({
            'alpha': alphas,
            'coef': coefs
            }).set_index("alpha")
        alpha_vs_penalty = pd.DataFrame({
            'alpha': alphas,
            'pen_factors':pen_factors
            }).set_index("alpha")
        vifs = pd.DataFrame(vif_ridge(X, pen_factors))
        vifs.columns = X.columns
        vifs_ = vifs.copy()     # vifs_ = vifs before inserting alphas
        vifs.insert(0, 'alpha', alphas)
        vifs.set_index('alpha',inplace=True)
        model_outputs['alpha_vs_vif'] = vifs        
        model_outputs['alpha_vs_coef'] = alpha_vs_coef
        model_outputs['alpha_vs_penalty'] = alpha_vs_penalty
    
    # RidgeCV default using MSE
    model_cv = RidgeCV(alphas=alphas, store_cv_results=True).fit(X, y)
    model_objects['RidgeCV'] = model_cv
    alpha_cv = model_cv.alpha_
    # Get the cross-validated MSE for each alpha
    model_cv_mse_each_fold = model_cv.cv_results_  # Shape: (n_samples, n_alphas)
    model_cv_mse_mean = np.mean(model_cv.cv_results_, axis=0)

    if not has_dummies:
        # RidgeVIF - Ridge with VIF target
        vif_target = data['vif_target']
        rmse_vif_res = np.sqrt(np.sum((vif_target-vifs_)**2,1))
        idx = (np.abs(rmse_vif_res)).argmin()
        best_alpha_vif = alphas[idx]
        model_vif = Ridge(alpha=best_alpha_vif).fit(X, y)
        model_objects['RidgeVIF'] = model_vif  
        model_outputs['best_alpha_vif'] = best_alpha_vif
    
    # Plot the results of ridge coef as function of alpha
    if data['verbose'] == 'on' and data['verbose'] != 1:
        ax = plt.gca()
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.axis('tight')
        plt.xlabel(r"$\alpha$")
        plt.legend(X.columns)
        plt.ylabel('Coefficients')
        plt.title(r'Ridge regression coefficients as a function of $\alpha$');
        plt.savefig("Ridge_alpha_vs_coef.png", dpi=300)

    # Plot the VIF of coefficients as function of alpha
    if not has_dummies:
        if data['verbose'] == 'on' and data['verbose'] != 1:
            # model = model_vif
            plt.figure()
            ax = plt.gca()
            ax.plot(alphas, vifs)
            ax.set_xscale('log')
            ax.set_yscale('log')
            # plt.yscale("log")
            plt.axvline(best_alpha_vif, linestyle="--", color="black")        
            ax.text(best_alpha_vif, np.percentile(vifs.values.flatten(),5), 
                    "alpha at VIF target {:.1f} ={:.3e}".format(vif_target,best_alpha_vif), 
                    rotation=90, va='bottom', ha='right')
            plt.axis('tight')
            plt.xlabel(r"$\alpha$")
            plt.legend(X.columns)
            plt.ylabel('VIF')
            plt.title(r'VIF of coefficients as a function of $\alpha$');
            ax2 = ax.twinx()
            ax2.plot(alphas, rmse_vif_res, 'r--', label='target')
            ax2.set_ylabel('RMS difference between VIF and target {:.1f}'.format(vif_target), color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_yscale('log')
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            plt.savefig("Ridge_alpha_vs_vif.png", dpi=300)

    # RidgeCV Plot the MSE vs alpha for each fold
    if data['verbose'] == 'on' and data['verbose'] != 1:
        model = model_cv
        plt.figure()
        plt.semilogx(alphas, model_cv_mse_each_fold.T, linestyle=":")
        plt.plot(
            alphas,
            model_cv_mse_mean,
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(model.alpha_, linestyle="--", color="black", 
                    label="CV selected alpha={:.3e}".format(model_cv.alpha_))        
        # ymin, ymax = 2300, 3800
        # plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean Square Error")
        # plt.yscale("log")
        plt.legend()
        _ = plt.title(
            "RidgeCV - Mean Square Error on each fold"
        )
        plt.savefig("RidgeCV_alpha_vs_MSE.png", dpi=300)
    
    # Calculate regression stats
    stats_cv = stats_given_model(X, y, model_cv)    
    if not has_dummies:
        stats_vif = stats_given_model(X, y, model_vif)

    # residual plot for training error
    if data['verbose'] == 'on' or data['verbose'] == 1:
        # predicted vs actual
        y_pred_cv = stats_cv['y_pred']
        res_cv = stats_cv['residuals']
        rmse_cv = stats_cv['RMSE']
        plt.figure()
        plt.scatter(y_pred_cv, y, s=40, label=('RidgeCV (RMSE={:.2f})'.format(rmse_cv)))
        if not has_dummies:
            y_pred_vif = stats_vif['y_pred']
            res_vif = stats_vif['residuals']
            rmse_vif = stats_vif['RMSE']
            plt.scatter(y_pred_vif, y, s=5, label=('RidgeVIF (RMSE={:.2f})'.format(rmse_vif)))
        y45 = np.linspace(min(y), max(y), 100)  # Adjust range as needed
        x45 = y45  # 45-degree line: y = x
        plt.plot(x45, y45, color="k")

        plt.title("Actual vs. Predicted")
        plt.legend();
        plt.xlabel('y_pred')
        plt.ylabel('y')
        plt.savefig("Ridge_predicted_vs_actual.png", dpi=300)
        # predicted vs residual
        y_pred_cv = stats_cv['y_pred']
        res_cv = stats_cv['residuals']
        rmse_cv = stats_cv['RMSE']
        plt.figure()
        plt.scatter(y_pred_cv, (res_cv), s=40, label=('RidgeCV (RMSE={:.2f})'.format(rmse_cv)))
        if not has_dummies:
            y_pred_vif = stats_vif['y_pred']
            res_vif = stats_vif['residuals']
            rmse_vif = stats_vif['RMSE']
            plt.scatter(y_pred_vif, (res_vif), s=5, label=('RidgeVIF (RMSE={:.2f})'.format(rmse_vif)))
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residuals vs. Predicted")
        plt.legend();
        plt.xlabel('y_pred')
        plt.ylabel('residual')
        plt.savefig("Ridge_predicted_vs_residuals.png", dpi=300)

    # Make the model_outputs dataframes
    list1_name = ['alpha','r-squared','adjusted r-squared',
                        'n_samples','df residuals','df model',
                        'F-statistic','Prob (F-statistic)','RMSE',
                        'Log-Likelihood','AIC','BIC']
    list2_name = list(stats_cv['popt']['Feature'])
    list3_name = list1_name + list2_name

    list1_cv = [model_cv.alpha_, stats_cv["rsquared"], stats_cv["adj_rsquared"],
                       stats_cv["n_samples"], stats_cv["df"], stats_cv["dfn"], 
                       stats_cv["Fstat"], stats_cv["pvalue"], stats_cv["RMSE"],  
                       stats_cv["log_likelihood"],stats_cv["aic"],stats_cv["bic"]]
    list2_cv = list(stats_cv['popt']['param'])
    list3_cv = list1_cv + list2_cv

    if not has_dummies:
        list1_vif = [best_alpha_vif, stats_vif["rsquared"], stats_vif["adj_rsquared"], 
                           stats_vif["n_samples"], stats_vif["df"], stats_vif["dfn"], 
                           stats_vif["Fstat"], stats_vif["pvalue"], stats_vif["RMSE"], 
                           stats_vif["log_likelihood"],stats_vif["aic"],stats_vif["bic"]]
        list2_vif = list(stats_vif['popt']['param'])
        list3_vif = list1_vif + list2_vif
        y_pred = pd.DataFrame(
            {
                "RidgeCV": stats_cv['y_pred'],
                "RidgeVIF": stats_vif['y_pred']
            }
            )
        y_pred.index = y.index
        model_outputs['y_pred'] = y_pred
        residuals = pd.DataFrame(
            {
                "RidgeCV": stats_cv['residuals'],
                "RidgeVIF": stats_vif['residuals']
            }
            )
        residuals.index = y.index
        model_outputs['residuals'] = residuals
        popt_table = pd.DataFrame(
            {
                "Feature": list2_name,
                "RidgeCV": list2_cv,
                "RidgeVIF": list2_vif
            }
            )
        popt_table.set_index('Feature',inplace=True)
        model_outputs['popt_table'] = popt_table
        stats = pd.DataFrame(
            {
                "Statistic": list1_name,
                "RidgeCV": list1_cv,
                "RidgeVIF": list1_vif
            }
            )
        stats.set_index('Statistic',inplace=True)
        model_outputs['stats'] = stats
    else:
        y_pred = pd.DataFrame(
            {
                "RidgeCV": stats_cv['y_pred'],
            }
            )
        y_pred.index = y.index
        model_outputs['y_pred'] = y_pred
        residuals = pd.DataFrame(
            {
                "RidgeCV": stats_cv['residuals'],
            }
            )
        residuals.index = y.index
        model_outputs['residuals'] = residuals
        popt_table = pd.DataFrame(
            {
                "Feature": list2_name,
                "RidgeCV": list2_cv,
            }
            )
        popt_table.set_index('Feature',inplace=True)
        model_outputs['popt_table'] = popt_table
        stats = pd.DataFrame(
            {
                "Statistic": list1_name,
                "RidgeCV": list1_cv,
            }
            )
        stats.set_index('Statistic',inplace=True)
        model_outputs['stats'] = stats

    if not has_dummies:
        # Calculate VIF of X at Ridge regression alpha values
        alphas = [model_outputs['stats']['RidgeCV']['alpha'],
                      model_outputs['stats']['RidgeVIF']['alpha'],] 
        df = model_outputs['alpha_vs_penalty'].copy()
        df.reset_index(drop=False, inplace=True)
        pf_cv = np.array(df[df['alpha'] == alphas[0]]['pen_factors'])
        pf_vif = np.array(df[df['alpha'] == alphas[1]]['pen_factors'])
        pen_factors = [pf_cv, pf_vif]
        vif_calc = vif_ridge(X,pen_factors)
    
    # Calculate the covariance matrix of the features
    popt_all = {}
    
    if not has_dummies:
        pcov_all = {}
        vif_all = {}
    
    col = X.columns

    # RidgeCV
    model_ = model_objects['RidgeCV']
    popt = stats_cv['popt'].copy()
    X_ = X.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        popt.set_index('Feature',inplace=True)
        popt_all['RidgeCV'] = popt
        pcov_all['RidgeCV'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = np.insert(vif_calc[0], 0, np.nan) 
        #        
        vif.set_index('Feature',inplace=True)
        vif_all["RidgeCV"] = vif

    if not has_dummies:
        # RidgeVIF
        model_ = model_objects['RidgeVIF']
        popt = stats_vif['popt'].copy()
        X_ = X.copy()
        for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
            if model_.coef_[i]==0:
                X_ = X_.drop(col[i], axis = 1)
                popt = popt.drop(index=i+1)
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        popt.set_index('Feature',inplace=True)
        popt_all['RidgeVIF'] = popt
        pcov_all['RidgeVIF'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns

        # vif["VIF"] = np.insert(vif_calc[3], 0, np.nan) 
        vif["VIF"] = np.insert(vif_calc[1], 0, np.nan) 

        #        
        vif.set_index('Feature',inplace=True)
        vif_all["RidgeVIF"] = vif
    
    # save vif and pcov in model_outputs
    model_outputs['popt'] = popt_all

    if not has_dummies:
        model_outputs['pcov'] = pcov_all
        model_outputs['vif'] = vif_all
        # Make big VIF table of all models in one table
        # get row indicdes of non-zero coef values in each model col
        idx = popt_table.apply(lambda col: col[col != 0].index.tolist())
        # initialize vif_table same as popt_table but with nan values
        vif_table = pd.DataFrame(np.nan, index=popt_table.index, columns=popt_table.columns)
        # Put in the VIF values in each model column
        # RidgeCV
        vif = model_outputs['vif']['RidgeCV']['VIF'].values
        vif_table.loc[idx['RidgeCV'], "RidgeCV"] = vif
        # RidgeVIF
        vif = model_outputs['vif']['RidgeVIF']['VIF'].values
        vif_table.loc[idx['RidgeVIF'], "RidgeVIF"] = vif
        # drop const row from VIF table and save in outputs
        vif_table = vif_table.drop(index=['const'])
        model_outputs['vif_table'] = vif_table

    # Print model_outputs
    if data['verbose'] == 'on' or data['verbose'] == 1:
        print("Ridge regression statistics of best models in model_outputs['stats']:")
        print('')
        print(model_outputs['stats'].to_markdown(index=True))
        print('')
        if data['verbose'] != 1:
            print("Coefficients of best models in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
            if not has_dummies:
                print("Variance Inflation Factors model_outputs['vif']:")
                print("Note: VIF>5 indicates excessive collinearity")
                print('')
                print(model_outputs['vif_table'].to_markdown(index=True))
                print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return model_objects, model_outputs

def elastic(X, y, **kwargs):

    """
    Python function for ElasticNetCV linear regression 

    by
    Greg Pelletier
    gjpelletier@gmail.com
    29-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        nfolds= number of folds to use for cross-validation (CV)
            (default nfolds=20)
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        alpha_min= minimum value of range of alphas to evaluate (default=1e-3)
        alpha_max= maximum value of range of alphas to evaluate (default=1e3)
        n_alpha= number of log-spaced alphas to evaluate (default=100)
        l1_ratio= Float between 0 and 1 passed to ElasticNet 
            (scaling between l1 and l2 penalties). 
            For l1_ratio = 0 the penalty is an L2 penalty. 
            For l1_ratio = 1 it is an L1 penalty. 
            For 0 < l1_ratio < 1, the penalty is a combination of 
            L1 and L2 This parameter can be a list, in which case 
            the different values are tested by cross-validation 
            and the one giving the best prediction score is used. 
            default is l1_ratio= np.linspace(0.01,1,100)        
        verbose= 'on' (default), 'off', or 1=show stats and residuals plot
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    It is generally recommended to use a largest possible number of folds 
    to ensure more accurate model selection. 
    The only disadvantage of a large number of folds is the increase 
    computational time. The elastic function allows you to specify 
    the number of folds using the nfolds argument. 
    Using a larger number can lead to better performance. 
    For optimal results, consider experimenting 
    with different fold sizes to find the best balance 
    between performance and speed.

    RETURNS
        model_objects, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'alpha_vs_coef': model coefficients for each X variable
                    as a function of alpha using ElasticNet
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'popt': Constant (intercept) and coefficients for the 
                    best fit models from each of the four methods
                - 'popt_table': Constant (intercept) and coefficients
                    of best fit of all four methods in one table
                - 'pcov': Covariance matrix of features 
                - 'vif': Variance Inlfation Factors of features
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = elastic(X, y)

    """

    from PyMLR import stats_given_model, detect_dummy_variables
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    import time
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.linear_model import MultiTaskElasticNet, MultiTaskElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import PredictionErrorDisplay
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    
   
    # Define default values of input data arguments
    defaults = {
        'nfolds': 20,
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'alpha_min': 1.0e-3,
        'alpha_max': 1.0e3,
        'n_alpha': 100,
        # 'l1_ratio': np.linspace(0.01,1,100),      # e.g. 0.5 or list [.1, .5, .7, .9, .95, .99, 1]
        'l1_ratio': [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1],     
        'verbose': 'on'
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    ctrl = data['alpha_min'] > 0 
    if not ctrl:
        print('Check input of alpha_min, it must be greater than zero!','\n')
        sys.exit()
    ctrl = data['alpha_max'] > data['alpha_min'] 
    if not ctrl:
        print('Check input of alpha_max, it must be greater than alpha_min!','\n')
        sys.exit()
    ctrl = data['n_alpha'] > 1 
    if not ctrl:
        print('Check inputs of n_alpha, it must be greater than 1!','\n')
        sys.exit()
    ctrl = min(data['l1_ratio'])>=0 and max(data['l1_ratio'])<=1
    if not ctrl:
        print('Check inputs of l1_ratio, it must be between 0-1!','\n')
        sys.exit()

    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting Elastic Net regression model, please wait ...')
    if data['verbose'] == 'on' or data['verbose'] == 1:
        print('')

    # ElasticNetCV k-fold cross validation
    model_cv = ElasticNetCV(l1_ratio=data['l1_ratio'], cv=data['nfolds'], 
        random_state=0).fit(X, y)
    model_objects['ElasticNetCV'] = model_cv
    # model_objects = model_cv
    alpha_ = model_cv.alpha_
    l1_ratio_ = model_cv.l1_ratio_
    l1_ratio_idx = np.where(data['l1_ratio'] == model_cv.l1_ratio_)[0]

    # Calculate the role of alpha vs coefficient values at best fit l1_ratio
    alpha_min = np.log10(data['alpha_min'])
    alpha_max = np.log10(data['alpha_max'])    
    n_alpha = data['n_alpha']    
    alphas = 10**np.linspace(alpha_min,alpha_max,n_alpha)
    # alphas = 10**np.linspace(-3,3,100)
    elastic = ElasticNet(l1_ratio=model_cv.l1_ratio_)
    coefs = []
    for a in alphas:
        elastic.set_params(alpha=a)
        elastic.fit(X, y)
        coefs.append(elastic.coef_)
    alpha_vs_coef = pd.DataFrame({
        'alpha': alphas,
        'coef': coefs
        }).set_index("alpha")
    model_outputs['alpha_vs_coef'] = alpha_vs_coef
    
    # Plot the results of coef as function of alpha
    if data['verbose'] == 'on' and data['verbose'] != 1:
        ax = plt.gca()
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.axis('tight')
        plt.xlabel(r"$\alpha$")
        if X.shape[1] < 20:
            plt.legend(X.columns)
        plt.ylabel('Coefficients')
        plt.title(r'ElasticNet regression coefficients as a function of $\alpha$');
        plt.savefig("ElasticNet_alpha_vs_coef.png", dpi=300)

    # ElasticNetCV Plot the MSE vs alpha for each fold
    if data['verbose'] == 'on' and data['verbose'] != 1:
        model = model_cv
        plt.figure()
        plt.semilogx(np.squeeze(model.alphas_[l1_ratio_idx]), 
            np.squeeze(model.mse_path_[l1_ratio_idx]), linestyle=":")
        plt.plot(
            np.squeeze(model.alphas_[l1_ratio_idx]),
            np.squeeze(model.mse_path_.mean(axis=-1)[l1_ratio_idx]),
            color="black",
            label="Average across the folds",
            linewidth=2,
        )
        plt.axvline(model.alpha_, linestyle="--", color="black", 
                    label="CV selected alpha={:.3e}".format(model_cv.alpha_))        
        # ymin, ymax = 2300, 3800
        # plt.ylim(ymin, ymax)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Mean Square Error")
        plt.legend()
        _ = plt.title(
            "ElasticNetCV - Mean Square Error on each fold"
        )
        plt.savefig("ElasticNetCV_alpha_vs_MSE.png", dpi=300)

    # Calculate regression stats
    stats_cv = stats_given_model(X, y, model_cv)

    # residual plot for training error
    if data['verbose'] == 'on' or data['verbose'] == 1:
        '''
        y_pred_cv = stats_cv['y_pred']
        res_cv = stats_cv['residuals']
        rmse_cv = stats_cv['RMSE']
        plt.figure()
        plt.scatter(y_pred_cv, (res_cv), s=40, label=('ElasticNetCV (RMSE={:.2f})'.format(rmse_cv)))
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residual plot for training error")
        plt.legend();
        plt.xlabel('y_pred')
        plt.ylabel('residual')
        plt.savefig("ElasticNetCV_residuals.png", dpi=300)
        '''
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=stats_cv['y_pred'],
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=stats_cv['y_pred'],
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={stats_cv['RMSE']:.3f})")
        plt.tight_layout()
        # plt.show()
        plt.savefig("ElasticNetCV_predictions.png", dpi=300)

    # Make the model_outputs dataframes
    list1_name = ['alpha','r-squared','adjusted r-squared',
                        'n_samples','df residuals','df model',
                        'F-statistic','Prob (F-statistic)','RMSE',
                        'Log-Likelihood','AIC','BIC','L1-ratio']
    list2_name = list(stats_cv['popt']['Feature'])
    list3_name = list1_name + list2_name

    list1_cv = [model_cv.alpha_, stats_cv["rsquared"], stats_cv["adj_rsquared"],
                       stats_cv["n_samples"], stats_cv["df"], stats_cv["dfn"], 
                       stats_cv["Fstat"], stats_cv["pvalue"], stats_cv["RMSE"],  
                       stats_cv["log_likelihood"],stats_cv["aic"],stats_cv["bic"],l1_ratio_]
    list2_cv = list(stats_cv['popt']['param'])
    list3_cv = list1_cv + list2_cv

    y_pred = pd.DataFrame(
        {
            "ElasticNetCV": stats_cv['y_pred']
        }
        )
    y_pred.index = y.index
    model_outputs['y_pred'] = y_pred

    residuals = pd.DataFrame(
        {
            "ElasticNetCV": stats_cv['residuals']
        }
        )
    residuals.index = y.index
    model_outputs['residuals'] = residuals

    # Table of all popt incl coef=0
    popt_table = pd.DataFrame(
        {
            "Feature": list2_name,
            "ElasticNetCV": list2_cv
        }
        )
    popt_table.set_index('Feature',inplace=True)
    model_outputs['popt_table'] = popt_table
    
    # Calculate the covariance matrix of the features
    # popt, pcov, and vif of only the selected features (excl coef=0)
    popt_all = {}
    pcov_all = {}
    vif_all = {}
    # vif = pd.DataFrame()
    col = X.columns
    # ElasticNetCV
    model_ = model_objects['ElasticNetCV']
    popt = stats_cv['popt'].copy()
    X_ = X.copy()
    for i in range(len(model_.coef_)):   # set X col to zero if coef = 0
        if model_.coef_[i]==0:
            X_ = X_.drop(col[i], axis = 1)
            popt = popt.drop(index=i+1)
    if not X_has_dummies:
        X__ = sm.add_constant(X_)    # Add a constant for the intercept
        pcov = pd.DataFrame(np.cov(X__, rowvar=False), index=X__.columns)
        pcov.columns = X__.columns
        pcov_all['ElasticNetCV'] = pcov
        vif = pd.DataFrame()
        vif['Feature'] = X__.columns
        vif["VIF"] = [variance_inflation_factor(X__.values, i)
                            for i in range(len(X__.columns))]
        vif.set_index('Feature',inplace=True)
        vif_all["ElasticNetCV"] = vif
    popt.set_index('Feature',inplace=True)
    popt_all['ElasticNetCV'] = popt
    # save pcov, vif, popt
    if not X_has_dummies:
        # save vif and pcov
        model_outputs['vif'] = vif_all
        model_outputs['pcov'] = pcov_all
    model_outputs['popt'] = popt_all

    if not X_has_dummies:
        # Make big VIF table of all models in one table
        # get row indicdes of non-zero coef values in each model col
        idx = popt_table.apply(lambda col: col[col != 0].index.tolist())
        # initialize vif_table same as popt_table but with nan values
        vif_table = pd.DataFrame(np.nan, index=popt_table.index, columns=popt_table.columns)
        # Put in the VIF values in each model column
        # ElasticNetCV
        vif = model_outputs['vif']['ElasticNetCV']['VIF'].values
        vif_table.loc[idx['ElasticNetCV'], "ElasticNetCV"] = vif
        model_outputs['vif_table'] = vif_table

    stats = pd.DataFrame(
        {
            "Statistic": list1_name,
            "ElasticNetCV": list1_cv
        }
        )
    stats.set_index('Statistic',inplace=True)
    model_outputs['stats'] = stats
    
    # Print model_outputs
    if data['verbose'] == 'on' or data['verbose'] == 1:
        print("ElasticNetCV regression statistics of best model in model_outputs['stats']:")
        print('')
        print(model_outputs['stats'].to_markdown(index=True))
        print('')
        if data['verbose'] != 1:
            print("Coefficients of best model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
            if not X_has_dummies:
                print("Variance Inflation Factors model_outputs['vif']:")
                print("Note: VIF>5 indicates excessive collinearity")
                print('')
                print(model_outputs['vif_table'].to_markdown(index=True))
                print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return model_objects, model_outputs

def stacking(X, y, **kwargs):

    """
    Python function for StackingRegressor 

    by
    Greg Pelletier
    gjpelletier@gmail.com
    30-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        random_state= (default random_state=42)        - initial random seed

        meta= 'linear' (default), 'lasso', or 'elastic' 
            A meta-model regressor which will be used to combine the base estimators. 
            The default meta regressor is RidgeCV
            'linear' = LinearRegression 
            'lasso' = LassoCV
            'ridge' = RidgeCV (default)
            'elastic' = ElasticNetCV

        The following keyword arguments activate/deactivate selected base_regressors,
        and any combination of the following base_regressors may be 'on' or 'off':

            lasso= 'on' (default) or 'off'        - LassoCV
            ridge= 'on' (default) or 'off'        - RidgeCV
            elastic= 'on' (default) or 'off'      - ElasticNetCV
            sgd= 'on' (default) or 'off'          - SGDRegressor
            knr= 'on' (default) or 'off'          - KNeighborsRegressor
            gbr= 'on' (default) or 'off'          - GradientBoostingRegressor
            tree= 'on' (default) or 'off'         - DecisionTreeRegressor
            forest= 'on' (default) or 'off'       - RandomForestRegressor
            svr= 'on' or 'off' (default)          - SVR(kernel='rbf')
            mlp= 'on' or 'off' (default)          - MLPRegressor

        verbose= 'on' (default) or 'off'
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        model_objects, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'strength': Intercept and coefficients of the 
                    strength of each base_regressor
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = stacking(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.linear_model import SGDRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import PredictionErrorDisplay
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # import xgboost as xgb

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'random_state': 42,
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'meta': 'ridge',
        'lasso': 'on',
        'ridge': 'on',
        'elastic': 'on',
        'sgd': 'on',       
        'knr': 'on',
        'svr': 'off',       # off
        'mlp': 'off',       # off
        'gbr': 'on',
        'tree': 'on',
        'forest': 'on',
        'alpha_min': 1e-5,
        'alpha_max': 1e2,
        'n_alpha': 100,
        'l1_ratio': np.linspace(0.01,1,100),    
        'verbose': 'on'
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    ctrl = data['alpha_min'] > 0 
    if not ctrl:
        print('Check input of alpha_min, it must be greater than zero!','\n')
        sys.exit()
    ctrl = data['alpha_max'] > data['alpha_min'] 
    if not ctrl:
        print('Check input of alpha_max, it must be greater than alpha_min!','\n')
        sys.exit()
    ctrl = data['n_alpha'] > 1 
    if not ctrl:
        print('Check inputs of n_alpha, it must be greater than 1!','\n')
        sys.exit()
    ctrl = min(data['l1_ratio'])>=0 and max(data['l1_ratio'])<=1
    if not ctrl:
        print('Check inputs of l1_ratio, it must be between 0-1!','\n')
        sys.exit()

    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting StackingRegressor models, please wait ...')
    if data['verbose'] == 'on':
        print('')
    
    # Calculate alphas for RidgeCV
    alpha_min = np.log10(data['alpha_min'])
    alpha_max = np.log10(data['alpha_max'])    
    n_alpha = data['n_alpha']    
    alphas = 10**np.linspace(alpha_min,alpha_max,n_alpha)
    
    # Define base regressors
    base_regressors = []
    if data['lasso']=='on':
        base_regressors.append(('LassoCV', 
            LassoCV(random_state=data['random_state'])))
    if data['ridge']=='on':
        base_regressors.append(('RidgeCV', 
            RidgeCV(alphas=alphas)))
    if data['elastic']=='on':
        base_regressors.append(('ElasticNetCV', 
            ElasticNetCV(l1_ratio=data['l1_ratio'],
                random_state=data['random_state'])))
    if data['sgd']=='on':
        base_regressors.append(('SGDRegressor', 
            SGDRegressor(random_state=data['random_state'])))
    if data['knr']=='on':
        base_regressors.append(('KNeighborsRegressor', 
            KNeighborsRegressor()))
    if data['svr']=='on':
        base_regressors.append(('SVR', 
            SVR(kernel='rbf')))
    if data['mlp']=='on':
        base_regressors.append(('MLPRegressor', 
            MLPRegressor(random_state=data['random_state'])))
    if data['gbr']=='on':
        base_regressors.append(('GradientBoostingRegressor', 
            GradientBoostingRegressor(random_state=data['random_state'])))
    # if data['xgb']=='on':
    #     base_regressors.append(('XGBoost', 
    #         xgb.XGBRegressor(
    #             objective='reg:squarederror', n_estimators=100, 
    #             random_state=data['random_state'])))
    if data['tree']=='on':
        base_regressors.append(('DecisionTreeRegressor', 
            DecisionTreeRegressor(random_state=data['random_state'])))
    if data['forest']=='on':
        base_regressors.append(('RandomForestRegressor', 
            RandomForestRegressor(
                n_estimators=50, random_state=data['random_state'])))
    if not base_regressors:
        print('Check input arguments, all base regressors are turned off!','\n')
        sys.exit()

    # Define the meta-regressor (final estimator)
    # meta_regressor = LinearRegression()
    if data['meta']=='lasso':
        meta_regressor = LassoCV(random_state=data['random_state'])
    elif data['meta']=='ridge':
        meta_regressor = RidgeCV(alphas=alphas)
    elif data['meta']=='elastic':
        meta_regressor = ElasticNetCV(
            l1_ratio=data['l1_ratio'],random_state=data['random_state'])
    else:
        meta_regressor = LinearRegression()
        
    # Create the Stacking Regressor
    stacking_regressor = StackingRegressor(
        estimators=base_regressors, final_estimator=meta_regressor)
    
    # Train the Stacking Regressor
    stacking_regressor.fit(X, y)
    
    # Extract parameters of the stacking_regressor
    intercept = stacking_regressor.final_estimator_.intercept_
    coefficients = stacking_regressor.final_estimator_.coef_
    n_param = 1 + coefficients.size      # number of parameters including intercept
    popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
    for i in range(n_param):
        if i == 0:
            popt[0][i] = 'Intercept'
            popt[1][i] = intercept
        else:
            popt[0][i] = base_regressors[i-1][0]
            popt[1][i] = coefficients[i-1]
    popt = pd.DataFrame(popt).T
    popt.columns = ['name', 'param']

    # Calculate regression statistics
    y_pred = stacking_regressor.predict(X)
    stats = stats_given_y_pred(X,y,y_pred)
    
    # model objects and outputs returned by stacking
    # model_outputs['scaler'] = scaler                     # scaler used to standardize X
    # model_outputs['standardize'] = data['standardize']   # True: X_scaled was used to fit, False: X was used
    model_outputs['y_pred'] = stats['y_pred']
    model_outputs['residuals'] = stats['residuals']
    model_objects = stacking_regressor
    
    # residual plot for training error
    if data['verbose'] == 'on':
        '''
        y_pred = stats['y_pred']
        res = stats['residuals']
        rmse = stats['RMSE']
        plt.figure()
        plt.scatter(y_pred, (res), s=40, label=('StackingRegressor (RMSE={:.2f})'.format(rmse)))
        rmse_cv = np.sqrt(np.mean((res)**2))
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residual plot for training error")
        plt.legend();
        plt.xlabel('y_pred')
        plt.ylabel('residual')
        plt.savefig("StackingRegressor_residuals.png", dpi=300)
        '''
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=stats['y_pred'],
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=stats['y_pred'],
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={stats['RMSE']:.3f})")
        plt.tight_layout()
        # plt.show()
        plt.savefig("StackingRegressor_predictions.png", dpi=300)

    
    # Make the model_outputs dataframes
    '''
    list1_name = ['r-squared','adjusted r-squared',
                        'n_samples','df residuals','df model',
                        'F-statistic','Prob (F-statistic)','RMSE',
                        'Log-Likelihood','AIC','BIC']    
    list1_val = [stats["rsquared"], stats["adj_rsquared"],
                       stats["n_samples"], stats["df"], stats["dfn"], 
                       stats["Fstat"], stats["pvalue"], stats["RMSE"],  
                       stats["log_likelihood"],stats["aic"],stats["bic"]]
    '''
    list1_name = ['r-squared', 'RMSE', 'n_samples']        
    list1_val = [stats["rsquared"], stats["RMSE"], stats["n_samples"]]
    list2_name = list(popt['name'])
    list2_val = list(popt['param'])
    
    stats = pd.DataFrame(
        {
            "Statistic": list1_name,
            "StackingRegressor": list1_val
        }
        )
    stats.set_index('Statistic',inplace=True)
    model_outputs['stats'] = stats
    print("StackingRegressor statistics of fitted ensemble model in model_outputs['stats']:")
    print('')
    print(model_outputs['stats'].to_markdown(index=True))
    print('')
    
    # Table of all popt incl coef=0
    meta_params = pd.DataFrame(
        {
            "Coefficient": list2_name,
            "StackingRegressor": list2_val
        }
        )
    meta_params.set_index('Coefficient',inplace=True)
    model_outputs['meta_params'] = meta_params
    print("Meta-model coefficients of base_regressors in model_outputs['meta_params']:")
    print('')
    print('- positive intercept suggests base models under-predict target')
    print('- negative intercept suggests base models over-predict target')
    print('- positive coefficients have high importance')
    print('- coefficients near zero have low importance')
    print('- negative coefficients have counteracting importance')
    print('')
    print(model_outputs['meta_params'].to_markdown(index=True))
    print('')
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return model_objects, model_outputs

def svr(X, y, **kwargs):

    """
    Python function for SVR (regression) or SVC (classification) 

    by
    Greg Pelletier
    gjpelletier@gmail.com
    13-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        classify= False,            # True for SVC
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on' (default) or 'off' 
        degree= 3          # int, default 3, deg of polynomial, used only if kernel='poly'
        gamma= 'scale'     # 'scale' (default), 'auto', or float (if float, must be non-negative)
        coef0= 0.0         # term in kernel function, only significant in ‘poly’ and ‘sigmoid’
        tol= 0.001         # Tolerance for stopping criterion
        C= 1.0             # Regularization parameter. The strength of the regularization 
                           # is inversely proportional to C. Must be strictly positive
                           # The penalty is a squared L2. 
        epsilon= 0.1       # float, default 0.1, for the epsilon-SVR model, must be non-negative
        shrinking= True    # Whether to use the shrinking heuristic
        cache_size= 200    # Specify the size of the kernel cache (in MB)
        max_iter= -1       # Hard limit on iterations within solver, or -1 for no limit.
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        model_objects, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = svr(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.svm import SVR, SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import PredictionErrorDisplay
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # import xgboost as xgb

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'classify': False,            # True for SVC
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        # 'kernel': 'rbf',
        'degree': 3,
        'gamma': 'scale',
        'coef0': 0.0,
        'tol': 0.001,
        'C': 1.0,
        'epsilon': 0.1,
        'shrinking': True,
        'cache_size': 200,
        'max_iter': -1
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")
    
    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    ctrl = data['gamma']=='scale' or data['gamma']=='auto' or data['gamma']>0   
    if not ctrl:
        print('Check inputs of gamma, it must be scale, auto, or float>0!','\n')
        sys.exit()
    ctrl = data['epsilon']>0   
    if not ctrl:
        print('Check inputs of epsilon, it float>0!','\n')
        sys.exit()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    if data['classify']:
        print('Fitting SVC model, please wait ...')
        data['kernel'] = 'rbf'                                  
        data['probability'] = True                                  
        fitted_model = SVC(
            gamma= data['gamma'],
            kernel= data['kernel'],                                  
            probability= data['probability'],                                  
            degree= data['degree'],    
            coef0= data['coef0'],    
            tol= data['tol'],         
            C= data['C'],                                                                   
            shrinking= data['shrinking'],    
            cache_size= data['cache_size'],    
            max_iter= data['max_iter']      
            ).fit(X,y)
    else:
        print('Fitting SVR model, please wait ...')
        data['kernel'] = 'rbf'                                  
        fitted_model = SVR(
            gamma= data['gamma'],
            epsilon= data['epsilon'],
            kernel= data['kernel'],                                  
            degree= data['degree'],    
            coef0= data['coef0'],    
            tol= data['tol'],         
            C= data['C'],                                                                   
            shrinking= data['shrinking'],    
            cache_size= data['cache_size'],    
            max_iter= data['max_iter']      
            ).fit(X,y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            # selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X, y)
            hfig.savefig("SVC_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X, y)
            hfig.savefig("SVC_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X, y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['SVC']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)    
        if data['verbose'] == 'on':
            print('')
            print("SVC goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:        
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X.columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X.columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table

        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X, y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['SVR']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)
    
        if data['verbose'] == 'on':
            print('')
            print("SVR goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("SVR_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def svr_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by optuna 
    to find the optimum hyper-parameters for SVR
    '''
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from sklearn.svm import SVR, SVC

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')

    # Detect if the computer has an nvidia gpu, and if so use the gpu
    use_gpu = detect_gpu()
    if use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)
    
    params = {
        "C": trial.suggest_float("C",
            *kwargs['C'], log=True),
        # "epsilon": trial.suggest_float("epsilon",
        #     *kwargs['epsilon'], log=True),
    }

    if not kwargs['classify']:
        params['epsilon'] = trial.suggest_float("epsilon",
            *kwargs['epsilon'], log=True),
    
    if kwargs["gamma"] == "scale" or kwargs["gamma"] == "auto":
        params["gamma"] = kwargs["gamma"]
    else:
        params["gamma"] = trial.suggest_float("gamma", 0.0001, 1.0, log=True)

    extra_params = {
        'kernel': kwargs['kernel'],                                  
        'degree': kwargs['degree'],    
        'coef0': kwargs['coef0'],    
        'tol': kwargs['tol'],         
        'shrinking': kwargs['shrinking'],    
        'cache_size': kwargs['cache_size'],    
        'max_iter': kwargs['max_iter']      
    }

    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        if kwargs['classify']:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", SVC(**params, **extra_params))
            ])
        else:        
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", SVR(**params, **extra_params))
            ])

    else:

        if kwargs['classify']:
            pipeline = Pipeline([
                ("regressor", SVC(**params, **extra_params))
            ])
        else:        
            pipeline = Pipeline([
                ("regressor", SVR(**params, **extra_params))
            ])

        num_features = None

    # Cross-validated scoring
    if kwargs['classify']:
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="f1_weighted"        
            scoring=kwargs["scoring"]
        )
    else:
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )
    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean

def svr_auto(X, y, **kwargs):

    """
    Autocalibration of SVR or SVC hyper-parameters
    Beta version

    by
    Greg Pelletier
    gjpelletier@gmail.com
    13-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        verbose= 'on' (default) or 'off'
        classify= False (default) or True to use XGBClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        random_state= 42,                 # Random seed for reproducibility.
        n_trials= 50,                     # number of optuna trials
        n_splits= 5,                      # number of splits for KFold CV
        gpu= True,                        # Autodetect to use gpu if present
        verbose= 'on' (default) or 'off'

        pruning= False,                   # prune poor optuna trials
        feature_selection= True,          # optuna feature selection

        # [min, max] ranges of params for model to be optimized by optuna:
        C= [0.1, 1000],           # C Regularization parameter. 
                                  # The strength of the regularization is 
                                  # inversely proportional to C. 
                                  # Must be strictly positive. The penalty is a squared l2.
        epsilon= [1e-4, 1.0],     # Epsilon in the epsilon-SVR model. Must be non-negative
        # gamma= [0.0001, 1.0],   # range of gamma values if not using 'scale' or 'auto'
        gamma= 'scale',           # {'scale', 'auto'}, default='scale'

        # extra_params that are optional user-specified
        degree= 3,                # Degree of the polynomial kernel function (‘poly’). 
                                  # Must be non-negative. Ignored by all other kernels.
        coef0= 0.0,               # Independent term in kernel function. 
                                  # It is only significant in ‘poly’ and ‘sigmoid’
        tol= 0.001,               # Tolerance for stopping criterion
        shrinking= True,          # Whether to use the shrinking heuristic.
        cache_size= 200,          # Specify the size of the kernel cache (in MB)
        max_iter= -1              # Hard limit on iterations within solver, 
                                  # or -1 for no limit
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'optuna_study': optimzed optuna study object
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = svr_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    # import xgboost as xgb
    # from xgboost import XGBRegressor
    import optuna
    from sklearn.svm import SVR, SVC

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'random_state': 42,                 # Random seed for reproducibility.
        'n_trials': 5,                     # number of optuna trials
        'n_splits': 5,          # number of splits for KFold CV
        'gpu': True,                        # Autodetect to use gpu if present
        'classify': False,            # Use SVC if True
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',

        'pruning': False,                   # prune poor optuna trials
        'feature_selection': True,          # optuna feature selection
        'scoring': None,                     # cross_val_score scoring name
        'show_trial_progress': True,         # print trial numbers during execution
        
        # params for model that are optimized by optuna
        'C': [0.1, 1000],           # range of C Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2.
        'epsilon': [1e-4, 1.0],     # range of epsilon Epsilon in the epsilon-SVR model. Must be non-negative
        # 'gamma': [0.0001, 1.0],   # range of gamma values if not using 'scale' or 'auto'
        'gamma': 'scale',           # {'scale', 'auto'}, default='scale'

        # extra_params for model that are optional user-specified
        # 'kernel': 'rbf',            # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        'degree': 3,                # Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.
        'coef0': 0.0,               # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’
        'tol': 0.001,               # Tolerance for stopping criterion
        'shrinking': True,          # Whether to use the shrinking heuristic.
        'cache_size': 200,          # Specify the size of the kernel cache (in MB)
        'max_iter': -1              # Hard limit on iterations within solver, or -1 for no limit
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Dictionary to pass to optuna

    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    # extra params in addition to those being optimized by optuna
    extra_params = {
        # 'kernel': data['kernel'],                                  
        'degree': data['degree'],    
        'coef0': data['coef0'],    
        'tol': data['tol'],         
        'shrinking': data['shrinking'],    
        'cache_size': data['cache_size'],    
        'max_iter': data['max_iter']      
    }

    if data['classify']:
        # these settings needed for SVC model.predict_proba
        data['kernel'] = 'rbf'                                  
        data['probability'] = True                                  
        extra_params['kernel'] = data['kernel']                                  
        extra_params['probability'] = data['probability']                                 
    else:
        data['kernel'] = 'rbf'                                  
        extra_params['kernel'] = data['kernel']                                  

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import svr_objective
    study.optimize(lambda trial: svr_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial
        
    best_params = study.best_params
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']

    if data['classify']:
        print('Fitting SVC model with best parameters, please wait ...')        
        fitted_model = SVC(
            **best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
    else:
        print('Fitting SVR model with best parameters, please wait ...')        
        fitted_model = SVR(
            **best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
       
    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("SVC_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("SVC_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['SVC']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("SVC goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:            
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table        

        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        # metrics = stats_given_model(
        #     X[model_outputs['selected_features']], y, fitted_model)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['SVR']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("SVR goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("SVR_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def sgd(X, y, **kwargs):

    """
    Python function for SGDRegressor regression 
    Beta version

    by
    Greg Pelletier
    gjpelletier@gmail.com
    30-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        random_state= (default random_state=42)        - initial random seed
        verbose= 'on' (default) or 'off'
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        model_objects, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = sgd(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.linear_model import SGDRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import PredictionErrorDisplay
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # import xgboost as xgb

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'random_state': 42,
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on'
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting SGDRegressor model, please wait ...')
    if data['verbose'] == 'on':
        print('')
    
    model = SGDRegressor(
        random_state=data['random_state']).fit(X,y)

    # check to see of the model has intercept and coefficients
    if (hasattr(model, 'intercept_') and hasattr(model, 'coef_') 
            and model.coef_.size==len(X.columns)):
        intercept = model.intercept_
        coefficients = model.coef_
        # dataframe of model parameters, intercept and coefficients, including zero coefs
        n_param = 1 + model.coef_.size               # number of parameters including intercept
        popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
        for i in range(n_param):
            if i == 0:
                popt[0][i] = 'Intercept'
                popt[1][i] = model.intercept_
            else:
                popt[0][i] = X.columns[i-1]
                popt[1][i] = model.coef_[i-1]
        popt = pd.DataFrame(popt).T
        popt.columns = ['Feature', 'Parameter']
        # Table of intercept and coef
        popt_table = pd.DataFrame({
                "Feature": popt['Feature'],
                "Parameter": popt['Parameter']
            })
        popt_table.set_index('Feature',inplace=True)
        model_outputs['popt_table'] = popt_table
    
    # Calculate regression statistics
    y_pred = model.predict(X)
    stats = stats_given_y_pred(X,y,y_pred)
    
    # model objects and outputs returned by stacking
    # model_outputs['scaler'] = scaler                     # scaler used to standardize X
    # model_outputs['standardize'] = data['standardize']   # True: X_scaled was used to fit, False: X was used
    model_outputs['y_pred'] = stats['y_pred']
    model_outputs['residuals'] = stats['residuals']
    model_objects = model
    
    # residual plot for training error
    if data['verbose'] == 'on':
        '''
        y_pred = stats['y_pred']
        res = stats['residuals']
        rmse = stats['RMSE']
        plt.figure()
        plt.scatter(y_pred, (res), s=40, label=('SGDRegressor (RMSE={:.2f})'.format(rmse)))
        rmse_cv = np.sqrt(np.mean((res)**2))
        plt.hlines(y=0, xmin=min(y), xmax=max(y), color='k')
        plt.title("Residual plot for training error")
        plt.legend();
        plt.xlabel('y_pred')
        plt.ylabel('residual')
        plt.savefig("SGDRegressor_residuals.png", dpi=300)
        '''
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=stats['y_pred'],
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=stats['y_pred'],
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={stats['RMSE']:.3f})")
        plt.tight_layout()
        # plt.show()
        plt.savefig("SGDRegressor_predictions.png", dpi=300)
    
    # Make the model_outputs dataframes
    '''
    list1_name = ['r-squared','adjusted r-squared',
                        'n_samples','df residuals','df model',
                        'F-statistic','Prob (F-statistic)','RMSE',
                        'Log-Likelihood','AIC','BIC']    
    list1_val = [stats["rsquared"], stats["adj_rsquared"],
                       stats["n_samples"], stats["df"], stats["dfn"], 
                       stats["Fstat"], stats["pvalue"], stats["RMSE"],  
                       stats["log_likelihood"],stats["aic"],stats["bic"]]
    '''
    list1_name = ['r-squared', 'RMSE', 'n_samples']        
    list1_val = [stats["rsquared"], stats["RMSE"], stats["n_samples"]]
    
    stats = pd.DataFrame(
        {
            "Statistic": list1_name,
            "SGDRegressor": list1_val
        }
        )
    stats.set_index('Statistic',inplace=True)
    model_outputs['stats'] = stats
    print("SGDRegressor statistics of fitted model in model_outputs['stats']:")
    print('')
    print(model_outputs['stats'].to_markdown(index=True))
    print('')
    if hasattr(model, 'intercept_') and hasattr(model, 'coef_'):
        print("Parameters of fitted model in model_outputs['popt']:")
        print('')
        print(model_outputs['popt_table'].to_markdown(index=True))
        print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return model_objects, model_outputs

def gbr(X, y, **kwargs):

    """
    GradientBoostingRegressor regression
    Beta version

    by
    Greg Pelletier
    gjpelletier@gmail.com
    30-May-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        random_state= (default random_state=42)        - initial random seed
        loss='squared_error',          # Loss function to optimize. 
                                       # Default is 'squared_error' (mean squared error).
        learning_rate=0.1,             # Shrinks the contribution of each tree. Default is 0.1.
        n_estimators=100,              # Number of boosting stages (trees). Default is 100.
        subsample=1.0,                 # Fraction of samples used for fitting each tree. 
                                       # Default is 1.0 (use all samples).
        criterion='friedman_mse',      # Function to measure the quality of a split. 
                                       # Default is 'friedman_mse'.
        min_samples_split=2,           # Minimum samples required to split an internal node. 
                                       # Default is 2.
        min_samples_leaf=1,            # Minimum samples required to be a leaf node. 
                                       # Default is 1.
        min_weight_fraction_leaf=0.0,  # Minimum weighted fraction of the sum of weights 
                                       # for a leaf node. Default is 0.0.
        max_depth=3,                   # Maximum depth of the individual regression estimators. 
                                       # Default is 3.
        min_impurity_decrease=0.0,     # Minimum impurity decrease required to split a node. 
                                       # Default is 0.0.
        init=None,                     # Initial estimator (e.g., a constant predictor). 
                                       # Default is None.
        random_state=None,             # Seed for reproducibility. Default is None.
        max_features=None,             # Number of features to consider for the best split. 
                                       # Default is None (all features).
        alpha=0.9,                     # Quantile for 'huber' and 'quantile' loss functions. 
                                       # Default is 0.9.
        verbose=0,                     # Verbosity level. Default is 0 (no output).
        max_leaf_nodes=None,           # Maximum number of leaf nodes. Default is None (unlimited).
        warm_start=False,              # Reuse previous solution to add more estimators. 
                                       # Default is False.
        validation_fraction=0.1,       # Fraction of training data for validation. Default is 0.1.
        n_iter_no_change=None,         # Stop training if no improvement after this many iterations. 
                                       # Default is None.
        tol=1e-4,                      # Tolerance for early stopping. Default is 1e-4.
        ccp_alpha=0.0                  # Complexity parameter for Minimal Cost-Complexity Pruning. 
                                       # Default is 0.0.
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        model_objects, model_outputs
            model_objects is the fitted model object 
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = gbr(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    # import xgboost as xgb

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',

        # [min, max] range of params optimized by optuna
        'learning_rate': 0.1,             # Shrinks the contribution of each tree. Default is 0.1.
        'n_estimators': 100,              # Number of boosting stages (trees). Default is 100.
        'max_depth': 3,                   # Maximum depth of the individual regression estimators. Default is 3.
        'min_samples_split': 2,           # Minimum samples required to split an internal node. Default is 2.
        'min_samples_leaf': 1,            # Minimum samples required to be a leaf node. Default is 1.
        'subsample': 1.0,                 # Fraction of samples used for fitting each tree. Default is 1.0 (use all samples).
        'max_features': None,             # {‘sqrt’, ‘log2’}, int or float, default=None. Number of features to consider for the best split. Default is None (all features).

        # extra_params user-specified
        'random_state':  42,              # initial random seed
        'loss': 'squared_error',          # Loss function to optimize. Default is 'squared_error' (mean squared error).
        'criterion': 'friedman_mse',      # Function to measure the quality of a split. Default is 'friedman_mse'.
        'min_weight_fraction_leaf': 0.0,  # Minimum weighted fraction of the sum of weights for a leaf node. Default is 0.0.
        'min_impurity_decrease': 0.0,     # Minimum impurity decrease required to split a node. Default is 0.0.
        'init': None,                     # Initial estimator (e.g., a constant predictor). Default is None.
        'alpha': 0.9,                     # Quantile for 'huber' and 'quantile' loss functions. Default is 0.9.
        'verbosity': 0,                     # Verbosity level. Default is 0 (no output).
        'max_leaf_nodes': None,           # Maximum number of leaf nodes. Default is None (unlimited).
        'warm_start': False,              # Reuse previous solution to add more estimators. Default is False.
        'validation_fraction': 0.1,       # Fraction of training data for validation. Default is 0.1.
        'n_iter_no_change': None,         # Stop training if no improvement after this many iterations. Default is None.
        'tol': 1e-4,                      # Tolerance for early stopping. Default is 1e-4.
        'ccp_alpha': 0.0                  # Complexity parameter for Minimal Cost-Complexity Pruning. Default is 0.0.
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting GradientBoostingRegressor model, please wait ...')
    if data['verbose'] == 'on':
        print('')

    fitted_model = GradientBoostingRegressor(
        random_state=data['random_state'],
        loss= data['loss'],          
        learning_rate= data['learning_rate'],             
        n_estimators= data['n_estimators'],              
        subsample= data['subsample'],                 
        criterion= data['criterion'],      
        min_samples_split= data['min_samples_split'],           
        min_samples_leaf= data['min_samples_leaf'],            
        min_weight_fraction_leaf= data['min_weight_fraction_leaf'],  
        max_depth= data['max_depth'],                   
        min_impurity_decrease= data['min_impurity_decrease'],     
        init= data['init'],                     
        max_features= data['max_features'],             
        alpha= data['alpha'],                     
        max_leaf_nodes= data['max_leaf_nodes'],           
        warm_start= data['warm_start'],              
        validation_fraction= data['validation_fraction'],       
        n_iter_no_change= data['n_iter_no_change'],         
        tol= data['tol'],                      
        ccp_alpha= data['ccp_alpha']                          
        ).fit(X,y)
        
    # check to see of the model has intercept and coefficients
    if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
            and fitted_model.coef_.size==len(X.columns)):
        intercept = fitted_model.intercept_
        coefficients = fitted_model.coef_
        # dataframe of model parameters, intercept and coefficients, including zero coefs
        n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
        popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
        for i in range(n_param):
            if i == 0:
                popt[0][i] = 'Intercept'
                popt[1][i] = fitted_model.intercept_
            else:
                popt[0][i] = X.columns[i-1]
                popt[1][i] = fitted_model.coef_[i-1]
        popt = pd.DataFrame(popt).T
        popt.columns = ['Feature', 'Parameter']
        # Table of intercept and coef
        popt_table = pd.DataFrame({
                "Feature": popt['Feature'],
                "Parameter": popt['Parameter']
            })
        popt_table.set_index('Feature',inplace=True)
        model_outputs['popt_table'] = popt_table
    
    # Goodness of fit statistics
    metrics = fitness_metrics(
        fitted_model, 
        X, y)
    stats = pd.DataFrame([metrics]).T
    stats.index.name = 'Statistic'
    stats.columns = ['GradientBoostingRegressor']
    model_outputs['metrics'] = metrics
    model_outputs['stats'] = stats
    model_outputs['y_pred'] = fitted_model.predict(X)

    if data['verbose'] == 'on':
        print('')
        print("GradientBoostingRegressor goodness of fit to training data in model_outputs['stats']:")
        print('')
        print(model_outputs['stats'].to_markdown(index=True))
        print('')

    if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
        print("Parameters of fitted model in model_outputs['popt']:")
        print('')
        print(model_outputs['popt_table'].to_markdown(index=True))
        print('')

    # residual plot for training error
    if data['verbose'] == 'on':
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
        plt.tight_layout()
        # plt.show()
        plt.savefig("GradientBoostingRegressor_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs
    
def gbr_objective(trial, X, y, **kwargs):
    '''
    Objective function used by optuna to find 
    the optimum hyper-parameters for GradientBoostingRegressor
    '''
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold
    from PyMLR import detect_gpu
    from sklearn.ensemble import GradientBoostingRegressor

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)

    params = {
        "learning_rate": trial.suggest_float("learning_rate",
            *kwargs['learning_rate'], log=True),
        "n_estimators": trial.suggest_int("n_estimators",
            *kwargs['n_estimators']),
        "max_depth": trial.suggest_int("max_depth",
            *kwargs['max_depth']),
        "min_samples_split": trial.suggest_int("min_samples_split",
            *kwargs['min_samples_split']),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf",
            *kwargs['min_samples_leaf']),
        "subsample": trial.suggest_float("subsample",
            *kwargs['subsample']),
    }

    # categorical params
    if len(kwargs["max_features"]) > 1:
        params["max_features"] = trial.suggest_categorical(
            "max_features", kwargs["max_features"])
    elif len(kwargs["max_features"]) == 1:
        params["max_features"] = kwargs["max_features"]
    else:
        params["max_features"] = None
    
    extra_params = {
        'random_state': kwargs['random_state'],         
        'loss': kwargs['loss'],     
        'criterion': kwargs['criterion'],     
        'min_weight_fraction_leaf': kwargs['min_weight_fraction_leaf'],     
        'min_impurity_decrease': kwargs['min_impurity_decrease'],     
        'init': kwargs['init'],    
        'alpha': kwargs['alpha'],    
        'max_leaf_nodes': kwargs['max_leaf_nodes'],    
        'warm_start': kwargs['warm_start'],    
        'validation_fraction': kwargs['validation_fraction'],    
        'n_iter_no_change': kwargs['n_iter_no_change'],    
        'tol': kwargs['tol'],    
        'ccp_alpha': kwargs['ccp_alpha']    
    }

    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        pipeline = Pipeline([
            ("feature_selector", selector),
            ("regressor", GradientBoostingRegressor(**params, **extra_params))
        ])
    else:
        pipeline = Pipeline([
            ("regressor", GradientBoostingRegressor(**params, **extra_params))
        ])
        num_features = None

    # Cross-validated scoring with RepeatedKFold
    cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=kwargs['n_jobs']
    )
    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("score", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean

def gbr_auto(X, y, **kwargs):

    """
    Autocalibration of sklearn GradientBoostingRegressor hyper-parameters
    Beta version

    by
    Greg Pelletier
    gjpelletier@gmail.com
    07-June-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        random_state= 42,    # initial random seed
        n_trials= 50,         # number of optuna trials
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',        # 'on' to display summary stats and residual plots
        n_splits= 5,          # number of splits for KFold CV
        gpu= True,            # Autodetect to use gpu if present
        n_jobs= 1,            # number of CPU cores to use for optuna
                              # n_jobs=1 is reproducible
                              # n_jobs-1 uses all cores but is not reproducible
        pruning= False,             # prune poor optuna trials
        feature_selection= True,    # optuna feature selection

        # [min, max] range of params optimized by optuna
        learning_rate= [1e-4, 1.0],    # Shrinks the contribution of each tree
        n_estimators= [100, 1000],     # Number of boosting stages (trees)
        max_depth= [3, 10],            # Max depth of individual regression estimators
        min_samples_split= [2, 10],    # Min samples required to split an internal node
        min_samples_leaf= [1, 10],     # Min samples required to be a leaf node
        subsample= [0.5, 1.0],         # Fraction of samples for fitting each tree

        # categorical params optimized by optuna
        max_features= [None, "sqrt", "log2"],    # Number of features for the best split

        # extra_params user-specified
        loss= 'squared_error',          # Loss function to optimize
        criterion= 'friedman_mse',      # Function to measure the quality of a split
        min_weight_fraction_leaf= 0.0,  # Min wt fraction of sum of weights for leaf
        min_impurity_decrease= 0.0,     # Min impurity decrease to split a node
        init= None,                     # Initial estimator (constant predictor)
        alpha= 0.9,                     # Quantile for 'huber' & 'quantile' loss
        verbosity= 0,                   # Verbosity level
        max_leaf_nodes= None,           # Maximum number of leaf nodes
        warm_start= False,              # Reuse previous solution to add more estimators
        validation_fraction= 0.1,       # Fraction of training data for validation
        n_iter_no_change= None,         # Stop training if no improvement
        tol= 1e-4,                      # Tolerance for early stopping
        ccp_alpha= 0.0                  # Parameter for Min Cost-Complexity Pruning
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'optuna_study': optimzed optuna study object
                - 'optuna_model': optimzed optuna model object
                - 'best_trial': best trial from the optuna study
                - 'feature_selection' = option to select features (True, False)
                - 'selected_features' = selected features
                - 'best_params': best model hyper-parameters found by optuna
                - 'extra_params': other model options used to fit the model
                - 'metrics': dict of goodness of fit metrics for train data
                - 'stats': dataframe of goodness of fit metrics for train data
                - 'X_processed': pre-processed X with encoding and scaling
                - 'y_pred': best model predicted y

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = gbr_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split, KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'random_state':  42,    # initial random seed
        'n_trials': 50,         # number of optuna trials
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',        # 'on' to display summary stats and residual plots
        'n_splits': 5,          # number of splits for KFold CV
        'gpu': True,            # Autodetect to use gpu if present
        'n_jobs': 1,            # number of CPU cores to use for optuna
                                # n_jobs=1 is reproducible
                                # n_jobs=-1 uses all cores

        'pruning': False,                   # prune poor optuna trials
        'feature_selection': True,          # optuna feature selection
        
        # [min, max] range of params optimized by optuna
        'learning_rate': [1e-4, 1.0],    # Shrinks the contribution of each tree
        'n_estimators': [100, 1000],     # Number of boosting stages (trees)
        'max_depth': [3, 10],            # Maximum depth of the individual regression estimators
        'min_samples_split': [2, 10],    # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 10],     # Minimum samples required to be a leaf node
        'subsample': [0.5, 1.0],         # Fraction of samples used for fitting each tree

        # categorical params optimized by optuna
        'max_features': [None, "sqrt", "log2"],    # Number of features for the best split

        # extra_params user-specified
        'loss': 'squared_error',          # Loss function to optimize
        'criterion': 'friedman_mse',      # Function to measure the quality of a split
        'min_weight_fraction_leaf': 0.0,  # Min wtd fraction of sum of weights for leaf
        'min_impurity_decrease': 0.0,     # Min impurity decrease required to split a node
        'init': None,                     # Initial estimator (e.g., a constant predictor)
        'alpha': 0.9,                     # Quantile for 'huber' and 'quantile' loss functions
        'verbosity': 0,                   # Verbosity level
        'max_leaf_nodes': None,           # Maximum number of leaf nodes
        'warm_start': False,              # Reuse previous solution to add more estimators
        'validation_fraction': 0.1,       # Fraction of training data for validation
        'n_iter_no_change': None,         # Stop training if no improvement
        'tol': 1e-4,                      # Tolerance for early stopping
        'ccp_alpha': 0.0                  # Parameter for Minimal Cost-Complexity Pruning
        }
    
    # Update input data arguments with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    extra_params = {
        'random_state': data['random_state'],         
        'loss': data['loss'],     # Loss function to optimize. Default is 'squared_error' (mean squared error).
        'criterion': data['criterion'],     # Function to measure the quality of a split. Default is 'friedman_mse'.
        'min_weight_fraction_leaf': data['min_weight_fraction_leaf'],     # Minimum weighted fraction of the sum of weights for a leaf node. Default is 0.0.
        'min_impurity_decrease': data['min_impurity_decrease'],     # Minimum impurity decrease required to split a node. Default is 0.0.
        'init': data['init'],    # Initial estimator (e.g., a constant predictor). Default is None.
        'alpha': data['alpha'],    # Quantile for 'huber' and 'quantile' loss functions. Default is 0.9.
        'max_leaf_nodes': data['max_leaf_nodes'],    # Maximum number of leaf nodes. Default is None (unlimited).
        'warm_start': data['warm_start'],    # Reuse previous solution to add more estimators. Default is False.
        'validation_fraction': data['validation_fraction'],    # Fraction of training data for validation. Default is 0.1.
        'n_iter_no_change': data['n_iter_no_change'],    # Stop training if no improvement after this many iterations. Default is None.
        'tol': data['tol'],    # Tolerance for early stopping. Default is 1e-4.
        'ccp_alpha': data['ccp_alpha']    # Complexity parameter for Minimal Cost-Complexity Pruning. Default is 0.0.
    }

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import gbr_objective
    study.optimize(lambda trial: gbr_objective(
        trial, X_opt, y, **data), n_trials=data['n_trials'], n_jobs=data['n_jobs'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial
        
    best_params = study.best_params
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    print('Fitting GradientBoostingRegressor model with best parameters, please wait ...')
    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']
    fitted_model = GradientBoostingRegressor(
        **best_params, **extra_params).fit(
        X[model_outputs['selected_features']],y)
    
    # check to see of the model has intercept and coefficients
    if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
            and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
        intercept = fitted_model.intercept_
        coefficients = fitted_model.coef_
        # dataframe of model parameters, intercept and coefficients, including zero coefs
        n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
        popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
        for i in range(n_param):
            if i == 0:
                popt[0][i] = 'Intercept'
                popt[1][i] = fitted_model.intercept_
            else:
                popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                popt[1][i] = fitted_model.coef_[i-1]
        popt = pd.DataFrame(popt).T
        popt.columns = ['Feature', 'Parameter']
        # Table of intercept and coef
        popt_table = pd.DataFrame({
                "Feature": popt['Feature'],
                "Parameter": popt['Parameter']
            })
        popt_table.set_index('Feature',inplace=True)
        model_outputs['popt_table'] = popt_table
        
    '''
    # Calculate regression statistics
    y_pred = fitted_model.predict(X)
    stats = stats_given_y_pred(X,y,y_pred)
    
    # model objects and outputs returned by stacking
    # model_outputs['scaler'] = scaler                     # scaler used to standardize X
    # model_outputs['standardize'] = data['standardize']   # True: X_scaled was used to fit, False: X was used
    model_outputs['y_pred'] = stats['y_pred']
    model_outputs['residuals'] = stats['residuals']
    # model_objects = model
    
    # residual plot for training error
    if data['verbose'] == 'on':
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=stats['y_pred'],
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=stats['y_pred'],
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={stats['RMSE']:.3f})")
        plt.tight_layout()
        # plt.show()
        plt.savefig("GradientBoostingRegressor_predictions.png", dpi=300)
    
    # Make the model_outputs dataframes
    list1_name = ['r-squared', 'RMSE', 'n_samples']        
    list1_val = [stats["rsquared"], stats["RMSE"], stats["n_samples"]]
    
    stats = pd.DataFrame(
        {
            "Statistic": list1_name,
            "GradientBoostingRegressor": list1_val
        }
        )
    stats.set_index('Statistic',inplace=True)
    model_outputs['stats'] = stats
    print("GradientBoostingRegressor statistics of fitted model in model_outputs['stats']:")
    print('')
    print(model_outputs['stats'].to_markdown(index=True))
    print('')
    if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
        print("Parameters of fitted model in model_outputs['popt']:")
        print('')
        print(model_outputs['popt_table'].to_markdown(index=True))
        print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs
    '''

    # Goodness of fit statistics
    metrics = fitness_metrics(
        fitted_model, 
        X[model_outputs['selected_features']], y)
    stats = pd.DataFrame([metrics]).T
    stats.index.name = 'Statistic'
    stats.columns = ['GradientBoostingRegressor']
    model_outputs['metrics'] = metrics
    model_outputs['stats'] = stats
    model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])

    if data['verbose'] == 'on':
        print('')
        print("GradientBoostingRegressor goodness of fit to training data in model_outputs['stats']:")
        print('')
        print(model_outputs['stats'].to_markdown(index=True))
        print('')

    if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
        print("Parameters of fitted model in model_outputs['popt']:")
        print('')
        print(model_outputs['popt_table'].to_markdown(index=True))
        print('')

    # residual plot for training error
    if data['verbose'] == 'on':
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
        plt.tight_layout()
        # plt.show()
        plt.savefig("GradientBoostingRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def xgb(X, y, **kwargs):

    """
    Regression or classification with XGBoost

    by
    Greg Pelletier
    gjpelletier@gmail.com
    13-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        verbose= 'on' (default) or 'off'
        classify= False,            # Use XGBClassifier if True
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        gpu= True (default) or False to autodetect if the computer has a gpu and use it

        # params that are optimized by optuna
        learning_rate= 0.05,        # Step size shrinkage (also called eta).
        max_depth= 3,               # Maximum depth of a tree.
        min_child_weight= 0,        # Minimum sum of instance weight (hessian) needed in a child.
        subsample= 0.7,             # Fraction of samples used for training each tree.
        colsample_bytree= 0.7,      # Fraction of features used for each tree.
        gamma= 0,                   # Minimum loss reduction to make a split.
        reg_lambda= 1,              # L2 regularization term on weights.
        alpha= 0,                   # L1 regularization term on weights.
        n_estimators= 100,          # Number of boosting rounds (trees).

        # extra_params that are optional user-specified
        random_state= 42,           # Random seed for reproducibility.
        verbosity= 1,               # Verbosity of output (0 = silent, 1 = warnings, 2 = info).
        booster= "gbtree",          # Type of booster ('gbtree', 'gblinear', or 'dart').
        tree_method= "auto",        # Tree construction algorithm.
        nthread= -1,                # Number of parallel threads.
        colsample_bylevel= 1,       # Fraction of features used per tree level.
        colsample_bynode= 1,        # Fraction of features used per tree node.
        scale_pos_weight= 1,        # Balancing of positive and negative weights.
        base_score= 0.5,            # Initial prediction score (global bias).
        missing= np.nan,            # Value in the data to be treated as missing.
        importance_type= "gain",    # Feature importance type ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
        predictor= "auto",          # Type of predictor ('cpu_predictor', 'gpu_predictor').
        enable_categorical= False   # Whether to enable categorical data support.    
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'metrics': dict of goodness of fit metrics for train data
                - 'stats': dataframe of goodness of fit metrics for train data
                - 'params': core model parameters used for fitting
                - 'extra_params': extra model paramters used for fitting
                - 'selected_features': selected features for fitting
                - 'X_processed': final pre-processed and selected features
                - 'y_pred': best model predicted y

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = xgb(X, y)

    """

    from PyMLR import stats_given_y_pred
    from PyMLR import detect_dummy_variables, detect_gpu
    from PyMLR import check_X_y, fitness_metrics
    from PyMLR import preprocess_train, preprocess_test
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'classify': False,            # Use XGBClassifier if True
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                  # Autodetect if the computer has a gpu, if no gpu is detected then cpu will be used
        'device': None,               # placeholder for device argument 
        'objective': None,            # placeholder for objective argument 
        'num_class': None,            # placeholder for num_class argument 

        # params that are optimized by optuna
        'learning_rate': 0.05,        # Step size shrinkage (also called eta).
        'max_depth': 3,               # Maximum depth of a tree.
        'min_child_weight': 0,        # Minimum sum of instance weight (hessian) needed in a child.
        'subsample': 0.7,             # Fraction of samples used for training each tree.
        'colsample_bytree': 0.7,      # Fraction of features used for each tree.
        'gamma': 0,                   # Minimum loss reduction to make a split.
        'reg_lambda': 1,              # L2 regularization term on weights.
        'alpha': 0,               # L1 regularization term on weights.
        'n_estimators': 100,          # Number of boosting rounds (trees).

        # extra_params that are optional user-specified
        'random_state': 42,           # Random seed for reproducibility.
        'verbosity': 1,               # Verbosity of output (0 = silent, 1 = warnings, 2 = info).
        'booster': "gbtree",          # Type of booster ('gbtree', 'gblinear', or 'dart').
        'tree_method': "auto",        # Tree construction algorithm.
        'nthread': -1,                # Number of parallel threads.
        'colsample_bylevel': 1,       # Fraction of features used per tree level.
        'colsample_bynode': 1,        # Fraction of features used per tree node.
        'scale_pos_weight': 1,        # Balancing of positive and negative weights.
        'base_score': 0.5,            # Initial prediction score (global bias).
        'missing': np.nan,            # Value in the data to be treated as missing.
        'importance_type': "gain",    # Feature importance type ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
        'predictor': "auto",          # Type of predictor ('cpu_predictor', 'gpu_predictor').
        'enable_categorical': False   # Whether to enable categorical data support.    

    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Suppress warnings
    warnings.filterwarnings('ignore')
    # print('Fitting XGBRegressor model, please wait ...')
    if data['verbose'] == 'on':
        print('')
    
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign objective depending on type of model
    if data['classify']:
        # objective for XGBClassifier
        num_class = y.nunique()
        if num_class == 2:
            # binomial response variable
            data['objective'] = 'binary:logistic'
        else:
            # multinomial response variable
            data['objective'] = 'multi:softmax'
            data['num_class'] = num_class
    else:
        # objective for XGBRegressor
        data['objective'] = 'reg:squarederror'
    
    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()
    
    params = {
        # params that are optimized by optuna
        'learning_rate': data['learning_rate'],        
        'max_depth': data['max_depth'],               
        'min_child_weight': data['min_child_weight'],        
        'subsample': data['subsample'],             
        'colsample_bytree': data['colsample_bytree'],      
        'gamma': data['gamma'],                 
        'reg_lambda': data['reg_lambda'],            
        'alpha': data['alpha'],              
        'n_estimators': data['n_estimators']         
    }

    extra_params = {
        'random_state': data['random_state'],         
        'device': data['device'],                 
        'verbosity': data['verbosity'],              
        'objective': data['objective'], 
        'booster': data['booster'],          
        'tree_method': data['tree_method'],        
        'nthread': data['nthread'],                  
        'colsample_bylevel': data['colsample_bylevel'],       
        'colsample_bynode': data['colsample_bynode'],        
        # 'scale_pos_weight': data['scale_pos_weight'],        
        'base_score': data['base_score'],            
        'missing': data['missing'],           
        'importance_type': data['importance_type'],    
        # 'predictor': data['predictor'],          
        'enable_categorical': data['enable_categorical']  
    }
    
    if data['classify']:
        print('Fitting XGBClassifier model, please wait ...')
        if y.nunique() > 2:
            extra_params['num_class'] = data['num_class']
        fitted_model = XGBClassifier(**params, **extra_params).fit(X,y)
    else:
        print('Fitting XGBRegressor model, please wait ...')    
        extra_params['predictor'] = data['predictor']
        extra_params['scale_pos_weight'] = data['scale_pos_weight']
        fitted_model = XGBRegressor(**params, **extra_params).fit(X,y)
        
    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            # selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X, y)
            hfig.savefig("XGBClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X, y)
            hfig.savefig("XGBClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X, y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['XGBClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)    
        if data['verbose'] == 'on':
            print('')
            print("XGBClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:            
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X.columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X.columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table

        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X, y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['XGBRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)
    
        if data['verbose'] == 'on':
            print('')
            print("XGBRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("XGBRegressor_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def xgb_objective(trial, X, y, study, **kwargs):
    '''
    Optuna objective for optimizing XGBRegressor or XGBClassifier with optional feature selection.
    Supports selector choice, logs importances, and ensures reproducibility.
    '''

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)

    # Define hyperparameter space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", *kwargs["learning_rate"], log=True),
        "max_depth": trial.suggest_int("max_depth", *kwargs["max_depth"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *kwargs["min_child_weight"]),
        "subsample": trial.suggest_float("subsample", *kwargs["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *kwargs["colsample_bytree"]),
        "gamma": trial.suggest_float("gamma", *kwargs["gamma"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *kwargs["reg_lambda"], log=True),
        "alpha": trial.suggest_float("alpha", *kwargs["alpha"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *kwargs["n_estimators"]),
    }

    extra_params = {
        "random_state": seed,
        "device": kwargs["device"],
        "verbosity": kwargs["verbosity"],
        "objective": kwargs["objective"],
        "booster": kwargs["booster"],
        "tree_method": kwargs["tree_method"],
        "nthread": kwargs["nthread"],
        "colsample_bylevel": kwargs["colsample_bylevel"],
        "colsample_bynode": kwargs["colsample_bynode"],
        # "scale_pos_weight": kwargs["scale_pos_weight"],
        "base_score": kwargs["base_score"],
        "missing": kwargs["missing"],
        "importance_type": kwargs["importance_type"],
        # "predictor": kwargs["predictor"],
        "enable_categorical": kwargs["enable_categorical"],
    }

    if not kwargs['classify']:
        extra_params['predictor'] = kwargs['predictor']
        extra_params['scale_pos_weight'] = kwargs['scale_pos_weight']

    if kwargs['objective'] == 'multi:softmax':
        extra_params['num_class'] = kwargs['num_class']
    
    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        if kwargs['classify']:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", xgb.XGBClassifier(**params, **extra_params))
            ])
        else:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", xgb.XGBRegressor(**params, **extra_params))
            ])

    else:
        if kwargs['classify']:
            pipeline = Pipeline([
                ("regressor", xgb.XGBClassifier(**params, **extra_params))
            ])
        else:
            pipeline = Pipeline([
                ("regressor", xgb.XGBRegressor(**params, **extra_params))
            ])
        num_features = None

    # Cross-validated scoring
    if kwargs['classify']:
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="f1_weighted"        
            scoring=kwargs["scoring"]
        )
    else:
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )

    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean

def xgb_auto(X, y, **kwargs):

    """
    Autocalibration of XGBoost XGBRegressor or XGBClassifier hyper-parameters
    Preprocess with OneHotEncoder and StandardScaler
    Pipeline for feature selector and regressor

    by
    Greg Pelletier
    gjpelletier@gmail.com
    13-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        verbose= 'on' (default) or 'off'
        classify= False (default) or True to use XGBClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler

        NOTE: LightGBM and XGBoost automatically use dtype category as categorical features
              One-hot encoding is not recommended for categorical features treated this way
              The preprocessor will not one-hot encode dtype category 
              if enable_categorical=True

        'enable_categorical': True,         # Whether to enable categorical data support.    
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,                # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)

        gpu= True (default) or False to autodetect if the computer has a gpu and use it
        n_trials= 50,               # number of optuna trials
        n_splits= 5,                # number of splits for KFold CV
        pruning= False,             # prune poor optuna trials
        feature_selection= True,    # optuna feature selection

        # [min, max] ranges of params for model to be optimized by optuna:
        learning_rate= [1e-4, 1.0], # Step size shrinkage (also called eta).
        max_depth= [3, 15],         # Maximum depth of a tree.
        min_child_weight= [1, 10],  # Minimum sum of instance weight 
                                    # (hessian) needed in a child.
        subsample= [0.5, 1],        # Fraction of samples used for training each tree.
        colsample_bytree= [0.5, 1], # Fraction of features used for each tree.
        gamma= [1e-8, 10.0],        # Minimum loss reduction to make a split.
        reg_lambda= [1e-8, 10.0],   # L2 regularization term on weights.
        alpha= [1e-8, 10.0],        # L1 regularization term on weights.
        n_estimators= [100, 2000]   # Number of boosting rounds (trees).

        # extra_params for model that are optional user-specified
        random_state= 42,           # Random seed for reproducibility.
        verbosity= 1,               # Verbosity of output 
                                    # (0 = silent, 1 = warnings, 2 = info).
        booster= "gbtree",          # Type of booster ('gbtree', 'gblinear', or 'dart').
        tree_method= "auto",        # Tree construction algorithm.
        nthread= -1,                # Number of parallel threads.
        colsample_bylevel= 1,       # Fraction of features used per tree level.
        colsample_bynode= 1,        # Fraction of features used per tree node.
        scale_pos_weight= 1,        # Balancing of positive and negative weights.
        base_score= 0.5,            # Initial prediction score (global bias).
        missing= np.nan,            # Value in the data to be treated as missing.
        importance_type= "gain",    # Feature importance type 
                                    # ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
        predictor= "auto",          # Type of predictor ('cpu_predictor', 'gpu_predictor').
        enable_categorical= False   # Whether to enable categorical data support.    
        preprocessing options:
            use_encoder (bool): True (default) or False 
            use_scaler (bool): True (default) or False 
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns
                    - 'non_numeric_cats': non-numeric categorical columns
                    - 'continous_cols': continuous numerical columns                
                - 'optuna_study': optimzed optuna study object
                - 'optuna_model': optimzed optuna model object
                - 'best_trial': best trial from the optuna study
                - 'feature_selection' = option to select features (True, False)
                - 'selected_features' = selected features
                - 'best_params': best model hyper-parameters found by optuna
                - 'extra_params': other model options used to fit the model
                - 'metrics': dict of goodness of fit metrics for train data
                - 'stats': dataframe of goodness of fit metrics for train data
                - 'X_processed': pre-processed X with encoding and scaling
                - 'y_pred': best model predicted y

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = xgb_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, fitness_metrics, check_X_y
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    import optuna

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,                  # Use XGBClassifier if True
        'preprocess': True,                 # Apply OneHotEncoder and StandardScaler
        'preprocess_result': None,          # dict of  the following result from 
                                            # preprocess_train if available:         
                                            # - encoder          (OneHotEncoder) 
                                            # - scaler           (StandardScaler)
                                            # - categorical_cols (categorical columns)
                                            # - non_numeric_cats (non-numeric cats)
                                            # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'verbose': 'on',
        'gpu': True,                        # Autodetect to use gpu if present
        'n_splits': 5,                      # number of splits for KFold CV

        'pruning': False,                   # prune poor optuna trials
        'feature_selection': True,          # optuna feature selection
        'scoring': None,                     # cross_val_score scoring name
        'show_trial_progress': True,         # print trial numbers during execution

        # params that are optimized by optuna
        'learning_rate': [1e-4, 1.0],       # Step size shrinkage (also called eta).
        'max_depth': [3, 15],               # Maximum depth of a tree.
        'min_child_weight': [1, 10],        # Minimum sum of instance weight (hessian) needed in a child.
        'subsample': [0.5, 1],              # Fraction of samples used for training each tree.
        'colsample_bytree': [0.5, 1],       # Fraction of features used for each tree.
        'gamma': [1e-8, 10.0],              # Minimum loss reduction to make a split.
        'reg_lambda': [1e-8, 10.0],         # L2 regularization term on weights.
        'alpha': [1e-8, 10.0],              # L1 regularization term on weights.
        'n_estimators': [100, 2000],        # Number of boosting rounds (trees).

        # extra_params that are optional user-specified
        'random_state': 42,           # Random seed for reproducibility.
        'verbosity': 1,               # Verbosity of output (0 = silent, 1 = warnings, 2 = info).
        # 'objective': "reg:squarederror",  # Loss function for regression.
        'booster': "gbtree",          # Type of booster ('gbtree', 'gblinear', or 'dart').
        'tree_method': "auto",        # Tree construction algorithm.
        'nthread': -1,                # Number of parallel threads.
        'colsample_bylevel': 1,       # Fraction of features used per tree level.
        'colsample_bynode': 1,        # Fraction of features used per tree node.
        'scale_pos_weight': 1,        # Balancing of positive and negative weights.
        'base_score': 0.5,            # Initial prediction score (global bias).
        'missing': np.nan,            # Value in the data to be treated as missing.
        'importance_type': "gain",    # Feature importance type ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
        'predictor': "auto",          # Type of predictor ('cpu_predictor', 'gpu_predictor').
        'enable_categorical': False   # Whether to enable categorical data support.    
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")
        
    # assign objective depending on type of model
    if data['classify']:
        # objective for XGBClassifier
        num_class = y.nunique()
        if num_class == 2:
            # binomial response variable
            data['objective'] = 'binary:logistic'
        else:
            # multinomial response variable
            data['objective'] = 'multi:softmax'
            data['num_class'] = num_class
    else:
        # objective for XGBRegressor
        data['objective'] = 'reg:squarederror'

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'enable_categorical': data['enable_categorical'],
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()
    
    extra_params = {
        'random_state': data['random_state'],         
        'device': data['device'],                 
        'verbosity': data['verbosity'],              
        'objective': data['objective'], 
        'booster': data['booster'],          
        'tree_method': data['tree_method'],        
        'nthread': data['nthread'],                  
        'colsample_bylevel': data['colsample_bylevel'],       
        'colsample_bynode': data['colsample_bynode'],        
        # 'scale_pos_weight': data['scale_pos_weight'],        
        'base_score': data['base_score'],            
        'missing': data['missing'],           
        'importance_type': data['importance_type'],    
        # 'predictor': data['predictor'],          
        'enable_categorical': data['enable_categorical']  
    }

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import xgb_objective
    study.optimize(lambda trial: xgb_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial
        
    best_params = study.best_params
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    # Final fit wit best_params and selected_features
    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']

    if data['classify']:
        print('Fitting XGBClassifier model with best parameters, please wait ...')
        fitted_model = XGBClassifier(
            **best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
    else:
        print('Fitting XGBRegressor model with best parameters, please wait ...')
        extra_params['predictor'] = data['predictor']
        extra_params['scale_pos_weight'] = data['scale_pos_weight']
        fitted_model = XGBRegressor(
            **best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
       
    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("XGBClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("XGBClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['XGBClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("XGBClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:        
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table

        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['XGBRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("XGBRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("XGBRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def lgbm(X, y, **kwargs):

    """
    Regression or classification with LightBGM

    by
    Greg Pelletier
    gjpelletier@gmail.com
    09-Sep-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        verbose= 'on' (default) or 'off'
        classify= False,            # Use XGBClassifier if True
        preprocess= True,           # Apply OneHotEncoder and StandardScaler

        NOTE: LightGBM and XGBoost automatically use dtype category as categorical features
              One-hot encoding is not recommended for categorical features treated this way
              The preprocessor will not one-hot encode dtype category 
              if enable_categorical=True
        
        'enable_categorical': True,         # Whether to enable categorical data support.    
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,                # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        

        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        n_trials= 50,               # number of optuna trials
        n_splits= 5,                # number of splits for KFold CV
        pruning= False,             # prune poor optuna trials
        feature_selection= False,   # optuna feature selection

        # params that are optimized by optuna
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 1e-8,
        'reg_lambda': 1e-8,
        'class_weight': None,
        'boosting_type': "gbdt",

        # extra_params that are optional user-specified
        'num_threads': -1,    # available cpus for LGBM, -1 to use all cpus
        'random_state': 42,   # random seed for reproducibility
        'objective': None,    # auto set to 'regression', 'binary', or 'multiclass'
        'verbosity': -1,                                    

        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'metrics': dict of goodness of fit metrics for train data
                - 'stats': dataframe of goodness of fit metrics for train data
                - 'params': core model parameters used for fitting
                - 'extra_params': extra model paramters used for fitting
                - 'selected_features': selected features for fitting
                - 'X_processed': final pre-processed and selected features
                - 'y_pred': best model predicted y

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = lgbm(X, y)

    """

    from PyMLR import stats_given_y_pred
    from PyMLR import detect_dummy_variables, detect_gpu
    from PyMLR import check_X_y, fitness_metrics
    from PyMLR import preprocess_train, preprocess_test
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from lightgbm import LGBMRegressor, LGBMClassifier

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'classify': False,            # Use XGBClassifier if True
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'enable_categorical': True,   # Whether to enable categorical data support.    
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                  # Autodetect if the computer has a gpu, if no gpu is detected then cpu will be used
        'device': None,               # placeholder for device argument 
        'objective': None,            # placeholder for objective argument 
        'num_class': None,            # placeholder for num_class argument 

        # params that are optimized by optuna
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': 15,
        'min_child_samples': 20,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 1e-8,
        'reg_lambda': 1e-8,
        'class_weight': None,
        'min_split_gain': 0.0,
        'min_child_weight': 1e-3,
        'boosting_type': "gbdt",

        # extra_params that are optional user-specified
        'num_threads': -1,    # available cpus for LGBM, -1 to use all cpus
        'random_state': 42,   # random seed for reproducibility
        'objective': None,    # auto set to 'regression', 'binary', or 'multiclass'
        'verbosity': -1,                                    

    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Suppress warnings
    warnings.filterwarnings('ignore')
    # print('Fitting XGBRegressor model, please wait ...')
    if data['verbose'] == 'on':
        print('')
    
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign objective depending on type of model
    if data['classify']:
        # objective for XGBClassifier
        num_class = y.nunique()
        if num_class == 2:
            # binomial response variable
            data['objective'] = 'binary'
        else:
            # multinomial response variable
            data['objective'] = 'multiclass'
            data['num_class'] = num_class
    else:
        # objective for XGBRegressor
        data['objective'] = 'regression'
    
    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()
    
    params = {
        "learning_rate": data["learning_rate"],
        "max_depth": data["max_depth"],
        "n_estimators": data["n_estimators"],
        "min_child_samples": data["min_child_samples"],
        "subsample": data["subsample"],
        "colsample_bytree": data["colsample_bytree"],
        "reg_lambda": data["reg_lambda"],
        "reg_alpha": data["reg_alpha"],
        "num_leaves": data["num_leaves"],
        'min_split_gain': data["min_split_gain"],
        'min_child_weight': data["min_child_weight"],
        "boosting_type": data["boosting_type"],
    }

    extra_params = {
        "random_state": data['random_state'],
        'num_threads': data['num_threads'],
        'objective': data['objective'],
        'verbosity': data['verbosity'],                                    
    }
    
    if data['classify']:
        print('Fitting LGBMClassifier model, please wait ...')
        if y.nunique() > 2:
            extra_params['num_class'] = data['num_class']
        fitted_model = LGBMClassifier(**params, **extra_params).fit(X,y)
    else:
        print('Fitting LGBMRegressor model, please wait ...')    
        fitted_model = LGBMRegressor(**params, **extra_params).fit(X,y)
        
    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            # selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X, y)
            hfig.savefig("LGBMClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X, y)
            hfig.savefig("LGBMClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X, y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['LGBMClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)    
        if data['verbose'] == 'on':
            print('')
            print("LGBMClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:            
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X.columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X.columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table

        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X, y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['LGBMRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)
    
        if data['verbose'] == 'on':
            print('')
            print("LGBMRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("LGBMRegressor_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def lgbm_objective(trial, X, y, study, **kwargs):
    '''
    Optuna objective for optimizing LGBMRegressor or LGBMClassifier with optional feature selection.
    Supports selector choice, logs importances, and ensures reproducibility.
    '''

    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor, LGBMClassifier
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)

    # Define hyperparameter space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", *kwargs["learning_rate"], log=True),
        "max_depth": trial.suggest_int("max_depth", *kwargs["max_depth"]),
        "n_estimators": trial.suggest_int("n_estimators", *kwargs["n_estimators"]),
        "min_child_samples": trial.suggest_int("min_child_samples", *kwargs["min_child_samples"]),
        "subsample": trial.suggest_float("subsample", *kwargs["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *kwargs["colsample_bytree"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *kwargs["reg_lambda"], log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", *kwargs["reg_alpha"], log=True),
        "num_leaves": trial.suggest_int("num_leaves", *kwargs["num_leaves"]),
        "min_split_gain": trial.suggest_float("min_split_gain", *kwargs["min_split_gain"]),
        "min_child_weight": trial.suggest_float("min_child_weight", *kwargs["min_child_weight"], log=True),
        "boosting_type": trial.suggest_categorical("boosting_type", kwargs["boosting_type"]),
    }

    extra_params = {
        "random_state": seed,
        'num_threads': kwargs['num_threads'],
        'objective': kwargs['objective'],
        'verbosity': kwargs['verbosity'],                                    
    }

    if kwargs['objective'] == 'multiclass':
        extra_params['num_class'] = kwargs['num_class']
    
    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        if kwargs['classify']:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", LGBMClassifier(**params, **extra_params))
            ])
        else:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", LGBMRegressor(**params, **extra_params))
            ])

    else:
        if kwargs['classify']:
            pipeline = Pipeline([
                ("regressor", LGBMClassifier(**params, **extra_params))
            ])
        else:
            pipeline = Pipeline([
                ("regressor", LGBMRegressor(**params, **extra_params))
            ])
        num_features = None

    # Cross-validated scoring
    if kwargs['classify']:
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="f1_weighted"        
            scoring=kwargs["scoring"]
        )
    else:
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )

    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    feature_names = getattr(model_step, "feature_name_", None)    
    if importances is not None:
        # trial.set_user_attr("feature_importances", importances.tolist())
        # trial.set_user_attr("feature_names", feature_names.tolist())
        trial.set_user_attr("feature_importances", importances)
        trial.set_user_attr("feature_names", feature_names)

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean

def lgbm_auto(X, y, **kwargs):

    """
    Autocalibration of LightGBM LGBMRegressor or LGBMClassifier hyper-parameters
    Preprocess with OneHotEncoder and StandardScaler
    Pipeline for feature selector and regressor

    by
    Greg Pelletier
    gjpelletier@gmail.com
    09-Sep-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        verbose= 'on' (default) or 'off'
        classify= False (default) or True to use LGBMClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler

        NOTE: LightGBM and XGBoost automatically use dtype category as categorical features
              One-hot encoding is not recommended for categorical features treated this way
              The preprocessor will not one-hot encode dtype category 
              if enable_categorical=True
        
        'enable_categorical': True,         # Whether to enable categorical data support.    
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,                # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        

        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        n_trials= 50,               # number of optuna trials
        n_splits= 5,                # number of splits for KFold CV
        pruning= False,             # prune poor optuna trials
        feature_selection= False,   # optuna feature selection

        # params that are optimized by optuna
        'n_estimators': [100, 2000],
        'learning_rate': [1e-3, 0.3],
        'num_leaves': [16, 256],
        'max_depth': [3, 15],
        'min_child_samples': [5, 100],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.5, 1.0],
        'reg_alpha': [1e-8, 10.0],
        'reg_lambda': [1e-8, 10.0],
        'class_weight': [None, "balanced"],
        'min_split_gain': [0.0, 1.0],
        'min_child_weight': [1e-3, 10.0],
        'boosting_type': ["gbdt", "dart"],

        # extra_params that are optional user-specified
        'num_threads': -1,    # available cpus for LGBM, -1 to use all cpus
        'random_state': 42,   # random seed for reproducibility
        'objective': None,    # auto set to 'regression', 'binary', or 'multiclass'
        'verbosity': -1,                                    

        preprocessing options:
            use_encoder (bool): True (default) or False 
            use_scaler (bool): True (default) or False 
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns
                    - 'non_numeric_cats': non-numeric categorical columns
                    - 'continous_cols': continuous numerical columns                
                - 'optuna_study': optimzed optuna study object
                - 'optuna_model': optimzed optuna model object
                - 'best_trial': best trial from the optuna study
                - 'feature_selection' = option to select features (True, False)
                - 'selected_features' = selected features
                - 'best_params': best model hyper-parameters found by optuna
                - 'extra_params': other model options used to fit the model
                - 'metrics': dict of goodness of fit metrics for train data
                - 'stats': dataframe of goodness of fit metrics for train data
                - 'X_processed': pre-processed X with encoding and scaling
                - 'y_pred': best model predicted y

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = lgbm_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, fitness_metrics, check_X_y
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna
    from lightgbm import LGBMRegressor, LGBMClassifier

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,                  # Use XGBClassifier if True
        'preprocess': True,                 # Apply OneHotEncoder and StandardScaler
        'preprocess_result': None,          # dict of  the following result from 
                                            # preprocess_train if available:         
                                            # - encoder          (OneHotEncoder) 
                                            # - scaler           (StandardScaler)
                                            # - categorical_cols (categorical columns)
                                            # - non_numeric_cats (non-numeric cats)
                                            # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'enable_categorical': True,         # Whether to enable categorical data support.    
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,                # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'verbose': 'on',
        'gpu': True,                     # Autodetect gpu if present (not used by LGBM)
        'n_splits': 5,                   # number of splits for KFold CV

        'pruning': False,                # prune poor optuna trials
        'feature_selection': False,      # optuna feature selection
        'scoring': None,                 # cross_val_score scoring name
        'show_trial_progress': True,     # print trial numbers during execution

        # params that are optimized by optuna
        'n_estimators': [100, 2000],
        'learning_rate': [1e-3, 0.3],
        'num_leaves': [16, 256],
        'max_depth': [3, 15],
        'min_child_samples': [5, 100],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.5, 1.0],
        'reg_alpha': [1e-8, 10.0],
        'reg_lambda': [1e-8, 10.0],
        'class_weight': [None, "balanced"],
        'min_split_gain': [0.0, 1.0],
        'min_child_weight': [1e-3, 10.0],
        'boosting_type': ["gbdt", "dart"],

        # extra_params that are optional user-specified
        'num_threads': -1,    # available cpus for LGBM, -1 to use all cpus
        'random_state': 42,   # random seed for reproducibility
        'objective': None,    # auto set to 'regression', 'binary', or 'multiclass'
        'verbosity': -1,                                    
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present (not used for LightGBM)
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")
        
    # assign objective depending on type of model
    if data['classify']:
        # objective for LGBMClassifier
        num_class = y.nunique()
        if num_class == 2:
            # binomial response variable
            data['objective'] = 'binary'
        else:
            # multinomial response variable
            data['objective'] = 'multiclass'
            data['num_class'] = num_class
    else:
        # objective for LGBMRegressor
        data['objective'] = 'regression'

    # assign cross_val_score scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'enable_categorical': data['enable_categorical'],
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()
    
    extra_params = {
        'num_threads': data['num_threads'],          
        'random_state': data['random_state'],  
        'objective': data['objective'], 
        'verbosity': data['verbosity'],                                    
    }

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import lgbm_objective
    study.optimize(lambda trial: lgbm_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial
        
    best_params = study.best_params
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    # Final fit wit best_params and selected_features
    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']

    if data['classify']:
        print('Fitting LGBMClassifier model with best parameters, please wait ...')
        fitted_model = LGBMClassifier(
            **best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
    else:
        print('Fitting LGBMRegressor model with best parameters, please wait ...')
        fitted_model = LGBMRegressor(
            **best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
       
    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("LGBMClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("LGBMClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['LGBMClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("LGBMClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:        
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table

        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['LGBMRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("LGBMRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("LGBMRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def catboost(X, y, **kwargs):

    """
    CatboostRegressor or CatBoost Classifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    25-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        random_state= 42,    # initial random seed
        'classify': False,          # Use CatBoostClassifier if True
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        selected_features= None,    # pre-optimized selected features
        verbose= 'on',        # 'on' to display summary stats and residual plots
        gpu= False,           # Autodetect to use gpu if present
        thread_count= -1,     # number of CPU cores to use (-1 for all cores)
        
        # [min, max] range of params optimized by optuna
        cat_features= None,         # CatBoost categorical features
        'subsample': 1.0,              # frac of samples for train each iter
        'colsample_bylevel': 1.0,      # frac features to determine best split         
        learning_rate= [0.01, 0.3],         # Balances step size in gradient updates.
        depth= [4, 10],                     # Controls tree depth
        iterations= [100, 3000],            # Number of boosting iterations
        l2_leaf_reg= [1, 10],               # Regularization strength       
        random_strength= [0, 1],            # Adds noise for diversity
        bagging_temperature= [0.1, 1.0],    # Controls randomness in sampling
        border_count= [32, 255],            # Number of bins for feature discretization
        min_data_in_leaf= [1, 100],         # Minimum samples per leaf         
        max_bin= [64, 255],                 # Number of bins for feature quantization

        # categorical params optimized by optuna
        use_border_count= [True, False]     # True = use border_count 
                                            # (best for categorical features)
                                            # False = use max_bin 
                                            # (best for continuous features)
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        model_objects, model_outputs
            model_objects is the fitted model object 
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = catboost(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from catboost import CatBoostRegressor, CatBoostClassifier

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    
    
    # Define default values of input data arguments
    defaults = {
        'random_state': 42,     # Random seed for reproducibility.
        'classify': False,            # Use CatBoostClassifier if True
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',        # 'on' to display stats and residual plots
        'gpu': False,           # Autodetect to use gpu if present
        'devices': '0',         # Which GPU to use (0 to use first GPU)
        'thread_count': -1,     # number of CPUs to use (-1 for all cores)

        # [min, max] range of params optimized by optuna
        'cat_features': None,          # CatBoost categorical features
        'subsample': 1.0,              # frac of samples for train each iter
        'colsample_bylevel': 1.0,      # frac features to determine best split         
        'learning_rate': 0.03,         # Balances step size in gradient updates.
        'depth': 6,                    # Controls tree depth
        'iterations': 1000,            # Number of boosting iterations
        'l2_leaf_reg': 3,              # Regularization strength       
        'random_strength': 1.0,        # Adds noise for diversity
        'bagging_temperature': 1.0,    # Controls randomness in sampling
        'min_data_in_leaf': 1,         # Minimum samples per leaf         
        'grow_policy': 'SymmetricTree',    # all leaves from the last tree level 
                                           # are split with the same condition
        'border_count': 128,           # Number of bins for feature discretization
        'max_bin': 255,                # Number of bins for feature quantization
        'use_border_count': True       # True: border_count, False: max_bin
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'GPU'
        else:
            data['device'] = 'CPU'
    else:
        data['device'] = 'CPU'
    
    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    if data['cat_features']==None:
        X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['cat_features']!=None:
            print('Warning: The PyMLR processor is not compatible cat_features,')
            print('therefore X is not automatically preprocessed.')
            print('Use of cat_features is experimental and may not work properly')
        else:
            if data['preprocess_result']!=None:
                # print('preprocess_test')
                X = preprocess_test(X, data['preprocess_result'])
            else:
                kwargs_pre = {
                    'use_encoder': data['use_encoder'],
                    'use_scaler': data['use_scaler'],
                    'threshold_cat': data['threshold_cat'],
                    'scale': data['scale'], 
                    'unskew_pos': data['unskew_pos'], 
                    'threshold_skew_pos': data['threshold_skew_pos'],
                    'unskew_neg': data['unskew_neg'], 
                    'threshold_skew_neg': data['threshold_skew_neg']        
                }
                data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
                X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    params = {        
        # [min, max] range of params optimized by optuna
        'cat_features': data['cat_features'],
        'subsample': data['subsample'],
        'colsample_bylevel': data['colsample_bylevel'],
        'learning_rate': data['learning_rate'],
        'depth': data['depth'],                    # Controls tree depth
        'iterations': data['iterations'],            # Number of boosting iterations
        'l2_leaf_reg': data['l2_leaf_reg'],              # Regularization strength       
        'random_strength': data['random_strength'],        # Adds noise for diversity
        'bagging_temperature': data['bagging_temperature'],    # Controls randomness in sampling
        'min_data_in_leaf': data['min_data_in_leaf'],         # Minimum samples per leaf         
    }

    if data['use_border_count']:
        params['border_count'] = data['border_count']
    else:
        params['max_bin'] = data['max_bin']
            
    extra_params = {
        'random_seed': data['random_state'],         
        'task_type': data['device']                   
    }

    if data['device'] == 'GPU':
        extra_params['devices'] = data['devices']
    else:
        extra_params['thread_count'] = data['thread_count']
    
    if data['classify']:
        print('Fitting CatBoostClassifier model, please wait ...')
        fitted_model = CatBoostClassifier(
            **params, **extra_params, verbose=False).fit(X,y)
    else:
        print('Fitting CatBoostRegressor model, please wait ...')
        fitted_model = CatBoostRegressor(
            **params, **extra_params, verbose=False).fit(X,y)
        
    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            # selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X, y)
            hfig.savefig("CatBoostClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X, y)
            hfig.savefig("CatBoostClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X, y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['CatBoostClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)    
        if data['verbose'] == 'on':
            print('')
            print("CatBoostClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:            
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X.columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X.columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X, y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['CatBoostRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)

        if data['verbose'] == 'on':
            print('')
            print("CatBoostRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')

        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')

        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("CatBoostRegressor_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def catboost_objective(trial, X, y, study, **kwargs):
    '''
    Optuna objective for optimizing CatBoostRegressor or CatBoostClassifier 
    with optional feature selection.
    '''
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from catboost import CatBoostRegressor, CatBoostClassifier

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')
    
    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)

    # Define hyperparameter space
    params = {
        "subsample": trial.suggest_float("subsample",
            *kwargs['subsample']),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel",
            *kwargs['colsample_bylevel']),
        "learning_rate": trial.suggest_float("learning_rate",
            *kwargs['learning_rate'], log=True),
        "depth": trial.suggest_int("depth",
            *kwargs['depth']),
        "iterations": trial.suggest_int("iterations",
            *kwargs['iterations']),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg",
            *kwargs['l2_leaf_reg'], log=True),
        "random_strength": trial.suggest_float("random_strength",
            *kwargs['random_strength']),
        "bagging_temperature": trial.suggest_float("bagging_temperature",
            *kwargs['bagging_temperature']),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf",
            *kwargs['min_data_in_leaf']),
    }

    grow_policy = trial.suggest_categorical("grow_policy", ["Depthwise", "SymmetricTree"])
    boosting_type = "Ordered" if grow_policy == "SymmetricTree" else "Plain"
    
    params["grow_policy"] = grow_policy
    params["boosting_type"] = boosting_type

    use_border_count = trial.suggest_categorical("use_border_count", [True, False])
    if use_border_count:
        params["border_count"] = trial.suggest_int("border_count",
            *kwargs['border_count'])
    else:
        params["max_bin"] = trial.suggest_int("max_bin",
            *kwargs['max_bin'])
    
    extra_params = {
        'random_seed': kwargs['random_state'],         
        'task_type': kwargs['device']                   
    }

    if kwargs['cat_features']!=None:
        extra_params['cat_features'] = kwargs['cat_features']         

    if kwargs['device'] == 'GPU':
        extra_params['devices'] = kwargs['devices']
    else:
        extra_params['thread_count'] = kwargs['thread_count']
    
    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        if kwargs['classify']:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", CatBoostClassifier(**params, **extra_params, verbose=False))
            ])
        else:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", CatBoostRegressor(**params, **extra_params, verbose=False))
            ])

    else:

        if kwargs['classify']:
            pipeline = Pipeline([
                ("regressor", CatBoostClassifier(**params, **extra_params, verbose=False))
            ])
        else:
            pipeline = Pipeline([
                ("regressor", CatBoostRegressor(**params, **extra_params, verbose=False))
            ])
            num_features = None

    # Cross-validated scoring
    if kwargs['classify']:
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
    else:
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        scoring=kwargs["scoring"]
    )
        
    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean
 
def catboost_auto(X, y, **kwargs):

    """
    Autocalibration of hyperparameters for 
    CatBoostRegressor or CatBoostClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    25-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        random_state= 42,    # initial random seed
        'classify': False,          # Use CatBoostClassifier if True
        n_trials= 50,         # number of optuna trials
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        feature_selection= True,    # optuna feature selection
        verbose= 'on',        # 'on' to display summary stats and residual plots
        n_splits= 5,          # number of splits for KFold CV
        gpu= False,           # Autodetect to use gpu if present
        thread_count= -1,     # number of CPU cores to use (-1 for all cores)
        
        # [min, max] range of params optimized by optuna
        'subsample': [0.05, 1.0],             # frac of samples for train each iter
        'colsample_bylevel': [0.05, 1.0],     # frac features to determine best split         
        learning_rate= [0.01, 0.3],         # Balances step size in gradient updates.
        depth= [4, 10],                     # Controls tree depth
        iterations= [100, 3000],            # Number of boosting iterations
        l2_leaf_reg= [1, 10],               # Regularization strength       
        random_strength= [0, 1],            # Adds noise for diversity
        bagging_temperature= [0.1, 1.0],    # Controls randomness in sampling
        border_count= [32, 255],            # Number of bins for feature discretization
        min_data_in_leaf= [1, 100],         # Minimum samples per leaf         
        max_bin= [64, 255],                 # Number of bins for feature quantization

        # categorical params optimized by optuna
        use_border_count= [True, False]     # True = use border_count 
                                            # (best for categorical features)
                                            # False = use max_bin 
                                            # (best for continuous features)
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'optuna_study': optimzed optuna study object
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = catboost_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna
    from catboost import CatBoostRegressor, CatBoostClassifier

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'random_state': 42,     # Random seed for reproducibility.
        'classify': False,            # Use CatBoostClassifier if True
        'n_trials': 50,               # number of optuna trials
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'verbose': 'on',        # 'on' to display stats and residual plots
        'gpu': False,           # Autodetect to use gpu if present
        'devices': '0',         # Which GPU to use (0 to use first GPU)
        'n_splits': 5,          # number of splits for KFold CV
        'thread_count': -1,     # number of CPUs to use (-1 for all cores)

        'pruning': False,             # prune poor optuna trials
        'feature_selection': True,    # optuna feature selection
        'scoring': None,                     # cross_val_score scoring name
        'show_trial_progress': True,         # print trial numbers during execution
        
        # [min, max] range of params optimized by optuna
        'cat_features': None,                 # CatBoost categorical features         
        'subsample': [0.05, 1.0],             # frac of samples for train each iter
        'colsample_bylevel': [0.05, 1.0],     # frac features to determine best split         
        'learning_rate': [0.01, 0.3],         # Balances step size in gradient updates.
        'depth': [4, 10],                     # Controls tree depth
        'iterations': [100, 3000],            # Number of boosting iterations
        'l2_leaf_reg': [1, 10],               # Regularization strength       
        'random_strength': [0, 1],            # Adds noise for diversity
        'bagging_temperature': [0.1, 1.0],    # Controls randomness in sampling
        'border_count': [32, 255],            # Number of bins for feature discretization
        'min_data_in_leaf': [1, 100],         # Minimum samples per leaf         
        'max_bin': [64, 255],                 # Number of bins for feature quantization
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'GPU'
        else:
            data['device'] = 'CPU'
    else:
        data['device'] = 'CPU'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['cat_features']!=None:
            print('Warning: The PyMLR processor is not compatible with use of cat_features,')
            print('therefore X is not automatically preprocessed.')
            print('Use of cat_features is experimental and may not work properly')
        else:
            if data['preprocess_result']!=None:
                # print('preprocess_test')
                X = preprocess_test(X, data['preprocess_result'])
            else:
                kwargs_pre = {
                    'use_encoder': data['use_encoder'],
                    'use_scaler': data['use_scaler'],
                    'threshold_cat': data['threshold_cat'],
                    'scale': data['scale'], 
                    'unskew_pos': data['unskew_pos'], 
                    'threshold_skew_pos': data['threshold_skew_pos'],
                    'unskew_neg': data['unskew_neg'], 
                    'threshold_skew_neg': data['threshold_skew_neg']        
                }
                data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
                X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    extra_params = {
        'cat_features': data['cat_features'],         
        'random_seed': data['random_state'],         
        'task_type': data['device'], 
        'verbose': False
    }

    if data['device'] == 'GPU':
        extra_params['devices'] = data['devices']
    else:
        extra_params['thread_count'] = data['thread_count']

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import catboost_objective
    study.optimize(lambda trial: catboost_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['best_trial'] = study.best_trial

    best_params = study.best_params
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    if 'use_border_count' in best_params:
        del best_params['use_border_count']
    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']

    if data['classify']:
        print('Fitting CatBoostClassifier model with best parameters, please wait ...')
        fitted_model = CatBoostClassifier(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
    else:
        print('Fitting CatBoostRegressor model with best parameters, please wait ...')
        fitted_model = CatBoostRegressor(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("CatBoostClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("CatBoostClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['CatBoostClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("CatBoostClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table

        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['CatBoostRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])

        if data['verbose'] == 'on':
            print('')
            print("CatBoostRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')

        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')

        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("CatBoostRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs
 
def forest(X, y, **kwargs):

    """
    Regression with sklearn RandomForestRegressor
    or
    Classification with sklearn RandomForestClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    15-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,                     # number of optuna trials
        classify= False,            # True to use RandomForestClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',                    # 'on' to display all 
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV

        # params that are optimized by optuna
        n_estimators= 100,                # number of trees in the forest
        max_depth= None,                  # max depth of a tree
        min_samples_split= 2,             # min samples to split internal node
        min_samples_leaf= 1,              # min samples to be at a leaf node
        max_features= 1.0,                # number of features to consider 
                                          # when looking for the best split
        max_leaf_nodes= None,             # max number of leaf nodes
        min_impurity_decrease= 0.0,       # node will be split if this 
                                          # decrease of the impurity 
        ccp_alpha= 0.0,                   # parameter for 
                                          # minimum cost-complexity pruning
        bootstrap= True,                  # whether bootstrap samples are used

        # extra_params that are optional user-specified
        random_state= 42,                 # random seed for reproducibility
        criterion= 'squared_error',       # function to measure quality of split
        min_weight_fraction_leaf= 0.0,    # min weighted fraction of the 
                                          # sum total of weights 
                                          # (of all the input samples) 
                                          # required to be at a leaf node
                                          # greater than or equal to this value
        oob_score= False,                 # whether to use out-of-bag samples
        n_jobs= -1,                       # number of jobs to run in parallel
                                          # -1 means use all cpu cores
        warm_start= False,                # reuse the previous solution
        max_samples= None,                # If bootstrap is True, the number 
                                          # of samples to draw from X 
                                          # to train each base estimator
        monotonic_cst= None               # monotonicity constraint 
                                          # to enforce on each feature
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = forest(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # Use RandomForestClassifier if True
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                        # Autodetect to use gpu if present
        'n_splits': 5,                      # number of splits for KFold CV

        # params 
        'n_estimators': 100,                # number of trees in the forest
        'max_depth': None,                  # max depth of a tree
        'min_samples_split': 2,             # min samples to split internal node
        'min_samples_leaf': 1,              # min samples to be at a leaf node
        'max_features': 1.0,                # number of features to consider 
                                            # when looking for the best split
        'max_leaf_nodes': None,             # max number of leaf nodes
        'min_impurity_decrease': 0.0,       # node will be split if this 
                                            # induces a decrease of the impurity 
                                            # greater than or equal to this value
        'ccp_alpha': 0.0,                   # parameter for 
                                            # Minimum Cost-Complexity Pruning
        'bootstrap': True,                  # whether bootstrap samples are used

        # extra_params that are optional user-specified
        'random_state': 42,                 # random seed for reproducibility
        'criterion': 'squared_error',       # function to measure quality of split
        'min_weight_fraction_leaf': 0.0,    # min weighted fraction of the 
                                            # sum total of weights 
                                            # (of all the input samples) 
                                            # required to be at a leaf node
        'oob_score': False,                 # whether to use out-of-bag samples
        'n_jobs': -1,                       # number of jobs to run in parallel
                                            # -1 means use all cpu cores
        'warm_start': False,                # reuse the previous solution
        'max_samples': None,                # If bootstrap is True, the number 
                                            # of samples to draw from X 
                                            # to train each base estimator
        'monotonic_cst': None               # monotonicity constraint 
                                            # to enforce on each feature
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign criterion depending on type of model
    if data['classify']:
        data['criterion'] = 'gini'
    else:
        data['criterion'] = 'squared_error'
    
    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    params = {
        'n_estimators': data['n_estimators'],               
        'max_depth': data['max_depth'],                 
        'min_samples_split': data['min_samples_split'],            
        'min_samples_leaf': data['min_samples_leaf'],             
        'max_features': data['max_features'],             
        'max_leaf_nodes': data['max_leaf_nodes'],           
        'min_impurity_decrease': data['min_impurity_decrease'],       
        'ccp_alpha': data['ccp_alpha'],                 
        'bootstrap': data['bootstrap']                 
    }

    extra_params = {
        'verbose': 0,                 
        'random_state': data['random_state'],                
        'criterion': data['criterion'],       
        'min_weight_fraction_leaf': data['min_weight_fraction_leaf'],    
        'oob_score': data['oob_score'],                 
        'n_jobs': data['n_jobs'],                      
        'warm_start': data['warm_start'],               
        'max_samples': data['max_samples'],                
        'monotonic_cst': data['monotonic_cst']             
    }

    if data['classify']:
        print('Fitting RandomForestClassifier model, please wait ...')
        fitted_model = RandomForestClassifier(**params, **extra_params).fit(X,y)
    else:
        print('Fitting RandomForestRegressor model, please wait ...')
        fitted_model = RandomForestRegressor(**params, **extra_params).fit(X,y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            # selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X, y)
            hfig.savefig("RandomForestClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X, y)
            hfig.savefig("RandomForestClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X, y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['RandomForestClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)    
        if data['verbose'] == 'on':
            print('')
            print("RandomForestClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:            
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X.columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X.columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X, y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['RandomForestRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)
    
        if data['verbose'] == 'on':
            print('')
            print("RandomForestRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("RandomForestRegressor_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def forest_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by optuna 
    to find the optimum hyper-parameters for 
    sklearn RandomForestRegressor or RandomForestClassifier
    '''
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators",
            *kwargs['n_estimators']),
        "max_depth": trial.suggest_int("max_depth",
            *kwargs['max_depth']),
        "min_samples_split": trial.suggest_int("min_samples_split",
            *kwargs['min_samples_split'], log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf",
            *kwargs['min_samples_leaf'], log=True),
        "max_features": trial.suggest_float("max_features",
            *kwargs['max_features']),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes",
            *kwargs['max_leaf_nodes'], log=True),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease",
            *kwargs['min_impurity_decrease']),
        "ccp_alpha": trial.suggest_float("ccp_alpha",
            *kwargs['ccp_alpha'], log=True),
        "bootstrap":  trial.suggest_categorical("bootstrap",
            kwargs['bootstrap'])
    }    

    extra_params = {
        'verbose': 0,                 
        'random_state': kwargs['random_state'],                
        'criterion': kwargs['criterion'],       
        'min_weight_fraction_leaf': kwargs['min_weight_fraction_leaf'],    
        'oob_score': kwargs['oob_score'],                 
        'n_jobs': kwargs['n_jobs'],                      
        'warm_start': kwargs['warm_start'],               
        'max_samples': kwargs['max_samples'],                
        'monotonic_cst': kwargs['monotonic_cst']             
    }

    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        if kwargs['classify']:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", RandomForestClassifier(**params, **extra_params))
            ])
        else:        
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", RandomForestRegressor(**params, **extra_params))
            ])

    else:

        if kwargs['classify']:
            pipeline = Pipeline([
                ("regressor", RandomForestClassifier(**params, **extra_params))
            ])
        else:        

            pipeline = Pipeline([
                ("regressor", RandomForestRegressor(**params, **extra_params))
            ])

        num_features = None

    # Cross-validated scoring
    if kwargs['classify']:
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="f1_weighted"        
            scoring=kwargs["scoring"]
        )
    else:
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )
    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean
      
def forest_auto(X, y, **kwargs):

    """
    Autocalibration of hyperparameters for 
    sklearn RandomForestRegressor or RandomForestClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    15-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,                     # number of optuna trials
        classify= False,            # True for RandomForestClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',                    # 'on' to display all 
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV
        pruning= False,                   # prune poor optuna trials
        feature_selection= True,          # optuna feature selection

        # params that are optimized by optuna
        n_estimators= [50, 500],          # number of trees in the forest
        max_depth= [3, 30],               # max depth of a tree
        min_samples_split= [2, 50],       # min samples to split internal node
        min_samples_leaf= [1, 50],        # min samples to be at a leaf node
        max_features= [0.1, 1.0],         # number of features to consider 
                                          # when looking for the best split
        max_leaf_nodes= [10, 1000],       # max number of leaf nodes
        min_impurity_decrease= [0.0, 0.1],    # node will be split if this 
                                              # decrease of the impurity 
        ccp_alpha= [0.0001, 0.1],         # parameter for 
                                          # minimum cost-complexity pruning
        bootstrap= [True, False],         # whether bootstrap samples are used

        # extra_params that are optional user-specified
        random_state= 42,                 # random seed for reproducibility
        min_weight_fraction_leaf= 0.0,    # min weighted fraction of the 
                                          # sum total of weights 
                                          # (of all the input samples) 
                                          # required to be at a leaf node
                                          # greater than or equal to this value
        oob_score= False,                 # whether to use out-of-bag samples
        n_jobs= -1,                       # number of jobs to run in parallel
                                          # -1 means use all cpu cores
        warm_start= False,                # reuse the previous solution
        max_samples= None,                # If bootstrap is True, the number 
                                          # of samples to draw from X 
                                          # to train each base estimator
        monotonic_cst= None               # monotonicity constraint 
                                          # to enforce on each feature
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'optuna_study': optimzed optuna study object
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = forest_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # True for RandomForestClassifier
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                        # Autodetect to use gpu if present
        'n_splits': 5,                      # number of splits for KFold CV

        'pruning': False,                   # prune poor optuna trials
        'feature_selection': True,          # optuna feature selection
        'scoring': None,                     # cross_val_score scoring name
        'show_trial_progress': True,         # print trial numbers during execution
        
        # params that are optimized by optuna
        'n_estimators': [50, 500],          # number of trees in the forest
        'max_depth': [3, 30],               # max depth of a tree
        'min_samples_split': [2, 50],       # min samples to split internal node
        'min_samples_leaf': [1, 50],        # min samples to be at a leaf node
        'max_features': [0.1, 1.0],         # number of features to consider 
                                            # when looking for the best split
        'max_leaf_nodes': [10, 1000],       # max number of leaf nodes
        'min_impurity_decrease': [0.0, 0.1],   # node will be split if this 
                                            # induces a decrease of the impurity 
                                            # greater than or equal to this value
        'ccp_alpha': [0.0001, 0.1],         # parameter for 
                                            # Minimum Cost-Complexity Pruning
        'bootstrap': [True, False],         # whether bootstrap samples are used

        # extra_params that are optional user-specified
        'random_state': 42,                 # random seed for reproducibility
        # 'criterion': 'squared_error',       # function to measure quality of split
        'min_weight_fraction_leaf': 0.0,    # min weighted fraction of the 
                                            # sum total of weights 
                                            # (of all the input samples) 
                                            # required to be at a leaf node
        'oob_score': False,                 # whether to use out-of-bag samples
        'n_jobs': -1,                       # number of jobs to run in parallel
                                            # -1 means use all cpu cores
        'warm_start': False,                # reuse the previous solution
        'max_samples': None,                # If bootstrap is True, the number 
                                            # of samples to draw from X 
                                            # to train each base estimator
        'monotonic_cst': None               # monotonicity constraint 
                                            # to enforce on each feature
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"

    # assign criterion depending on type of model
    if data['classify']:
        data['criterion'] = 'gini'
    else:
        data['criterion'] = 'squared_error'
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    extra_params = {
        'verbose': 0,                 
        'random_state': data['random_state'],                
        'criterion': data['criterion'],       
        'min_weight_fraction_leaf': data['min_weight_fraction_leaf'],    
        'oob_score': data['oob_score'],                 
        'n_jobs': data['n_jobs'],                      
        'warm_start': data['warm_start'],               
        'max_samples': data['max_samples'],                
        'monotonic_cst': data['monotonic_cst']             
    }

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import forest_objective
    study.optimize(lambda trial: forest_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial
        
    best_params = study.best_params
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']
    
    if data['classify']:
        print('Fitting RandomForestClassifier model with best parameters, please wait ...')    
        fitted_model = RandomForestClassifier(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
    else:    
        print('Fitting RandomForestRegressor model with best parameters, please wait ...')    
        fitted_model = RandomForestRegressor(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("RandomForestClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("RandomForestClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['RandomForestClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("RandomForestClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['RandomForestRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("RandomForestRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("RandomForestRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def knn(X, y, **kwargs):

    """
    Regression with sklearn KNeighborsRegressor
    or
    Classiciation with KNeighborsClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    13-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        # general params that are user-specified
        random_state= 42,           # random seed for reproducibility
        n_trials= 50,               # number of optuna trials
        classify= False,            # True for KNeighborsClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',
        gpu= True,                  # Autodetect to use gpu if present

        # model params
        n_neighbors= 5,             # number of neighbors
        p= 2,                       # power for Minkowski
        leaf_size= 30,              # Leaf size for BallTree or KDTree
        weights= "uniform",         # weight function
        metric= "minkowski",        # for distance comp
        algorithm= "auto",          # algorithm    
        
        # model extra_params that are optional user-specified
        n_jobs= -1,                 # number of jobs to run in parallel    
        metric_params= None         # for user-specified metrics
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = knn(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {

        # general params that are user-specified
        'random_state': 42,                 # random seed for reproducibility
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # True for KNeighborsClassifier
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                        # Autodetect to use gpu if present

        # # PCA may be added in a future version
        # 'pca_transform': True,            # force PCA transform
        # 'pca_transform': False,           # force no PCA transform
        # 'pca': None,                      # fitted PCA transform object
        # 'n_components': None,             # number of PCA components
        
        # model params
        'n_neighbors': 5,                   # number of neighbors
        'p': 2,                             # power for Minkowski
        'leaf_size': 30,                    # Leaf size for BallTree or KDTree

        # categorical model params
        'weights': "uniform",               # weight function
        'metric': "minkowski",              # for distance comp
        'algorithm': "auto",                # algorithm    
        
        # model extra_params that are optional user-specified
        'n_jobs': -1,                       # number of jobs to run in parallel    
        'metric_params': None               # for user-specified metrics

    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")
    
    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    '''
    if data['pca_transform'] and data['pca'] == None:
        # fit new PCA transformer
        if n_components == None:
            n_components = min(X.shape[0],X.shape[1])   # lesser of n_samples and n_features
        else:
            n_components = data['n_components']
        pca = PCA(n_components=n_components).fit(X)
        X = pca.transform(X)        
        n_components = pca.n_components_
        X = pd.DataFrame(pca.transform(X), columns= [f"PC_{i+1}" for i in range(n_components)])
        X.index = y.index    
    if data['pca_transform'] and data['pca'] != None:
        # use input PCA transformer
        pca = data['pca']
        n_components = pca.n_components_
        X = pd.DataFrame(pca.transform(X), columns= [f"PC_{i+1}" for i in range(n_components)])
        X.index = y.index    
    if data['pca_transform']:
        model_outputs['pca_transform'] = data['pca_transform']              
        model_outputs['pca'] = pca                  
        model_outputs['n_components'] = n_components              
    else:
        model_outputs['pca_transform'] = data['pca_transform']              
        model_outputs['pca'] = data['pca']                  
        model_outputs['n_components'] = data['n_components']             
    '''
            
    params = {
        'n_neighbors': data['n_neighbors'],
        'leaf_size': data['leaf_size'],
        'weights': data['weights'],
        'metric': data['metric'],
        'algorithm': data['algorithm'],
        'p': data['p'],
    }    

    extra_params = {
        # extra_params that are optional user-specified
        'n_jobs': data['n_jobs'],                       # number of jobs to run in parallel    
        'metric_params': data['metric_params']               # for user-specified metrics
    }
    
    if data['classify']:
        print('Fitting KNeighborsClassifier model, please wait ...')
        fitted_model = KNeighborsClassifier(**params, **extra_params).fit(X,y)
    else:
        print('Fitting KNeighborsRegressor model, please wait ...')
        fitted_model = KNeighborsRegressor(**params, **extra_params).fit(X,y)
        
    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            # selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X, y)
            hfig.savefig("KNeighborsClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X, y)
            hfig.savefig("KNeighborsClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X, y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['KNeighborsClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)    
        if data['verbose'] == 'on':
            print('')
            print("KNeighborsClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:                
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X.columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X.columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
    
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X, y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['KNeighborsRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)
    
        if data['verbose'] == 'on':
            print('')
            print("KNeighborsRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("KNeighborsRegressor_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def knn_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by optuna 
    to find the optimum hyper-parameters for 
    sklearn KNeighborsRegressor or KNeighborsClassifier
    '''
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.metrics import mean_squared_error, accuracy_score
    import optuna
    # from sklearn.decomposition import PCA
    
    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)

    '''
    # PCA Transformation: Optimize number of components
    if isinstance(kwargs['pca_transform'], list):
        # optuna chooses if pca_Transorm
        pca_transform = trial.suggest_categorical('pca_transform', kwargs['pca_transform']) 
    else:
        # force input value of pca_transform= True or False
        pca_transform = kwargs['pca_transform']
    if pca_transform:
        n_components = trial.suggest_int("n_components", 5, X.shape[1])  
        pca = PCA(n_components=n_components).fit(X)
        X = pd.DataFrame(pca.transform(X), columns= [f"PC_{i+1}" for i in range(n_components)])
        X.index = y.index
    else:
        pca = None
        n_components = None
    '''
    
    params = {
        'n_neighbors': trial.suggest_int("n_neighbors",
            kwargs['n_neighbors'][0], kwargs['n_neighbors'][1]),
        'leaf_size': trial.suggest_int("leaf_size",
            kwargs['leaf_size'][0], kwargs['leaf_size'][1])  ,  
        'weights': trial.suggest_categorical("weights",
            kwargs['weights']),
        'metric': trial.suggest_categorical("metric",
            kwargs['metric']),
        'algorithm': trial.suggest_categorical("algorithm",
            kwargs['algorithm']) 
    }    

    if params['metric'] == "minkowski":
        params['p']= trial.suggest_int("p",
            kwargs['p'][0], kwargs['p'][1], log=True) 
    else:
        params['p']= None

    extra_params = {
        'n_jobs': kwargs['n_jobs'],             
        'metric_params': kwargs['metric_params']             
    }

    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        if kwargs['classify']:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", KNeighborsClassifier(**params, **extra_params))
            ])
        else:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", KNeighborsRegressor(**params, **extra_params))
            ])

    else:

        if kwargs['classify']:
            pipeline = Pipeline([
                ("regressor", KNeighborsClassifier(**params, **extra_params))
            ])
        else:
            pipeline = Pipeline([
                ("regressor", KNeighborsRegressor(**params, **extra_params))
            ])

        num_features = None

    # Cross-validated scoring
    if kwargs['classify']:
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="f1_weighted"        
            scoring=kwargs["scoring"]
        )
    else:
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )

    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    # prevent over-fitting of the train data
    if not kwargs['allow_overfit']:
        # pipeline.fit(X, y)
        train_pred = pipeline.predict(X)

        if not kwargs['classify']:
            train_mse = mean_squared_error(y, train_pred)    
            if train_mse <= kwargs['tol']:
                raise optuna.exceptions.TrialPruned(
                    "Training MSE is zero — likely overfitting")
        else:
            train_accuracy = accuracy_score(y, train_pred)    
            if train_accuracy >= 1.0 - kwargs['tol']:
                raise optuna.exceptions.TrialPruned(
                    "Training Accuracy is 1.0 — likely overfitting")
    
    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean
     
def knn_auto(X, y, **kwargs):

    """
    Autocalibration of KNeighborsRegressor or KNeighborsClassifier hyperparameters

    by
    Greg Pelletier
    gjpelletier@gmail.com
    13-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        # general params that are user-specified
        random_state= 42,                 # random seed for reproducibility
        n_trials= 50,                     # number of optuna trials
        classify= False,                  # True for KNeighborsClassifier
        preprocess= True,                 # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,          # dict of the following result from 
                                          # preprocess_train if available:         
                                          # - encoder          (OneHotEncoder)
                                          # - scaler           (StandardScaler)
                                          # - categorical_cols (categorical cols)
                                          # - non_numeric_cats (non-num cat cols)
                                          # - continuous_cols  (continuous cols)
        verbose= 'on',
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV
        pruning= False,                   # prune poor optuna trials
        allow_overfit= True,              # allow optuna to overfit train data
        tol= 1e-6,                        # tolerance for overfit
                                          # used if allow_overfit=False
                                          # as min allowable MSE
                                          # or max closeness to 1.0 accuracy

        pruning= False,                   # prune poor optuna trials
        feature_selection= True,          # optuna feature selection
        
        # model params that are optimized by optuna
        n_neighbors= [1, 50],             # number of neighbors
        p= [1, 5],                        # power for Minkowski
        leaf_size= [5, 100],              # Leaf size for BallTree or KDTree
        weights= ["uniform", "distance"],    # weight function
        metric= ["euclidean", "manhattan", "minkowski"],  # for distance comp
        algorithm= ["ball_tree", "kd_tree", "brute"],    # algorithm    
        
        # model extra_params that are optional user-specified
        n_jobs= -1,                       # number of jobs to run in parallel    
        metric_params= None               # for user-specified metrics
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'optuna_study': optimzed optuna study object
                - 'best_trial': best trial from the optuna study
                - 'feature_selection' = best_trial option to select features (True, False)
                - 'selected_features' = best_trial selected features
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = knn_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier 
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {

        # general params that are user-specified
        'random_state': 42,                 # random seed for reproducibility
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # True for KNeighborsClassifier
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                        # Autodetect to use gpu if present
        'n_splits': 5,                      # number of splits for KFold CV
        'allow_overfit': True,              # allow optuna to overfit train data
        'tol': 1e-6,                        # tolerance for overfit
                                            # used if allow_overfit=False
                                            # as min allowable MSE

        'pruning': False,                   # prune poor optuna trials
        'feature_selection': True,          # optuna feature selection
        'scoring': None,                     # cross_val_score scoring name
        'show_trial_progress': True,         # print trial numbers during execution
        
        # user params that are optimized by optuna (for future version)
        # 'pca_transform': [True, False],     # optuna chooses if PCA transform
        
        # [min,max] model params that are optimized by optuna
        'n_neighbors': [1, 50],             # number of neighbors
        'p': [1, 5],                        # power for Minkowski
        'leaf_size': [5, 100],              # Leaf size for BallTree or KDTree

        # categorical model params that are optimized by optuna
        'weights': ["uniform", "distance"],    # weight function
        'metric': ["euclidean", "manhattan", "minkowski"],  # for distance comp
        'algorithm': ["ball_tree", "kd_tree", "brute"],    # algorithm    
        
        # model extra_params that are optional user-specified
        'n_jobs': -1,                       # number of jobs to run in parallel    
        'metric_params': None               # for user-specified metrics
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}
     
    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    extra_params = {
        # extra_params that are optional user-specified
        'n_jobs': data['n_jobs'],                       # number of jobs to run in parallel    
        'metric_params': data['metric_params']               # for user-specified metrics
    }

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)
       
     # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import knn_objective
    study.optimize(lambda trial: knn_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial
        
    best_params = study.best_params
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']

    if data['classify']:
        print('Fitting KNeighborsClassifier model with best parameters, please wait ...')
        fitted_model = KNeighborsClassifier(
            **best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
    else:    
        fitted_model = KNeighborsRegressor(
            **best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
            
    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("KNeighborsClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("KNeighborsClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['KNeighborsClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("KNeighborsClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table

        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['KNeighborsRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("KNeighborsRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("KNeighborsRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def plot_confusion_matrix(model, X, y):
    '''
    plot the confusion matrix
    for binary or multinomial LogisticRegression.
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)        
    hfig = plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    return hfig
    
def plot_roc_auc(model, X, y):
    """
    Plots ROC curve(s) and computes AUC score(s) 
    for binary or multinomial LogisticRegression.
    
    Parameters:
        model: Fitted sklearn LogisticRegression model
        X: Feature matrix
        y: True labels
    
    Returns:
        None (displays the plot)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    
    y_score = model.predict_proba(X)
    classes = model.classes_
    n_classes = len(classes)

    # Binarize the output
    y_bin = label_binarize(y, classes=classes)

    # plt.figure(figsize=(8, 6))
    hfig = plt.figure(figsize=(6, 4))

    if n_classes == 2:
        # Binary classification case
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f"ROC curve (AUC = {roc_auc:.3f})")
    else:
        # Multiclass case - one ROC curve per class
        colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'cyan'])
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, 
                     label=f"Class {classes[i]} (AUC = {roc_auc:.3f})")
        
        # Macro-average AUC (optional summary line)
        all_fpr = np.unique(np.concatenate(
            [roc_curve(y_bin[:, i], y_score[:, i])[0] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, color='black', linestyle='--', 
                 lw=2, label=f"Average (AUC = {macro_auc:.3f})")

    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    # plt.grid(True)
    plt.tight_layout()
    # plt.show()

    return hfig

def test_model_logistic(
        model, X, y, preprocess_result=None, selected_features=None):
    '''
    plot the confusion matrix and ROC curve and fitness metrics
    for test data sets for LogisticRegression
    using the preprocess_result and selected_features 
    from the fitted model using training data
    previously saved by using logistic_auto, logistic, 
    or preprocess_train
    '''

    import numpy as np
    import pandas as pd
    from PyMLR import (preprocess_test, plot_confusion_matrix, 
        plot_roc_auc, fitness_metrics_logistic, check_X_y)

    # copy X and y to avoid altering originals
    X = X.copy()
    y = y.copy()

    # check X and y and put into dataframe if needed
    X, y = check_X_y(X, y)
    
    if preprocess_result!=None:
        X = X.copy()    # copy X to avoid changing the original
        X = preprocess_test(X, preprocess_result)

    if selected_features==None:
        selected_features = X.columns.to_list()

    # Goodness of fit statistics
    metrics = fitness_metrics_logistic(
        model, 
        X[selected_features], y)
    stats = pd.DataFrame([metrics]).T
    stats.index.name = 'Statistic'
    stats.columns = ['LogisticRegression']
    results = {}
    results['metrics'] = metrics
    results['stats'] = stats
    print('')
    print("LogisticRegression goodness of fit to testing data in results['stats']:")
    print('')
    print(results['stats'].to_markdown(index=True))
    print('')
        
    hfig_cm = plot_confusion_matrix(model, X[selected_features],y)    
    hfig_cm.savefig("LogisticRegression_confusion_matrix_test.png", dpi=300)
    results['hfig_cm'] = hfig_cm
    
    hfig_roc = plot_roc_auc(model, X[selected_features],y)    
    hfig_roc.savefig("LogisticRegression_ROC_curve_test.png", dpi=300)
    results['hfig_roc'] = hfig_roc

    results['y_pred'] = model.predict(X[selected_features])
    
    return results

def test_model_classifier(
        model, X, y, preprocess_result=None, selected_features=None):
    '''
    plot the confusion matrix and ROC curve and fitness metrics
    for test data sets for LogisticRegression
    using the preprocess_result and selected_features 
    from the fitted model using training data
    previously saved by using logistic_auto, logistic, 
    or preprocess_train
    '''

    import numpy as np
    import pandas as pd
    from PyMLR import (preprocess_test, plot_confusion_matrix, 
        plot_roc_auc, fitness_metrics_logistic, check_X_y)

    # copy X and y to avoid altering originals
    X = X.copy()
    y = y.copy()

    # check X and y and put into dataframe if needed
    X, y = check_X_y(X, y)
    
    if preprocess_result!=None:
        X = X.copy()    # copy X to avoid changing the original
        X = preprocess_test(X, preprocess_result)

    if selected_features==None:
        selected_features = X.columns.to_list()

    # Goodness of fit statistics
    metrics = fitness_metrics_logistic(
        model, 
        X[selected_features], y, brier=False)
    stats = pd.DataFrame([metrics]).T
    stats.index.name = 'Statistic'
    stats.columns = ['Classifier']
    results = {}
    results['metrics'] = metrics
    results['stats'] = stats
    print('')
    print("Classifier model goodness of fit to testing data in results['stats']:")
    print('')
    print(results['stats'].to_markdown(index=True))
    print('')
        
    hfig_cm = plot_confusion_matrix(model, X[selected_features],y)    
    hfig_cm.savefig("Classifier_confusion_matrix_test.png", dpi=300)
    results['hfig_cm'] = hfig_cm
    
    hfig_roc = plot_roc_auc(model, X[selected_features],y)    
    hfig_roc.savefig("Classifier_ROC_curve_test.png", dpi=300)
    results['hfig_roc'] = hfig_roc

    results['y_pred'] = model.predict(X[selected_features])
    
    return results

def logistic(X, y, **kwargs):

    """
    LogisticRegression with user-specified inputs
    Beta version

    by
    Greg Pelletier
    gjpelletier@gmail.com
    17-June-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe or array of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe or array of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        # general params that are user-specified
        preprocess= True,         # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,  # dict of the following result from 
                                  # preprocess_train if available:         
                                  # - encoder          (OneHotEncoder)
                                  # - scaler           (StandardScaler)
                                  # - categorical_cols (categorical cols)
                                  # - non_numeric_cats (non-num cat cols)
                                  # - continuous_cols  (continuous cols)
        selected_features= None,  # pre-optimized selected features
        verbose= 'on',      # display summary stats and plots
        gpu= True,          # autodetect gpu if present
        threshold_cat= 10,  # threshold for number of 
                            # unique values to identify
                            # categorical numeric features
                            # to encode with OneHotEncoder
         
        # [min,max] model params that are optimized by optuna
        C= 1.0,             # Inverse regularization strength

        # categorical model params that are optimized by optuna
        solver= lbfgs',,    # optimization algorithm
        penalty= l2,        # norm of the penalty
        
        # model extra_params that are optional user-specified
        random_state= 42,   # random seed for reproducibility
        max_iter= 500,      # max iterations for solver
        n_jobs= -1,         # number of jobs to run in parallel    
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    Note: StandardScaler standardizing of continuous numerical features 
    and OneHoteEncoder encoding of categorical numerical features is optional
    and is applied by default with kwarg preprocess=True.
    If fitted encoder, scaler, categorical_cols, and continuous_cols
    are provided then these will be used for preprocessing.

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'metrics': dict of goodness of fit metrics for train data
                - 'stats': dataframe of goodness of fit metrics for train data
                - 'params': core model parameters used for fitting
                - 'extra_params': extra model paramters used for fitting
                - 'selected_features': selected features for fitting
                - 'X_processed': final pre-processed and selected features
                - 'y_pred': best model predicted y

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column if it is a dataframe

    EXAMPLE 
    model_objects, model_outputs = logistic(X, y)

    """

    from PyMLR import preprocess_train, preprocess_test 
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    from PyMLR import detect_dummy_variables, detect_gpu
    from PyMLR import check_X_y
    import time
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay, confusion_matrix
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna
    import seaborn as sns

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {

        # general params that are user-specified
        'preprocess': True,    # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,  # dict of  the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder) 
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical columns)
                                    # - non_numeric_cats (non-numeric cats)
                                    # - continuous_cols  (continuous columns)
        'selected_features': None,         # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                       # autodetect gpu if present
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------        
        # [min,max] model params that are optimized by optuna
        'C': 1.0,                          # Inverse regularization strength
        # categorical model params that are optimized by optuna
        'solver': 'lbfgs',                 # optimization algorithm
        'penalty': 'l2',                   # norm of the penalty        
        # model extra_params that are optional user-specified
        'random_state': 42,                # random seed for reproducibility
        'max_iter': 500,                   # max iterations for  solver
        'n_jobs': -1,                      # number of jobs to run in parallel    
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()

    # print('before preprocess_train: ',X.shape, y.shape)
    X, y = check_X_y(X,y)
    # print('after check_X_y: ',X.shape, y.shape,X.columns)

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()
                                                    
    print('Fitting LogisticRegression model with best parameters, please wait ...')

    params = {
        'C': data['C'],                    # Inverse regularization strength
        'solver': data['solver'],          # optimization algorithm
        'penalty': data['penalty']         # norm of the penalty
    }
    
    extra_params = {
        'random_state': data['random_state'],
        'max_iter': data['max_iter'],
        'n_jobs': data['n_jobs'],
        'verbose': 0
    }

    # save params and extra_params
    model_outputs['params'] = params
    model_outputs['extra_params'] = extra_params

    fitted_model = LogisticRegression(**params, **extra_params).fit(X,y)

    if data['verbose'] == 'on':

        # confusion matrix
        selected_features = model_outputs['selected_features']
        hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
        hfig.savefig("LogisticRegression_confusion_matrix.png", dpi=300)
        
        # ROC curve with AUC
        selected_features = model_outputs['selected_features']
        hfig = plot_roc_auc(fitted_model, X[selected_features], y)
        hfig.savefig("LogisticRegression_ROC_curve.png", dpi=300)
        
    # Goodness of fit statistics
    metrics = fitness_metrics_logistic(
        fitted_model, 
        X[model_outputs['selected_features']], y)
    stats = pd.DataFrame([metrics]).T
    stats.index.name = 'Statistic'
    stats.columns = ['LogisticRegression']
    model_outputs['metrics'] = metrics
    model_outputs['stats'] = stats
    model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])

    if data['verbose'] == 'on':
        print('')
        print("LogisticRegression goodness of fit to training data in model_outputs['stats']:")
        print('')
        print(model_outputs['stats'].to_markdown(index=True))
        print('')
    
    # Print the run time
    fit_time = time.time() - start_time

    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def logistic_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by Optuna to optimize 
    hyperparameters for sklearn LogisticRegression
    with optional pipeline-integrated SelectKBest feature selection.
    '''
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.pipeline import make_pipeline

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')

    # Set random seed for reproducibility
    seed = kwargs.get('random_state', 42)
    np.random.seed(seed)

    # Hyperparameter search space
    params = {
        "C": trial.suggest_float("C", kwargs["C"][0], kwargs["C"][1], log=True),
        "solver": trial.suggest_categorical("solver", kwargs["solver"]),
    }

    # Only relevant penalties based on solver
    params["penalty"] = trial.suggest_categorical(
        "penalty", kwargs["penalty"]) if params["solver"] in ['liblinear', 'saga'] else 'l2'

    extra_params = {
        'random_state': seed,
        'max_iter': kwargs['max_iter'],
        'n_jobs': kwargs['n_jobs'],
        'verbose': 0
    }

    # Optional feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int(
            "num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector = SelectKBest(
            lambda X_, y_: mutual_info_classif(X_, y_, random_state=seed),
            k=num_features
        )
        pipeline = make_pipeline(selector, LogisticRegression(**params, **extra_params))
    else:
        pipeline = make_pipeline(LogisticRegression(**params, **extra_params))
        num_features = None  # Will track in case we want to log it

    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=kwargs["scoring"], n_jobs=kwargs['n_jobs'])
    score_mean = scores.mean()

    # Optional full pipeline fit to log selected features
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps['selectkbest']
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Save outputs to trial
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("model", pipeline)

    return score_mean
  
def logistic_auto(X, y, **kwargs):

    """
    Autocalibration of LogisticRegression hyperparameters
    Beta version

    by
    Greg Pelletier
    gjpelletier@gmail.com
    15-June-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe or array of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe or array of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        # general params that are user-specified
        n_trials= 50,             # Number of optuna trials
        preprocess= True,         # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,  # dict of the following result from 
                                  # preprocess_train if available:         
                                  # - encoder          (OneHotEncoder)
                                  # - scaler           (StandardScaler)
                                  # - categorical_cols (categorical cols)
                                  # - non_numeric_cats (non-num cat cols)
                                  # - continuous_cols  (continuous cols)
        verbose= 'on',            # display summary stats and plots
        gpu= True,                # autodetect gpu if present
        n_splits= 5,              # number of splits for KFold CV
        pruning= False,           # prune poor optuna trials
        feature_selection= True,  # optuna feature selection
        threshold_cat= 10,        # threshold for number of 
                                  # unique values to identify
                                  # categorical numeric features
                                  # to encode with OneHotEncoder
         
        # [min,max] model params that are optimized by optuna
        C= [1e-4, 10.0],          # Inverse regularization strength

        # categorical model params that are optimized by optuna
        solver= ['liblinear', 'lbfgs', 'saga'],   # optimization algorithm
        penalty= ['l1', 'l2],                     # norm of the penalty
        
        # model extra_params that are optional user-specified
        random_state= 42,         # random seed for reproducibility
        max_iter= 500,            # max iterations for solver
        n_jobs= -1,               # number of jobs to run in parallel    
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    Note: StandardScaler standardizing of continuous numerical features 
    and OneHoteEncoder encoding of categorical numerical features is optional
    and is applied by default with kwarg preprocess=True

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'optuna_study': optimzed optuna study object
                - 'optuna_model': optimzed optuna model object
                - 'best_trial': best trial from the optuna study
                - 'feature_selection' = option to select features (True, False)
                - 'selected_features' = selected features
                - 'best_params': best model hyper-parameters found by optuna
                - 'extra_params': other model options used to fit the model
                - 'metrics': dict of goodness of fit metrics for train data
                - 'stats': dataframe of goodness of fit metrics for train data
                - 'X_processed': pre-processed X with encoding and scaling
                - 'y_pred': best model predicted y

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column if it is a dataframe

    EXAMPLE 
    model_objects, model_outputs = logistic_auto(X, y)

    """

    from PyMLR import preprocess_train, preprocess_test
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    from PyMLR import detect_dummy_variables, detect_gpu, check_X_y
    import time
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay, confusion_matrix
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna
    import seaborn as sns

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {

        # general params that are user-specified
        'n_trials': 50,             # Number of optuna trials
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                # Autodetect to use gpu if present
        'n_splits': 5,              # number of splits for KFold CV
        'pruning': False,           # prune poor optuna trials
        'feature_selection': True,  # optuna feature selection
        'scoring': None,                     # cross_val_score scoring name
        'show_trial_progress': True,         # print trial numbers during execution
        
        # [min,max] model params that are optimized by optuna
        'C': [1e-4, 10.0],                  # Inverse regularization strength

        # categorical model params that are optimized by optuna
        # 'solver': ['liblinear', 'lbfgs', 'saga'],   # optimization algorithm
        'solver': ['lbfgs', 'saga'],   # optimization algorithm
        'penalty': ['l1', 'l2'],      # norm of the penalty
        
        # model extra_params that are optional user-specified
        'random_state': 42,                 # random seed for reproducibility
        'max_iter': 500,                    # max iterations for  solver
        'n_jobs': -1,                       # number of jobs to run in parallel    
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # assign default scoring 
    if data['scoring'] == None:
        data['scoring'] = "f1_weighted"

    X, y = check_X_y(X,y)

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()
    # print('after preprocess_train: ',X.shape, y.shape,X.columns)
    
    extra_params = {
        'random_state': data['random_state'],
        'max_iter': data['max_iter'],
        'n_jobs': data['n_jobs'],
        'verbose': 0
    }

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import logistic_objective
    study.optimize(
        lambda trial: logistic_objective(trial, X_opt, y, study, **data), 
        n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['best_trial'] = study.best_trial
    
    # user attributes for optuna
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')

    print('Fitting LogisticRegression model with best parameters, please wait ...')

    # extract best_params from study and remove non-model params
    best_params = study.best_params
    if 'feature_selection' in best_params:
        del best_params['feature_selection']
    if 'num_features' in best_params:
        del best_params['num_features']
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    # prepare X for use in the final fitted model
    # print('before final fit: ',X.shape, y.shape,X.columns)
    fitted_model = LogisticRegression(
        **best_params, **extra_params).fit(
        X[model_outputs['selected_features']],y)

    if data['verbose'] == 'on':

        # confusion matrix
        selected_features = model_outputs['selected_features']
        hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
        hfig.savefig("LogisticRegression_confusion_matrix.png", dpi=300)
        
        # ROC curve with AUC
        selected_features = model_outputs['selected_features']
        hfig = plot_roc_auc(fitted_model, X[selected_features], y)
        hfig.savefig("LogisticRegression_ROC_curve.png", dpi=300)
        
    # Goodness of fit statistics
    metrics = fitness_metrics_logistic(
        fitted_model, 
        X[model_outputs['selected_features']], y)
    stats = pd.DataFrame([metrics]).T
    stats.index.name = 'Statistic'
    stats.columns = ['LogisticRegression']
    model_outputs['metrics'] = metrics
    model_outputs['stats'] = stats
    model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])

    if data['verbose'] == 'on':
        print('')
        print("LogisticRegression goodness of fit to training data in model_outputs['stats']:")
        print('')
        print(model_outputs['stats'].to_markdown(index=True))
        print('')
    
    # Print the run time
    fit_time = time.time() - start_time

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')
        
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def model_agnostic(model, X_test, y_test,
    preprocess_result=None,
    selected_features=None,
    output_dir="agnostic_plots",
    show_skill=False,
    show_shap=False,
    show_perm=False,
    show_pdp=False,
    show_outputs=True):
    '''
    Model-agnostic analysis of a trained 
    Machine Learning regression or classification model

    Plots of the following model agnostics are provided:
    
    - Model skill metrics
    - Residual vs. Predicted Plot 
        (compares predictions vs residuals)
    - Prediction Error Plot 
        (compares predictions vs. actual values)
    - Permutation Importance 
        (model performance drop when a feature is shuffled)
    - Partial Dependence Plots (PDP) 
        (shows average effect of a feature on predictions)
    - ICE (Individual Conditional Expectation) 
        (how predictions change for one feature across samples)
    - SHAP Summary Plot (Beeswarm) 
        (feature importance and direction across all samples)
    - SHAP Bar Plot 
        (ranks features by mean absolute SHAP value)

    Args:
    model= fitted sklearn/XGB/etc regression or classification model object
    X_test = dataframe of the independent variables to test 
    y_test = series of the dependent variable to test (one column of data)
    preprocess_result = results of preprocess_train
    selected_features = optimized selected features
    output_dir = directory to store output plots
    show_skill = True or False (default) to show residual plots and skill metrics
    show_shap = True or False (default) to show SHAP beeswarm and bar plot    
    show_perm = True or False (default) to show Permutation Importance plot    
    show_pdp = True or False (default) to show PDP/ICE plots    
    show_outputs = True (default) or False to display the plots

    Returns: agnostic plots in output_dir 
    and dict of the following:
        metrics: model skill metrics
        shap_values: results of shap explainer(X_test)
        shap_importance: dataframe of 
            shap importance in order of
            np.abs(shap_values.values).mean(axis=0)
        shap_ordered_features: list of features 
            sorted in order or shap importance
        permutation_importance: dataframe of 
            permutation_importance result
        permutation_ordered_features: list of features 
            sorted in order or permutation importance
    '''

    from PyMLR import check_X_y, preprocess_test, test_model
    import shap
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.inspection import PartialDependenceDisplay, permutation_importance
    # from alepython import ale_plot
    import numpy as np
    import time

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    
    
    print('Performing model agnostic analysis, please wait...')

    if not show_skill and not show_shap and not show_perm and not show_pdp:
        print("No analysis is requested by default")
        print("Use any of the following keyword arguments:")
        print("- show_skill= True to analyze residual plots and regression metrics") 
        print("- show_shap= True to analyze SHAP beeswarm/bar plots") 
        print("- show_perm= True to analyze Permutation Importance") 
        print("- show_pdp= True to analyze PDP/ICE plots") 
    
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    # copy X_test and y_test to avoid altering the original
    X_test = X_test.copy()
    y_test = y_test.copy()

    # check X and y and put into dataframe if needed
    X_test, y_test = check_X_y(X_test, y_test)
    
    if preprocess_result != None:
        X_test_proc = preprocess_test(X_test, preprocess_result)
        continuous_cols = preprocess_result['continuous_cols']
    else:
        X_test_proc = X_test.copy()
        continuous_cols = None
    
    if selected_features != None:
        X_test_proc = X_test_proc[selected_features]
        if continuous_cols != None:
            selected_features_continuous = list(
                set(selected_features) & set(continuous_cols))
        else:
            selected_features_continuous = None
    else:
        if continuous_cols != None:
            selected_features_continuous = list(
                set(X_test_proc.columns) & set(continuous_cols))

    if preprocess_result != None and selected_features == None:
        selected_features= preprocess_result['columns_processed'] 

    if preprocess_result == None and selected_features == None:
        selected_features= X_test_proc.columns.to_list() 

    # initialize output dictionary
    output = {}
    
    # -------- Step 1: Residual Plot --------
    if show_skill:
        print('')
        print('Model skill metrics and residuals plot, please wait...')

        # Model skill metrics
        from PyMLR import fitness_metrics
        from sklearn.metrics import PredictionErrorDisplay
        metrics = fitness_metrics(
            model, 
            X_test_proc[selected_features], y_test)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['Regressor']

        # output = {}
        output['metrics'] = metrics
        y_pred = model.predict(X_test_proc[selected_features])
        output['y_pred'] = y_pred

        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y_test,
            y_pred,
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y_test,
            y_pred,
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={rmse:.3f})")
        plt.tight_layout()
        if show_outputs:
            print('')
            print("Model skill metrics:")
            print('')
            print(stats.to_markdown(index=True))
            print('')
            plt.show()
        plt.close()
            
    # -------- Step 2: SHAP Explainer (auto-detect) --------
    if show_shap:
        print('SHAP Beeswarm and Bar importance...')
        try:
            model_name = model.__class__.__name__.lower()
            if "linear" in model_name:
                explainer = shap.LinearExplainer(model, X_test_proc)
            else:
                explainer = shap.Explainer(model, X_test_proc)
        
            shap_values = explainer(X_test_proc)
            output['shap_values'] = shap_values
        
            # Beeswarm
            shap.plots.beeswarm(shap_values, show=False)  # show=False required to savefig
            plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=300, bbox_inches='tight')
            plt.close()
            if show_outputs:
                shap.plots.beeswarm(shap_values, show=True)  # show=True to display
                plt.close()
        
            # Bar plot for global feature importance
            shap.plots.bar(shap_values, show=False)  # show=False required to savefig
            plt.savefig(f"{output_dir}/shap_bar_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            if show_outputs:
                shap.plots.bar(shap_values, show=True)  # show=True to display
                plt.close()
                
            '''
            # Waterfall for first instance
            shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig(f"{output_dir}/shap_waterfall_sample0.png", dpi=300, bbox_inches='tight')
            plt.close()
            if show_outputs:
                shap.plots.waterfall(shap_values[0], show=True)  # show=True to display
                plt.close()
            '''
            
            # Compute mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            
            # Create a DataFrame for sorting
            importance_df = pd.DataFrame({
                'feature': shap_values.feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values(by='mean_abs_shap', ascending=False)
            
            # Extract ordered list of features
            shap_ordered_features = importance_df['feature'].tolist()
            output['shap_importance'] = importance_df
            output['shap_ordered_features'] = shap_ordered_features

        except Exception as e:
            print("SHAP skipped due to:", e)
            shap_ordered_features = None
    
    # -------- Step 3: Permutation Importance --------
    if show_perm:
        print('Permutation Importance...')
        try:
            result = permutation_importance(
                # model, X_test_proc, y_test, n_repeats=10, random_state=42
                model, X_test_proc, y_test, n_repeats=5, random_state=42
            )
            imp_series = pd.Series(result.importances_mean, index=X_test_proc.columns)
            imp_series.sort_values().plot.barh(title="Permutation Importance")
            plt.tight_layout()
            # output['permutation_importance_plot'] = result['hfig'] # change name of key        
            plt.savefig(f"{output_dir}/permutation_importance.png", dpi=300)
            if show_outputs:
                plt.show()
            plt.close()
            # Create a sorted DataFrame
            importance_df = pd.DataFrame({
                'feature': X_test_proc.columns,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values(by='importance_mean', ascending=False)        
            # Extract ordered list of features
            permutation_ordered_features = importance_df['feature'].tolist()        
            # save to result
            output['permutation_importance'] = importance_df
            output['permutation_ordered_features'] = permutation_ordered_features
        except Exception as e:
            print("Permutation Importance skipped due to:", e)
            permutation_ordered_features = None
        
    # -------- Step 4: PDP + ICE --------
    if show_pdp:
        print('PDP + ICE plots of each continuous features...')
        try:
            # features_for_pdp = optimum_selected_features[:2]  # choose top 2
            # if selected_features_continuous != None:
            #     features_for_pdp = selected_features_continuous
            if shap_ordered_features != None:
                for feat in shap_ordered_features:
                    if continuous_cols != None and feat in continuous_cols:
                        print('processing feature: ',feat)
                        PartialDependenceDisplay.from_estimator(
                            model, 
                            X_test_proc, 
                            [feat], 
                            kind='both', 
                            line_kw={"color": "black", "linewidth": 3},        # PDP line
                            ice_lines_kw={"color": "skyblue", "alpha": 0.3}  # ICE lines
                        )
                        plt.tight_layout()
                        plt.savefig(f"{output_dir}/pdp_ice_{feat}.png", dpi=300)
                        if show_outputs:
                            plt.show()
                        plt.close()
        
        except Exception as e:
            print("PDP skipped due to:", e)
   
    '''
    # -------- Step 5: ALE --------
    print('Step 5: ALE plots of top 2 features, please wait...')
    try:
        if ordered_features != None:
            features_for_ale = ordered_features[:2]  # choose top 2
            for feat in features_for_ale:
                ale_plot(model, X_test_proc, feat)
                plt.savefig(f"{output_dir}/ale_{feat}.png", dpi=300)
                plt.close()    
    except Exception as e:
        print("ALE skipped due to:", e)
    '''

    print(f"Interpretability plots saved to: {output_dir}")
    print('Done')
    fit_time = time.time() - start_time
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    return output

def linear(X, y, **kwargs):

    """
    Python function for sklearn LinearRegression 

    by
    Greg Pelletier
    gjpelletier@gmail.com
    01-Jul-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on' (default), 'off', or 1 (model skill and residuals plots) 
        fit_intercept= True,        # calculate intercept
        copy_X= True,               # True: X will be copied
        n_jobs= None,               # -1 to use all CPUs
        positive= False             # True forces coefficients to be positive
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        model_objects, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model
                - 'vif': Variance Inflation Factors of continuous features

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = svr(X, y)

    """

    from PyMLR import stats_given_y_pred, stats_given_model, detect_dummy_variables
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    import time
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import PredictionErrorDisplay
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # import xgboost as xgb

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'preprocess': True,         # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,  # dict of  the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder) 
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical columns)
                                    # - non_numeric_cats (non-numeric cats)
                                    # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,  # pre-optimized selected features
        'verbose': 'on',
        'fit_intercept': True,      # calculate intercept
        'copy_X': True,             # True: X will be copied
        'tol': 1e-6,                # precision of  solution (not used)
        'n_jobs': None,             # -1 to use all CPUs
        'positive': False           # True forces coefficients to be positive
        }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'], 
                'use_scaler': data['use_scaler'], 
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    # Suppress warnings
    warnings.filterwarnings('ignore')
    print('Fitting LinearRegression model, please wait ...')
    if data['verbose'] == 'on' or data['verbose'] == 1:
        print('')

    params = {
        'fit_intercept': data['fit_intercept'], # calculate intercept
        'copy_X':  data['copy_X'],              # True: X will be copied
        'n_jobs':  data['n_jobs'],              # -1 to use all CPUs
        'positive':  data['positive']           # True forces coefficients to be positive
    }
        
    fitted_model = LinearRegression(**params).fit(X,y)

    # check to see of the model has intercept and coefficients
    if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
            and fitted_model.coef_.size==len(X.columns)):
        intercept = fitted_model.intercept_
        coefficients = fitted_model.coef_
        # dataframe of model parameters, intercept and coefficients, including zero coefs
        n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
        popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
        for i in range(n_param):
            if i == 0:
                popt[0][i] = 'Intercept'
                popt[1][i] = fitted_model.intercept_
            else:
                popt[0][i] = X.columns[i-1]
                popt[1][i] = fitted_model.coef_[i-1]
        popt = pd.DataFrame(popt).T
        popt.columns = ['Feature', 'Parameter']
        # Table of intercept and coef
        popt_table = pd.DataFrame({
                "Feature": popt['Feature'],
                "Parameter": popt['Parameter']
            })
        popt_table.set_index('Feature',inplace=True)
        model_outputs['popt_table'] = popt_table
    
    # Goodness of fit statistics
    y_pred = fitted_model.predict(X)
    metrics = stats_given_y_pred(X, y, y_pred)
    keys_to_select = ['rsquared', 'adj_rsquared','n_samples',
        'df','dfn','Fstat','pvalue',
        'RMSE','log_likelihood','aic','bic']    
    metrics = {key: metrics[key] for key in keys_to_select if key in metrics}
    stats = pd.DataFrame([metrics])
    new_cols = ['r-squared','adjusted r-squared','n_samples',
        'df residuals','df model','F-statistic','Prob (F-statistic)',
        'RMSE','Log-Likelihood','AIC','BIC']
    stats.columns = new_cols
    stats = stats.T
    stats.index.name = 'Statistic'
    stats.columns = ['LinearRegression']
    model_outputs['metrics'] = metrics
    model_outputs['stats'] = stats
    model_outputs['y_pred'] = fitted_model.predict(X)

    # VIF
    '''
    model_ = fitted_model
    if data['preprocess']:
        selected_continuous_cols = [
            item for item in model_outputs['preprocess_result']['continuous_cols']
            if item in model_outputs['selected_features']]
        X_ = X[selected_continuous_cols].copy()
    else:
        X_ = X.copy()
    '''
    X_ = X.copy()
    X__ = sm.add_constant(X_)    # Add a constant for the intercept
    vif = pd.DataFrame()
    vif['Feature'] = X__.columns
    vif["VIF"] = [variance_inflation_factor(X__.values, i)
                        for i in range(len(X__.columns))]
    vif.set_index('Feature',inplace=True)
    vif.index.name = 'Feature'
    model_outputs['vif'] = vif

    if data['verbose'] == 'on' or data['verbose'] == 1:
        print("LinearRegression goodness of fit to training data in model_outputs['stats']:")
        print('')
        print(model_outputs['stats'].to_markdown(index=True))
        print('')

    if data['verbose'] == 'on' and hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
        print("Parameters of fitted model in model_outputs['popt']:")
        print('')
        print(model_outputs['popt_table'].to_markdown(index=True))
        print('')

    if data['verbose'] == 'on':
        print("Variance Inflation Factors in model_outputs['vif']:")
        print('')
        print(model_outputs['vif'].to_markdown(index=True))
        print('')

    # residual plot for training error
    if data['verbose'] == 'on' or data['verbose'] == 1:
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
        plt.tight_layout()
        # plt.show()
        plt.savefig("LinearRegression_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('')
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def linear_objective(trial, X, y, **kwargs):
    '''
    Optuna objective for optimizing XGBRegressor with optional feature selection.
    Supports selector choice, logs importances, and ensures reproducibility.
    '''

    from PyMLR import stats_given_y_pred
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import make_scorer

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)

    # make custom aic_scorer using AIC
    def aic_score(estimator, X, y):
        # Fit the model on this fold
        y_pred = estimator.predict(X)
        n = len(y)
        k = X.shape[1] + 1  # +1 for intercept        
        residual = y - y_pred
        # rss = np.sum(residual ** 2)
        rss = max(np.sum(residual ** 2), 1e-8)
        aic = n * np.log(rss / n) + 2 * k
        return -aic  # scikit-learn assumes higher is better
    aic_scorer = make_scorer(aic_score, greater_is_better=True)

    # make custom bic_scorer using BIC
    def bic_score(estimator, X, y):
        y_pred = estimator.predict(X)
        n = len(y)
        k = X.shape[1] + 1  # number of parameters (+1 for intercept)
        residual = y - y_pred
        # rss = np.sum(residual ** 2)
        rss = max(np.sum(residual ** 2), 1e-8)
        bic = n * np.log(rss / n) + k * np.log(n)
        return -bic  # Negative because scikit-learn assumes greater is better
    bic_scorer = make_scorer(bic_score, greater_is_better=True)

    # Define extra params
    extra_params = {
        'fit_intercept': kwargs['fit_intercept'], # calculate intercept
        'copy_X': kwargs['copy_X'],              # True: X will be copied
        'n_jobs': kwargs['n_jobs'],              # -1 to use all CPUs
        'positive': kwargs['positive']           # True forces coefficients to be positive
    }

    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        pipeline = Pipeline([
            ("feature_selector", selector),
            ("regressor", LinearRegression(**extra_params))
        ])
    else:
        pipeline = Pipeline([
            ("regressor", LinearRegression(**extra_params))
        ])
        num_features = None

    # Cross-validated scoring with RepeatedKFold
    cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)

    if kwargs['scorer'] == 'aic':
        scorer = aic_scorer
    elif kwargs['scorer'] == 'bic':
        scorer = bic_scorer
    else:
        scorer ="neg_root_mean_squared_error"
    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        # scoring=scorer                          # scorer still needs debugging as of 7/2/2025
        scoring="neg_root_mean_squared_error"     # hard-wire until scorer is debugged
    )    

    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("score", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean    

def linear_auto(X, y, **kwargs):

    """
    Autocalibration of LinearRegression 
    with optimized feature selection by optuna
    Preprocess with OneHotEncoder and StandardScaler
    Pipeline for feature selector and regressor

    by
    Greg Pelletier
    gjpelletier@gmail.com
    02-July-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        verbose= 'on' (default) or 'off'
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        gpu= True (default) or False to autodetect if the computer has a gpu and use it
        n_trials= 50,               # number of optuna trials
        n_splits= 5,                # number of splits for KFold CV
        pruning= False,             # prune poor optuna trials
        feature_selection= True,    # optuna feature selection

        random_state= 42,           # Random seed for reproducibility.
        fit_intercept= True,        # calculate intercept
        copy_X= True,               # True: X will be copied
        n_jobs= None,               # -1 to use all CPUs
        positive= False             # True forces coefficients to be positive
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns
                    - 'non_numeric_cats': non-numeric categorical columns
                    - 'continous_cols': continuous numerical columns                
                - 'optuna_study': optimzed optuna study object
                - 'optuna_model': optimzed optuna model object
                - 'best_trial': best trial from the optuna study
                - 'feature_selection' = option to select features (True, False)
                - 'selected_features' = selected features
                - 'best_params': best model hyper-parameters found by optuna
                - 'extra_params': other model options used to fit the model
                - 'metrics': dict of goodness of fit metrics for train data
                - 'stats': dataframe of goodness of fit metrics for train data
                - 'X_processed': pre-processed X with encoding and scaling
                - 'y_pred': best model predicted y

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_objects, model_outputs = linear_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, fitness_metrics, check_X_y
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna
    from sklearn.linear_model import LinearRegression
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'preprocess': True,                 # Apply OneHotEncoder and StandardScaler
        'preprocess_result': None,          # dict of  the following result from 
                                            # preprocess_train if available:         
                                            # - encoder          (OneHotEncoder) 
                                            # - scaler           (StandardScaler)
                                            # - categorical_cols (categorical columns)
                                            # - non_numeric_cats (non-numeric cats)
                                            # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'verbose': 'on',
        'gpu': True,                        # Autodetect to use gpu if present
        'n_splits': 5,                      # number of splits for KFold CV

        'pruning': False,                   # prune poor optuna trials
        'feature_selection': True,          # optuna feature selection

        'scorer': None,                     # 'aic', 'bic', or None. If None, then
                                            # 'neg_root_mean_squared_error' is used
        'random_state': 42,                 # Random seed for reproducibility.
        'fit_intercept': True,              # calculate intercept
        'copy_X': True,                     # True: X will be copied
        'tol': 1e-6,                        # precision of  solution (not used)
        'n_jobs': None,                     # -1 to use all CPUs
        'positive': False                   # True forces coefficients to be positive
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'], 
                'use_scaler': data['use_scaler'], 
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()
    
    extra_params = {
        'fit_intercept': data['fit_intercept'], # calculate intercept
        'copy_X':  data['copy_X'],              # True: X will be copied
        'n_jobs':  data['n_jobs'],              # -1 to use all CPUs
        'positive':  data['positive']           # True forces coefficients to be positive
    }
    
    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import linear_objective
    study.optimize(lambda trial: linear_objective(trial, X_opt, y, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial
        
    best_params = study.best_params
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    print('Fitting LinearRegression model with best parameters, please wait ...')
    fitted_model = LinearRegression(
        **extra_params).fit(
        X[model_outputs['selected_features']],y)
       
    # check to see of the model has intercept and coefficients
    if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
            and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
        intercept = fitted_model.intercept_
        coefficients = fitted_model.coef_
        # dataframe of model parameters, intercept and coefficients, including zero coefs
        n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
        popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
        for i in range(n_param):
            if i == 0:
                popt[0][i] = 'Intercept'
                popt[1][i] = fitted_model.intercept_
            else:
                popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                popt[1][i] = fitted_model.coef_[i-1]
        popt = pd.DataFrame(popt).T
        popt.columns = ['Feature', 'Parameter']
        # Table of intercept and coef
        popt_table = pd.DataFrame({
                "Feature": popt['Feature'],
                "Parameter": popt['Parameter']
            })
        popt_table.set_index('Feature',inplace=True)
        model_outputs['popt_table'] = popt_table
    
    # Goodness of fit statistics
    y_pred = fitted_model.predict(X[model_outputs['selected_features']])
    metrics = stats_given_y_pred(X[model_outputs['selected_features']], y, y_pred)
    keys_to_select = ['rsquared', 'adj_rsquared','n_samples',
        'df','dfn','Fstat','pvalue',
        'RMSE','log_likelihood','aic','bic']    
    metrics = {key: metrics[key] for key in keys_to_select if key in metrics}
    stats = pd.DataFrame([metrics])
    new_cols = ['r-squared','adjusted r-squared','n_samples',
        'df residuals','df model','F-statistic','Prob (F-statistic)',
        'RMSE','Log-Likelihood','AIC','BIC']
    stats.columns = new_cols
    stats = stats.T
    stats.index.name = 'Statistic'
    stats.columns = ['LinearRegression']
    model_outputs['metrics'] = metrics
    model_outputs['stats'] = stats
    model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])

    # VIF
    '''
    model_ = fitted_model
    if data['preprocess']:
        selected_continuous_cols = [
            item for item in model_outputs['preprocess_result']['continuous_cols']
            if item in model_outputs['selected_features']]
        X_ = X[selected_continuous_cols].copy()
    else:
        X_ = X.copy()
    '''
    X_ = X.copy()    
    X__ = sm.add_constant(X_)    # Add a constant for the intercept
    vif = pd.DataFrame()
    vif['Feature'] = X__.columns
    vif["VIF"] = [variance_inflation_factor(X__.values, i)
                        for i in range(len(X__.columns))]
    vif.set_index('Feature',inplace=True)
    vif.index.name = 'Feature'
    model_outputs['vif'] = vif

    if data['verbose'] == 'on' or data['verbose'] == 1:
        print("LinearRegression goodness of fit to training data in model_outputs['stats']:")
        print('')
        print(model_outputs['stats'].to_markdown(index=True))
        print('')

    if data['verbose'] == 'on' and hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
        print("Parameters of fitted model in model_outputs['popt']:")
        print('')
        print(model_outputs['popt_table'].to_markdown(index=True))
        print('')

    if data['verbose'] == 'on':
        print("Variance Inflation Factors in model_outputs['vif']:")
        print('')
        print(model_outputs['vif'].to_markdown(index=True))
        print('')

    # residual plot for training error
    if data['verbose'] == 'on' or data['verbose'] == 1:
        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="actual_vs_predicted",
            ax=axs[0]
        )
        axs[0].set_title("Actual vs. Predicted")
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=model_outputs['y_pred'],
            kind="residual_vs_predicted",
            ax=axs[1]
        )
        axs[1].set_title("Residuals vs. Predicted")
        fig.suptitle(
            f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
        plt.tight_layout()
        # plt.show()
        plt.savefig("LinearRegression_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def mlp(X, y, **kwargs):

    """
    Regression with sklearn MLPRegressor
    or
    Classification with sklearn MLPClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    15-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,                     # number of optuna trials
        classify= False,            # True to use RandomForestClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',                    # 'on' to display all 
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV

        # numerical core hyperparameters
        'hidden_layer_sizes': (100,),   # list of numbers of neurons in ith hidden layers
        'alpha': 0.0001,                # L2 regularization term
        'learning_rate_init': 0.001,    # initial learning rate

        # numerical solver-specific hyperparameters
        'batch_size': 'auto',         # batch size for stochastic optimizers
        'momentum': 0.9,              # for gradient descent update
        'power_t': 0.5,               # L2 regularization term
        'beta_1': 0.9,                # decay for first moment adam
        'beta_2': 0.999,              # decay rate for second moment adam
        'epsilon': 1e-8,              # numerical stability in adam
        'max_fun': 15000,             # used for solver lbfgs max number function calls

        # categorical hyperparameters optimized
        'activation': 'relu',         # hidden layer activation method
        'solver': 'adam',             # for weight optimization
        'learning_rate': 'constant',  # for weight updates
        'early_stopping': False,      # terminate when score not improving
        'nesterov': True,             # used for nesterovs_momentum if solver is sgd

        # extra_params 
        'random_state': 42,           # random seed for reproducibility
        'tol': 1e-4,
        'max_iter': 200,

                preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model, outputs = mlp(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from sklearn.neural_network import MLPRegressor, MLPClassifier

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # Use RandomForestClassifier if True
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                        # Autodetect to use gpu if present

        # numerical core hyperparameters
        'hidden_layer_sizes': (100,), # list of numbers of neurons in ith hidden layers
        'alpha': 0.0001,              # L2 regularization term
        'learning_rate_init': 0.001,  # initial learning rate

        # numerical solver-specific hyperparameters
        'batch_size': 'auto',         # batch size for stochastic optimizers
        'momentum': 0.9,              # for gradient descent update
        'power_t': 0.5,               # L2 regularization term
        'beta_1': 0.9,                # decay for first moment adam
        'beta_2': 0.999,              # decay rate for second moment adam
        'epsilon': 1e-8,              # numerical stability in adam
        'max_fun': 15000,             # used for solver lbfgs max number function calls

        # categorical hyperparameters optimized
        'activation': 'relu',         # hidden layer activation method
        'solver': 'adam',             # for weight optimization
        'learning_rate': 'constant',  # for weight updates
        'early_stopping': False,      # terminate when score not improving
        'nesterov': True,             # used for nesterovs_momentum if solver is sgd
        'nesterovs_momentum': True,   # placeholder for nesterovs_momentum if solver is sgd

        # extra_params 
        'random_state': 42,           # random seed for reproducibility
        'tol': 1e-4,
        'max_iter': 200,

    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign criterion depending on type of model
    if data['classify']:
        data['criterion'] = 'gini'
    else:
        data['criterion'] = 'squared_error'
    
    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    params = {
        'hidden_layer_sizes': data['hidden_layer_sizes'],
        'activation': data['activation'],
        'solver': data['solver'],
        'alpha': data['alpha'],
        'learning_rate': data['learning_rate'],
        'learning_rate_init': data['learning_rate_init'],
        'early_stopping': data['early_stopping'],
        'batch_size': data['batch_size'],
        'momentum': data['momentum'],
        'nesterovs_momentum': data['nesterovs_momentum'],
        'power_t': data['power_t'],
        'beta_1': data['beta_1'],
        'beta_2': data['beta_2'],
        'epsilon': data['epsilon'],
        'tol': data['tol'],
        'max_iter': data['max_iter'],
        'max_fun': data['max_fun']
    }    

    extra_params = {
        'verbose': False,                 
        'random_state': data['random_state'],                
    }

    model_outputs['best_params'] = params | extra_params   

    if data['classify']:
        print('Fitting MLPClassifier model, please wait ...')
        fitted_model = MLPClassifier(**params, **extra_params).fit(X,y)
    else:
        print('Fitting MLPRegressor model, please wait ...')
        fitted_model = MLPRegressor(**params, **extra_params).fit(X,y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            # selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X, y)
            hfig.savefig("MLPClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X, y)
            hfig.savefig("MLPClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X, y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['MLPClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)    
        if data['verbose'] == 'on':
            print('')
            print("MLPClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:            
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X.columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X.columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X, y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['MLPRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)
    
        if data['verbose'] == 'on':
            print('')
            print("MLPRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("MLPRegressor_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def mlp_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by optuna 
    to find the optimum hyper-parameters for 
    sklearn MLPRegressor or MLPClassifier
    '''
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from sklearn.neural_network import MLPRegressor, MLPClassifier

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)

    # 1. Sample architecture: depth and width
    n_layers = trial.suggest_int("n_layers", *kwargs['n_layers'])
    layer_units = []
    for i in range(n_layers):
        units = trial.suggest_int(f"n_units_l{i}", *kwargs['units'], log=True)
        layer_units.append(units)
    hidden_layer_sizes = tuple(layer_units)

    # 2. Sample core hyperparameters
    activation = trial.suggest_categorical("activation", kwargs['activation'])
    solver = trial.suggest_categorical("solver", kwargs['solver'])
    alpha = trial.suggest_loguniform("alpha", *kwargs['alpha'])
    learning_rate = trial.suggest_categorical("learning_rate", kwargs['learning_rate'])
    learning_rate_init = trial.suggest_loguniform("learning_rate_init", *kwargs['learning_rate_init'])
    early_stopping = trial.suggest_categorical("early_stopping", kwargs['early_stopping'])

    # 3. Solver-specific knobs
    batch_size = trial.suggest_int("batch_size", *kwargs['batch_size'], log=True)
    momentum = trial.suggest_float("momentum", *kwargs['momentum'])
    nesterov = trial.suggest_categorical("nesterovs_momentum", kwargs['nesterov'])
    power_t = trial.suggest_float("power_t", *kwargs['power_t'])
    beta_1 = trial.suggest_float("beta_1", *kwargs['beta_1'])
    beta_2 = trial.suggest_float("beta_2", *kwargs['beta_2'])
    epsilon = trial.suggest_loguniform("epsilon", *kwargs['epsilon'])
    max_fun = trial.suggest_int("max_fun", *kwargs['max_fun'])
    
    params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'alpha': alpha,
        'learning_rate': learning_rate,
        'learning_rate_init': learning_rate_init,
        'early_stopping': early_stopping,
        'batch_size': (batch_size if solver in ["sgd", "adam"] else "auto"),
        'momentum': (momentum if solver == "sgd" else 0.9),
        'nesterovs_momentum': (nesterov if solver == "sgd" else True),
        'power_t': (power_t if solver == "sgd" else 0.5),
        'beta_1': (beta_1 if solver == "adam" else 0.9),
        'beta_2': (beta_2 if solver == "adam" else 0.999),
        'epsilon': (epsilon if solver == "adam" else 1e-8),
        'tol': kwargs['tol'],
        'max_iter': kwargs['max_iter'],
        'max_fun': max_fun
    }    

    extra_params = {
        'verbose': False,                 
        'random_state': kwargs['random_state'],                
    }

    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        if kwargs['classify']:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", MLPClassifier(**params, **extra_params))
            ])
        else:        
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", MLPRegressor(**params, **extra_params))
            ])

    else:

        if kwargs['classify']:
            pipeline = Pipeline([
                ("regressor", MLPClassifier(**params, **extra_params))
            ])
        else:        

            pipeline = Pipeline([
                ("regressor", MLPRegressor(**params, **extra_params))
            ])

        num_features = None

    # Cross-validated scoring
    if kwargs['classify']:
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="f1_weighted"
            scoring=kwargs["scoring"]
        )
    else:
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )
    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("params", params)
    # trial.set_user_attr("hidden_layer_sizes", hidden_layer_sizes)
    # trial.set_user_attr("nesterovs_momentum", nesterov)
    # trial.set_user_attr("layer_units", layer_units)

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean
      
def mlp_auto(X, y, **kwargs):

    """
    Autocalibration of hyperparameters for 
    sklearn MLPRegressor or MLPClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    15-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,               # number of optuna trials
        classify= False,            # True for MLPClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',                    # 'on' to display all 
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV
        pruning= False,                   # prune poor optuna trials
        feature_selection= True,          # optuna feature selection

        # print trial progress
        'show_trial_progress': True,        # print each trial number and best cv score

        # numerical core hyperparameters optimized by optuna
        'n_layers': [1, 3],         # used to determine hiden_layer_sizes depth
        'units': [10, 200],         # used to determine hiden_layer_sizes width
        'alpha': [1e-6, 1e-1],      # L2 regularization term
        'learning_rate_init': [1e-5, 1e-1],  # initial learning rate

        # numerical solver-specific hyperparameters optimized by optuna
        'batch_size': [32, 1024],   # batch size for stochastic optimizers
        'momentum': [0.5, 0.99],    # for gradient descent update
        'power_t': [0.1, 0.9],      # L2 regularization term
        'beta_1': [0.8, 0.99],      # decay for first moment adam
        'beta_2': [0.9, 0.9999],    # decay rate for second moment adam
        'epsilon': [1e-8, 1e-4],    # numerical stability in adam
        'max_fun': [15000, 50000],  # used for solver lbfgs max number function calls

        # categorical hyperparameters optimized by optuna
        'activation': ["relu", "tanh", "logistic"],  # hidden layer activation method
        'solver': ["adam", "sgd", "lbfgs"],          # for weight optimization
        'learning_rate': ["constant", "invscaling", "adaptive"],  # for weight updates
        'early_stopping': [True, False],     # terminate when score not improving
        'nesterov': [True, False],           # used for nesterovs_momentum if solver is sgd

        # extra_params that are optional user-specified for optuna
        'random_state': 42,                 # random seed for reproducibility
        'tol': 1e-4,
        'max_iter': 200,

        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'optuna_study': optimzed optuna study object
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model, outputs = mlp_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # True for MLPClassifier
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,           # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                         # Autodetect to use gpu if present
        'n_splits': 5,                       # number of splits for KFold CV

        'pruning': False,                    # prune poor optuna trials
        'feature_selection': True,           # optuna feature selection
        'scoring': None,                     # cross_val_score scoring name
        'show_trial_progress': True,        # print each trial number and best cv score
        
        # numerical core hyperparameters optimized by optuna
        'n_layers': [1, 3],                  # used to determine hiden_layer_sizes depth
        'units': [10, 200],                  # used to determine hiden_layer_sizes width
        'alpha': [1e-6, 1e-1],               # L2 regularization term
        'learning_rate_init': [1e-5, 1e-1],  # initial learning rate

        # numerical solver-specific hyperparameters optimized by optuna
        'batch_size': [32, 1024],            # batch size for stochastic optimizers
        'momentum': [0.5, 0.99],             # for gradient descent update
        'power_t': [0.1, 0.9],               # L2 regularization term
        'beta_1': [0.8, 0.99],               # decay for first moment adam
        'beta_2': [0.9, 0.9999],             # decay rate for second moment adam
        'epsilon': [1e-8, 1e-4],             # numerical stability in adam
        'max_fun': [15000, 50000],           # used for solver lbfgs max number function calls

        # categorical hyperparameters optimized by optuna
        'activation': ["relu", "tanh", "logistic"],  # hidden layer activation method
        'solver': ["adam", "sgd", "lbfgs"],          # for weight optimization
        'learning_rate': ["constant", "invscaling", "adaptive"],  # for weight updates
        'early_stopping': [True, False],     # terminate when score not improving
        'nesterov': [True, False],           # used for nesterovs_momentum if solver is sgd

        # extra_params that are optional user-specified for optuna
        'random_state': 42,                 # random seed for reproducibility
        'tol': 1e-4,
        'max_iter': 200,
        
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    extra_params = {
        'verbose': False,                 
        'random_state': data['random_state'],                
    }

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import mlp_objective
    study.optimize(lambda trial: mlp_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial

    # get best_params from the optuna study
    '''
    best_params = study.best_params
    # add calculated optuna params to best_params
    best_params['hidden_layer_sizes'] = study.best_trial.user_attrs.get('hidden_layer_sizes')
    best_params['nesterovs_momentum'] = study.best_trial.user_attrs.get('nesterovs_momentum')
    '''
    best_params = study.best_trial.user_attrs.get('params')

    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']
    if 'n_layers' in best_params:
        del best_params['n_layers']
    if 'units' in best_params:
        del best_params['units']
    if 'nesterov' in best_params:
        del best_params['nesterov']
    prefix = 'n_units_l'
    matching_keys = [key for key in best_params if key.startswith(prefix)]
    for key in matching_keys:
        del best_params[key]
    
    if data['classify']:
        print('Fitting MLPClassifier model with best parameters, please wait ...')    
        fitted_model = MLPClassifier(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
    else:    
        print('Fitting MLPRegressor model with best parameters, please wait ...')    
        fitted_model = MLPRegressor(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("MLPClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("MLPClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['MLPClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("MLPClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['MLPRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("MLPRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("MLPRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def tree(X, y, **kwargs):

    """
    Regression with sklearn DecisionTreeRegressor
    or
    Classification with sklearn DecisionTreeClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    15-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,               # number of optuna trials
        classify= False,            # True to use RandomForestClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',                    # 'on' to display all 
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV

        # params 
        'criterion': 'squared_error', # default if classify=False
        'criterion': 'gini',          # default if classify=True
        'max_depth': None,            # max depth of a tree
        'min_samples_split': 2,       # min samples to split internal node
        'min_samples_leaf': 1,        # min samples to be at a leaf node
        'max_features': None,         # number of features to consider 
                                      # when looking for the best split
        'splitter': "best",           # strategy to split at each node
        'ccp_alpha': 0.0,             # parameter for 
                                      # Minimum Cost-Complexity Pruning
        'max_leaf_nodes': None,       # max number of leaf nodes

        # extra_params that are optional user-specified
        'random_state': 42,           # random seed for reproducibility

        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model, outputs = tree(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier 

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # Use RandomForestClassifier if True
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                  # Autodetect to use gpu if present
        'n_splits': 5,                # number of splits for KFold CV

        # params 
        'criterion': 'squared_error', # default if classify=False
        'max_depth': None,            # max depth of a tree
        'min_samples_split': 2,       # min samples to split internal node
        'min_samples_leaf': 1,        # min samples to be at a leaf node
        'max_features': None,         # number of features to consider 
                                      # when looking for the best split
        'splitter': "best",           # strategy to split at each node
        'ccp_alpha': 0.0,             # parameter for 
                                      # Minimum Cost-Complexity Pruning
        'max_leaf_nodes': None,       # max number of leaf nodes

        # extra_params that are optional user-specified
        'random_state': 42,           # random seed for reproducibility

    }

    # change default criterion to 'gini' if input of classify=True and no input of criterion
    if 'classify' in kwargs and kwargs['classify'] and not 'criterion' in kwargs:
        defaults['criterion'] = 'gini'
        
    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # check for valid input of criterion
    if data['classify']:
        valid_strings = {"gini", "entropy", "log_loss"}
        if not data.get("criterion") in valid_strings:
            invalid = data['criterion']
            data['criterion'] = 'gini'
            print(f"Warning: Invalid input of criterion {invalid} was changed to {data['criterion']}")
    else:
        valid_strings = {"squared_error", "absolute_error", "friedman_mse", "poisson"}
        if not data.get("criterion") in valid_strings:
            invalid = data['criterion']
            data['criterion'] = 'squared_error'
            print(f"Warning: Invalid input of criterion {invalid} was changed to {data['criterion']}")
            
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    params = {
        'max_depth': data['max_depth'],                 
        'min_samples_split': data['min_samples_split'],            
        'min_samples_leaf': data['min_samples_leaf'],             
        'max_features': data['max_features'],             
        'splitter': data['splitter'],             
        'ccp_alpha': data['ccp_alpha'],                 
        'max_leaf_nodes': data['max_leaf_nodes'],           
    }

    extra_params = {
        'random_state': data['random_state'],                
    }

    if data['classify']:
        print('Fitting DecisionTreeClassifier model, please wait ...')
        fitted_model = DecisionTreeClassifier(**params, **extra_params).fit(X,y)
    else:
        print('Fitting DecisionTreeRegressor model, please wait ...')
        fitted_model = DecisionTreeRegressor(**params, **extra_params).fit(X,y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            # selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X, y)
            hfig.savefig("DecisionTreeClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X, y)
            hfig.savefig("DecisionTreeClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X, y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['DecisionTreeClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)    
        if data['verbose'] == 'on':
            print('')
            print("DecisionTreeClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:            
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X.columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X.columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X, y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['DecisionTreeRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)
    
        if data['verbose'] == 'on':
            print('')
            print("DecisionTreeRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("DecisionTreeRegressor_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def tree_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by optuna 
    to find the optimum hyper-parameters for 
    sklearn DecisionTreeRegressor or DecisionTreeClassifier
    '''
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier 

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)
    
    params = {
        "max_depth": trial.suggest_int("max_depth",
            *kwargs['max_depth']),
        "min_samples_split": trial.suggest_int("min_samples_split",
            *kwargs['min_samples_split'], log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf",
            *kwargs['min_samples_leaf'], log=True),
        "max_features": trial.suggest_float("max_features",
            *kwargs['max_features']),
        'splitter': trial.suggest_categorical("splitter", kwargs['splitter']),
        "ccp_alpha": trial.suggest_float("ccp_alpha",
            *kwargs['ccp_alpha'], log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes",
            *kwargs['max_leaf_nodes'], log=True),
    }

    if kwargs['classify']:
        params['criterion'] = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    else:
        params['criterion'] = trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"])

    extra_params = {
        'random_state': kwargs['random_state'],                
    }

    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        if kwargs['classify']:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", DecisionTreeClassifier(**params, **extra_params))
            ])
        else:        
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", DecisionTreeRegressor(**params, **extra_params))
            ])

    else:

        if kwargs['classify']:
            pipeline = Pipeline([
                ("regressor", DecisionTreeClassifier(**params, **extra_params))
            ])
        else:        

            pipeline = Pipeline([
                ("regressor", DecisionTreeRegressor(**params, **extra_params))
            ])

        num_features = None

    # Cross-validated scoring
    if kwargs['classify']:
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="f1_weighted"        
            scoring=kwargs["scoring"]
        )
    else:
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )
    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean
  
def tree_auto(X, y, **kwargs):

    """
    Autocalibration of hyperparameters for 
    sklearn DecisionTreeRegressor or DecisionTreeClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    18-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,                     # number of optuna trials
        classify= False,            # True for RandomForestClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',                    # 'on' to display all 
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV
        pruning= False,                   # prune poor optuna trials
        feature_selection= True,          # optuna feature selection

        # params that are optimized by optuna
        'max_depth': [2, 30],               # max depth of a tree
        'min_samples_split': [2, 20],       # min samples to split internal node
        'min_samples_leaf': [1, 20],        # min samples to be at a leaf node
        'max_features': [0.1, 1.0],         # number of features to consider 
                                            # when looking for the best split
        'splitter': ["best", "random"],     # strategy to choose the split at each node
        'ccp_alpha': [0.0001, 0.1],         # parameter for 
                                            # Minimum Cost-Complexity Pruning
        'max_leaf_nodes': [10, 100],        # max number of leaf nodes

        # extra_params that are optional user-specified
        'random_state': 42,                 # random seed for reproducibility

        # extra_params that are optional user-specified
        random_state= 42,                 # random seed for reproducibility

        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'optuna_study': optimzed optuna study object
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_tree, outputs_tree = tree_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier 
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # True for RandomForestClassifier
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                        # Autodetect to use gpu if present
        'n_splits': 5,                      # number of splits for KFold CV

        'pruning': False,                   # prune poor optuna trials
        'feature_selection': True,          # optuna feature selection
        'scoring': None,                     # cross_val_score scoring name
        'show_trial_progress': True,         # print trial numbers during execution
        
        # params that are optimized by optuna
        'max_depth': [2, 30],               # max depth of a tree
        'min_samples_split': [2, 20],       # min samples to split internal node
        'min_samples_leaf': [1, 20],        # min samples to be at a leaf node
        'max_features': [0.1, 1.0],         # number of features to consider 
                                            # when looking for the best split
        'splitter': ["best", "random"],     # strategy to choose the split at each node
        'ccp_alpha': [0.0001, 0.1],         # parameter for 
                                            # Minimum Cost-Complexity Pruning
        'max_leaf_nodes': [10, 100],        # max number of leaf nodes

        # extra_params that are optional user-specified
        'random_state': 42,                 # random seed for reproducibility
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    extra_params = {
        'random_state': data['random_state'],                
    }

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import tree_objective
    study.optimize(lambda trial: tree_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial
        
    best_params = study.best_params
    model_outputs['best_params'] = best_params
    model_outputs['extra_params'] = extra_params

    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']
    
    if data['classify']:
        print('Fitting DecisionTreeClassifier model with best parameters, please wait ...')    
        fitted_model = DecisionTreeClassifier(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
    else:    
        print('Fitting DecisionTreeRegressor model with best parameters, please wait ...')    
        fitted_model = DecisionTreeRegressor(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("DecisionTreeClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("DecisionTreeClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['DecisionTreeClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("DecisionTreeClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['DecisionTreeRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("DecisionTreeRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("DecisionTreeRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def ada(X, y, **kwargs):

    """
    Regression with sklearn AdaBoostRegressor
    or
    Classification with sklearn AdaBoostClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    15-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,               # number of optuna trials
        classify= False,            # True to use RandomForestClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',                    # 'on' to display all 
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV

        # params for AdaBoost optimized by optuna
        'n_estimators': 50,
        'learning_rate': 1.0,
        'loss': "linear",  # loss fn for classification
        
        # params for base_estimator (DecisionTree)
        'max_leaf_nodes': None,       # max number of leaf nodes
        'max_depth': None,            # max depth of a tree
        'min_samples_split': 2,       # min samples to split internal node
        'min_samples_leaf': 1,        # min samples to be at a leaf node
        'max_features': None,         # number of features to consider 
                                      # when looking for the best split

        # extra_params that are optional user-specified
        'random_state': 42,                 # random seed for reproducibility

        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model, outputs = ada(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier 

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # Use RandomForestClassifier if True
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                  # Autodetect to use gpu if present
        'n_splits': 5,                # number of splits for KFold CV

        # params for AdaBoost optimized by optuna
        'estimator': None,
        'n_estimators': 50,
        'learning_rate': 1.0,
        'loss': "linear",             # loss fn for regressor
        
        # params for base_estimator (DecisionTree)
        'max_leaf_nodes': None,       # max number of leaf nodes
        'max_depth': None,            # max depth of a tree
        'min_samples_split': 2,       # min samples to split internal node
        'min_samples_leaf': 1,        # min samples to be at a leaf node
        'max_features': None,         # number of features to consider 
                                      # when looking for the best split

        # extra_params that are optional user-specified
        'random_state': 42,                 # random seed for reproducibility

    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    if data['classify']:
        if data['estimator'] == None:
            data['estimator'] = DecisionTreeClassifier(
                max_depth= data['max_depth'], 
                min_samples_split= data['min_samples_split'], 
                min_samples_leaf= data['min_samples_leaf'], 
                max_features= data['max_features'], 
                max_leaf_nodes= data['max_leaf_nodes'], 
                random_state=data['random_state'])
    else:
        if data['estimator'] == None:
            data['estimator'] = DecisionTreeRegressor(
                max_depth= data['max_depth'], 
                min_samples_split= data['min_samples_split'], 
                min_samples_leaf= data['min_samples_leaf'], 
                max_features= data['max_features'], 
                max_leaf_nodes= data['max_leaf_nodes'], 
                random_state=data['random_state'])
            
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to prevent altering original
    X = X.copy()
    y = y.copy()
    
    # QC check X and y
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # Set start time for calculating run time
    start_time = time.time()

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    if data['selected_features'] == None:
        data['selected_features'] = X.columns.to_list()
    else:
        X = X[data['selected_features']]

    # save preprocess outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['selected_features'] = data['selected_features']
    model_outputs['X_processed'] = X.copy()

    # Suppress warnings
    warnings.filterwarnings('ignore')


    params = {
        "n_estimators": data["n_estimators"],
        'learning_rate': data["learning_rate"], 
        'estimator': data["estimator"], 
    }

    if not data['classify']:
        params['loss'] = data['loss']

    if not data['classify']:
        params['loss'] = data['loss']
        
    extra_params = {
        'random_state': data['random_state'],                
    }

    if data['classify']:
        print('Fitting AdaBoostClassifier model, please wait ...')
        fitted_model = AdaBoostClassifier(**params, **extra_params).fit(X,y)
    else:
        print('params:\n',params)
        print('extra_params:\n',extra_params)
        print('Fitting AdaBoostRegressor model, please wait ...')
        fitted_model = AdaBoostRegressor(**params, **extra_params).fit(X,y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            # selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X, y)
            hfig.savefig("AdaBoostClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X, y)
            hfig.savefig("AdaBoostClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X, y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['AdaBoostClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)    
        if data['verbose'] == 'on':
            print('')
            print("AdaBoostClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:            
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X.columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X.columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X, y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['AdaBoostRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X)
    
        if data['verbose'] == 'on':
            print('')
            print("AdaBoostRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("AdaBoostRegressor_predictions.png", dpi=300)
    
    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs
  
def ada_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by optuna 
    to find the optimum hyper-parameters for 
    sklearn AdaBoostRegressor or AdaBoostClassifier
    '''
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier 

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')

    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)
    
    # AdaBoost params
    params = {
        # AdaBoost params
        "n_estimators": trial.suggest_int("n_estimators",
            *kwargs['n_estimators']),
        'learning_rate': trial.suggest_float("learning_rate", 
            *kwargs['learning_rate'], log=True),        
    }

    # DecisionTree params for base_estimator
    params_tree = {
        "max_depth": trial.suggest_int("max_depth",
            *kwargs['max_depth']),
        "min_samples_split": trial.suggest_int("min_samples_split",
            *kwargs['min_samples_split'], log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf",
            *kwargs['min_samples_leaf'], log=True),
        "max_features": trial.suggest_float("max_features",
            *kwargs['max_features']),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes",
            *kwargs['max_leaf_nodes'], log=True),
    }
    
    # Use DecisionTree as estimator for AdaBoost
    if kwargs['classify']:
        params['estimator'] = DecisionTreeClassifier(
            max_depth= params_tree['max_depth'], 
            min_samples_split= params_tree['min_samples_split'], 
            min_samples_leaf= params_tree['min_samples_leaf'], 
            max_features= params_tree['max_features'], 
            max_leaf_nodes= params_tree['max_leaf_nodes'], 
            random_state=seed)
    else:
        params['estimator'] = DecisionTreeRegressor(
            max_depth= params_tree['max_depth'], 
            min_samples_split= params_tree['min_samples_split'], 
            min_samples_leaf= params_tree['min_samples_leaf'], 
            max_features= params_tree['max_features'], 
            max_leaf_nodes= params_tree['max_leaf_nodes'], 
            random_state=seed)
        params['loss'] = trial.suggest_categorical("loss", kwargs['loss'])

    extra_params = {
        'random_state': kwargs['random_state'],                
    }

    # Feature selection
    if kwargs.get("feature_selection", True):
        num_features = trial.suggest_int("num_features", max(5, X.shape[1] // 10), X.shape[1])
        selector_type = trial.suggest_categorical("selector_type", ["mutual_info", "f_regression"])

        if selector_type == "mutual_info":
            score_func = lambda X_, y_: mutual_info_regression(X_, y_, random_state=seed)
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=num_features)

        if kwargs['classify']:
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", AdaBoostClassifier(**params, **extra_params))
            ])
        else:        
            pipeline = Pipeline([
                ("feature_selector", selector),
                ("regressor", AdaaBoostRegressor(**params, **extra_params))
            ])

    else:

        if kwargs['classify']:
            pipeline = Pipeline([
                ("regressor", AdaBoostClassifier(**params, **extra_params))
            ])
        else:        

            pipeline = Pipeline([
                ("regressor", AdaBoostRegressor(**params, **extra_params))
            ])

        num_features = None

    # Cross-validated scoring
    if kwargs['classify']:
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="f1_weighted",
            scoring=kwargs["scoring"],
            n_jobs=kwargs['n_jobs']
        )
    else:
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error",
            scoring=kwargs["scoring"],
            n_jobs=kwargs['n_jobs']
        )
    score_mean = np.mean(scores)

    # Fit on full data to extract feature info
    pipeline.fit(X, y)

    if kwargs.get("feature_selection", True):
        selector_step = pipeline.named_steps["feature_selector"]
        selected_indices = selector_step.get_support(indices=True)
        selected_features = np.array(kwargs["feature_names"])[selected_indices].tolist()
    else:
        selected_features = kwargs["feature_names"]

    # Log feature importances and metadata
    model_step = pipeline.named_steps["regressor"]
    importances = getattr(model_step, "feature_importances_", None)
    if importances is not None:
        trial.set_user_attr("feature_importances", importances.tolist())

    trial.set_user_attr("params", params)
    trial.set_user_attr("params_tree", params_tree)

    trial.set_user_attr("model", pipeline)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
    trial.set_user_attr("selected_features", selected_features)
    trial.set_user_attr("selector_type", selector_type if kwargs.get("feature_selection", True) else None)

    return score_mean

def ada_auto(X, y, **kwargs):

    """
    Autocalibration of hyperparameters for 
    sklearn AdaBoostRegressor or AdaBoostClassifier

    by
    Greg Pelletier
    gjpelletier@gmail.com
    18-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,                     # number of optuna trials
        classify= False,            # True for RandomForestClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',                    # 'on' to display all 
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV
        pruning= False,                   # prune poor optuna trials
        feature_selection= True,          # optuna feature selection
        njobs= -1,                  # use -1 for all cpu cores with cross_val_score

        # params for AdaBoost optimized by optuna
        'n_estimators': [50, 500],
        'learning_rate': [0.01, 1.0]
        'loss': ["linear", "square", "exponential"],  # loss fn for regressor
        
        # params for base_estimator (DecisionTree)
        'max_depth': [2, 30],               # max depth of a tree
        'min_samples_split': [2, 20],       # min samples to split internal node
        'min_samples_leaf': [1, 20],        # min samples to be at a leaf node
        'max_features': [0.1, 1.0],         # number of features to consider 
                                            # when looking for the best split
        'max_leaf_nodes': [10, 100],        # max number of leaf nodes

        # extra_params that are optional user-specified
        'random_state': 42,                 # random seed for reproducibility

        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'optuna_study': optimzed optuna study object
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model_ada, outputs_ada = ada_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier 
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # True for RandomForestClassifier
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'selected_features': None,    # pre-optimized selected features
        'verbose': 'on',
        'gpu': True,                        # Autodetect to use gpu if present
        'n_splits': 5,                      # number of splits for KFold CV

        'pruning': False,                   # prune poor optuna trials
        'feature_selection': True,          # optuna feature selection
        'scoring': None,                     # cross_val_score scoring name
        'n_jobs': -1,                 # -1 to use all CPU cores with cross_val_score
        'show_trial_progress': True,         # print trial numbers during execution
        
        # params for AdaBoost optimized by optuna
        'n_estimators': [50, 500],
        'learning_rate': [0.01, 1.0],
        'loss': ["linear", "square", "exponential"],  # loss fn for regressor
        
        # params for base_estimator (DecisionTree)
        'max_depth': [2, 30],               # max depth of a tree
        'min_samples_split': [2, 20],       # min samples to split internal node
        'min_samples_leaf': [1, 20],        # min samples to be at a leaf node
        'max_features': [0.1, 1.0],         # number of features to consider 
                                            # when looking for the best split
        'max_leaf_nodes': [10, 100],        # max number of leaf nodes

        # extra_params that are optional user-specified
        'random_state': 42,                 # random seed for reproducibility
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    extra_params = {
        'random_state': data['random_state'],                
    }

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import ada_objective
    study.optimize(lambda trial: ada_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['optuna_model'] = study.best_trial.user_attrs.get('model')
    model_outputs['feature_selection'] = data['feature_selection']
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial
        
    best_params = study.best_trial.user_attrs.get('params')
    best_params_tree = study.best_trial.user_attrs.get('params_tree')
    model_outputs['best_params'] = best_params
    model_outputs['best_params_tree'] = best_params_tree
    model_outputs['extra_params'] = extra_params

    if data['classify']:
        print('Fitting AdaBoostClassifier model with best parameters, please wait ...')    
        fitted_model = AdaBoostClassifier(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)
    else:    
        print('Fitting AdaBoostRegressor model with best parameters, please wait ...')    
        fitted_model = AdaBoostRegressor(**best_params, **extra_params).fit(
            X[model_outputs['selected_features']],y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("AdaBoostClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("AdaBoostClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['AdaBoostClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("AdaBoostClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['AdaBoostRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("AdaBoostRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("AdaBoostRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def xgbmlp_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by optuna 
    to find the optimum hyper-parameters for 
    hybrid model similar to Li et al 2022
    Stage 1: XGBoost for feature selection (XGBClassifier or XGBRegressor)
    Stage 2: sklnear MLP for classification (MLPClassifier or MLPRegressor)
    '''
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.inspection import permutation_importance

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')
    
    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)
    
    # XGBoost params
    params_stage1 = {
        "learning_rate": trial.suggest_float("xgb_learning_rate", *kwargs["xgb_learning_rate"], log=True),
        "max_depth": trial.suggest_int("max_depth", *kwargs["max_depth"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *kwargs["min_child_weight"]),
        "subsample": trial.suggest_float("subsample", *kwargs["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *kwargs["colsample_bytree"]),
        "gamma": trial.suggest_float("gamma", *kwargs["gamma"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *kwargs["reg_lambda"], log=True),
        "alpha": trial.suggest_float("alpha", *kwargs["alpha"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *kwargs["n_estimators"]),
        'random_state': kwargs['random_state'],                
        'device': kwargs['device'],                 

        # xgb extra specified args
        "verbosity": kwargs["verbosity"],
        "objective": kwargs["objective"],
        "booster": kwargs["booster"],
        "tree_method": kwargs["tree_method"],
        "nthread": kwargs["nthread"],
        "colsample_bylevel": kwargs["colsample_bylevel"],
        "colsample_bynode": kwargs["colsample_bynode"],
        "base_score": kwargs["base_score"],
        "missing": kwargs["missing"],
        "importance_type": kwargs["importance_type"],
        "enable_categorical": kwargs["enable_categorical"],

    }

    # Fit XGBoost for feature selection
    if kwargs['classify']:
        model_stage1 = XGBClassifier(**params_stage1)
    else:
        model_stage1 = XGBRegressor(**params_stage1)
    model_stage1.fit(X, y)
    
    # absolute value of feature importances (not used for feature selection)
    feature_importances_raw = np.abs(model_stage1.feature_importances_)
    feature_importances_norm = feature_importances_raw / feature_importances_raw.sum()

    # absolute value of mean permutation importances
    if kwargs['use_permutation']:
        result = permutation_importance(model_stage1, X, y, n_repeats=5, random_state=seed)
        permutation_importances_raw = np.abs(result.importances_mean)
        permutation_importances_norm = permutation_importances_raw / permutation_importances_raw.sum()

    # feature selection
    threshold = trial.suggest_float("feature_threshold", *kwargs["feature_threshold"], log=True) 
    if kwargs['use_permutation']:
        if kwargs['use_normalized']:
            selected_idx = np.where(permutation_importances_norm > threshold)[0]
        else:
            selected_idx = np.where(permutation_importances_raw > threshold)[0]
    else:
        if kwargs['use_normalized']:
            selected_idx = np.where(feature_importances_norm > threshold)[0]
        else:
            selected_idx = np.where(feature_importances_raw > threshold)[0]
        
    # heavily penalize trials with no selected features
    if len(selected_idx) == 0:
        trial.set_user_attr("selected_features", [])
        return -1e6

    # selected_features
    feature_names = kwargs['feature_names']
    selected_features = [feature_names[i] for i in selected_idx]
    trial.set_user_attr("selected_features", selected_features)

    # dictionary to log results of stage 1
    if kwargs['use_permutation']:
        results_stage1 = {
            "selected_idx": selected_idx,
            "selected_features": selected_features,
            'feature_names': feature_names,
            'use_normalized': kwargs['use_normalized'],
            'use_permutation': kwargs['use_permutation'],
            'threshold': threshold,
            "feature_importances_raw": feature_importances_raw,
            "feature_importances_norm": feature_importances_norm,
            "permutation_importances_raw": permutation_importances_raw,
            "permutation_importances_norm": permutation_importances_norm,
        }
    else:
        results_stage1 = {
            "selected_idx": selected_idx,
            "selected_features": selected_features,
            'feature_names': feature_names,
            'use_normalized': kwargs['use_normalized'],
            'use_permutation': kwargs['use_permutation'],
            'threshold': threshold,
            "feature_importances_raw": feature_importances_raw,
            "feature_importances_norm": feature_importances_norm,
        }
    trial.set_user_attr("results_stage1", results_stage1)

    # Subset data
    X_selected = X[selected_features]
    
    # 1. MLP Sample architecture: depth and width
    n_layers = trial.suggest_int("n_layers", *kwargs['n_layers'])
    layer_units = []
    for i in range(n_layers):
        units = trial.suggest_int(f"n_units_l{i}", *kwargs['units'], log=True)
        layer_units.append(units)
    hidden_layer_sizes = tuple(layer_units)

    # 2. MLP Sample core hyperparameters
    activation = trial.suggest_categorical("activation", kwargs['activation'])
    solver = trial.suggest_categorical("solver", kwargs['solver'])
    alpha = trial.suggest_loguniform("alpha", *kwargs['alpha'])
    mlp_learning_rate = trial.suggest_categorical("mlp_learning_rate", kwargs['mlp_learning_rate'])
    learning_rate_init = trial.suggest_loguniform("learning_rate_init", *kwargs['learning_rate_init'])
    early_stopping = trial.suggest_categorical("early_stopping", kwargs['early_stopping'])

    # 3. MLP Solver-specific knobs
    batch_size = trial.suggest_int("batch_size", *kwargs['batch_size'], log=True)
    momentum = trial.suggest_float("momentum", *kwargs['momentum'])
    nesterov = trial.suggest_categorical("nesterovs_momentum", kwargs['nesterov'])
    power_t = trial.suggest_float("power_t", *kwargs['power_t'])
    beta_1 = trial.suggest_float("beta_1", *kwargs['beta_1'])
    beta_2 = trial.suggest_float("beta_2", *kwargs['beta_2'])
    epsilon = trial.suggest_loguniform("epsilon", *kwargs['epsilon'])
    max_fun = trial.suggest_int("max_fun", *kwargs['max_fun'])
    
    # MLP params for base_estimator
    params_stage2 = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'alpha': alpha,
        'learning_rate': mlp_learning_rate,
        'learning_rate_init': learning_rate_init,
        'early_stopping': early_stopping,
        'batch_size': (batch_size if solver in ["sgd", "adam"] else "auto"),
        'momentum': (momentum if solver == "sgd" else 0.9),
        'nesterovs_momentum': (nesterov if solver == "sgd" else True),
        'power_t': (power_t if solver == "sgd" else 0.5),
        'beta_1': (beta_1 if solver == "adam" else 0.9),
        'beta_2': (beta_2 if solver == "adam" else 0.999),
        'epsilon': (epsilon if solver == "adam" else 1e-8),
        'tol': kwargs['tol'],
        'max_iter': kwargs['max_iter'],
        'max_fun': max_fun,
        'random_state': kwargs['random_state'],                
        'verbose': False,                 
    }

    # Fit MLP for classification or regression
    if kwargs['classify']:
        model_stage2 = MLPClassifier(**params_stage2)
        # Cross-validated scoring
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            model_stage2, X_selected, y,
            cv=cv,
            # scoring="f1_weighted"
            scoring=kwargs["scoring"]
        )
    else:
        model_stage2 = MLPRegressor(**params_stage2)
        # Cross-validated scoring
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            model_stage2, X_selected, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )
    score_mean = np.mean(scores)

    # log params, models, and score
    trial.set_user_attr("params_stage1", params_stage1)
    trial.set_user_attr("model_stage1", model_stage1)
    trial.set_user_attr("params_stage2", params_stage2)
    trial.set_user_attr("model_stage2", model_stage2)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
        
    return score_mean
    
def xgbmlp_auto(X, y, **kwargs):

    """
    Autocalibration of hyperparameters for a hybrid model 
    similar to Li et al 2022 (https://doi.org/10.3390/forecast4010011)
    Stage 1: XGBoost for feature selection 
    (XGBClassifier or XGBRegressor)
    Stage 2: sklnear MLP for classification 
    (MLPClassifier or MLPRegressor)

    by
    Greg Pelletier
    gjpelletier@gmail.com
    20-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,               # number of optuna trials
        classify= False,            # True for MLPClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',                    # 'on' to display all 
        gpu= True,                        # Autodetect to use gpu if present
        n_splits= 5,                      # number of splits for KFold CV
        pruning= False,                   # prune poor optuna trials

        # random seed for all functions 
        'random_state': 42,                 # random seed for reproducibility

        # objective function options
        'show_trial_progress': True,        # print trial numbers during execution
        'use_permutation': False,    # True to use abs permutation importances for RFE
                                     # False to use abs .feature_importances_ for RFE
        'use_normalized': True,      # True to normalize the abs importances for RFE
                                     # False to use raw abs importances

        # xgb params that are optimized by optuna
        'feature_threshold': [0.001, 0.1],   # threshold for feature_importance
        'xgb_learning_rate': [1e-4, 1.0],       # Step size shrinkage (also called eta).
        'max_depth': [3, 12],               # Max depth of a tree.
        'min_child_weight': [1, 10],  # Min sum of instance weight (hessian) in a child.
        'subsample': [0.5, 1],        # Fraction of samples used for training each tree.
        'colsample_bytree': [0.5, 1],   # Fraction of features used for each tree.
        'gamma': [1e-8, 10.0],          # Minimum loss reduction to make a split.
        'reg_lambda': [1e-8, 10.0],     # L2 regularization term on weights.
        'alpha': [1e-8, 10.0],          # L1 regularization term on weights.
        'n_estimators': [50, 500],      # Number of boosting rounds (trees).

        # xgb extra_params that are optional user-specified
        'verbosity': 1,           # Verbosity of output (0=silent, 1=warnings, 2=info).
        'booster': "gbtree",          # Type of booster ('gbtree','gblinear','dart').
        'tree_method': "auto",        # Tree construction algorithm.
        'nthread': -1,                # Number of parallel threads.
        'colsample_bylevel': 1,       # Fraction of features used per tree level.
        'colsample_bynode': 1,        # Fraction of features used per tree node.
        'scale_pos_weight': 1,        # Balancing of positive and negative weights.
        'base_score': 0.5,            # Initial prediction score (global bias).
        'missing': np.nan,            # Value in the data to be treated as missing.
        'importance_type': "gain",    # Feature importance type 
                                  # ('weight','gain','cover','total_gain','total_cover').
        'predictor': "auto",      # Type of predictor ('cpu_predictor', 'gpu_predictor').
        'enable_categorical': False,  # Whether to enable categorical data support.    

        # mlp numerical core hyperparameters optimized by optuna
        'n_layers': [1, 3],         # used to determine hiden_layer_sizes depth
        'units': [10, 200],         # used to determine hiden_layer_sizes width
        'alpha': [1e-6, 1e-1],      # L2 regularization term
        'learning_rate_init': [1e-5, 1e-1],  # initial learning rate

        # mlp numerical solver-specific hyperparameters optimized by optuna
        'batch_size': [32, 1024],   # batch size for stochastic optimizers
        'momentum': [0.5, 0.99],    # for gradient descent update
        'power_t': [0.1, 0.9],      # L2 regularization term
        'beta_1': [0.8, 0.99],      # decay for first moment adam
        'beta_2': [0.9, 0.9999],    # decay rate for second moment adam
        'epsilon': [1e-8, 1e-4],    # numerical stability in adam
        'max_fun': [15000, 50000],  # used for solver lbfgs max number function calls

        # mlp categorical hyperparameters optimized by optuna
        'activation': ["relu", "tanh", "logistic"],  # hidden layer activation method
        'solver': ["adam", "sgd", "lbfgs"],          # for weight optimization
        'mlp_learning_rate': 
            ["constant", "invscaling", "adaptive"],  # for weight updates
        'early_stopping': [True, False],     # terminate when score not improving
        'nesterov': [True, False],      # used for nesterovs_momentum if solver is sgd

        # mlp extra_params that are optional user-specified for optuna
        'tol': 1e-4,
        'max_iter': 200,

        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'params_stage1' = study.best_trial.user_attrs.get('params_stage1')
                - 'model_stage1' = study.best_trial.user_attrs.get('model_stage1')
                - 'params_stage2' = study.best_trial.user_attrs.get('params_stage2')
                - 'model_stage2' = study.best_trial.user_attrs.get('model_stage2')
                - 'results_stage1' = study.best_trial.user_attrs.get('results_stage1')
                - 'selected_features' = study.best_trial.user_attrs.get('selected_features')
                - 'score_mean' = study.best_trial.user_attrs.get('score_mean')
                - 'best_trial' = study.best_trial
                - 'optuna_study': optimzed optuna study object
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model, outputs = xgbmlp_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # True for MLPClassifier
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'verbose': 'on',
        'gpu': True,                         # Autodetect to use gpu if present
        'n_splits': 5,                       # number of splits for KFold CV

        'pruning': False,                    # prune poor optuna trials

        # objective function options
        'show_trial_progress': True,        # print trial numbers during execution
        'use_permutation': False,            # use permutation importances for RFE
        'use_normalized': True,              # normalize the importances for RFE
        'scoring': None,                     # cross_val_score scoring name
        
        # random seed for all functions 
        'random_state': 42,                 # random seed for reproducibility

        # xgb params that are optimized by optuna
        'feature_threshold': [0.001, 0.1],   # threshold for feature_importance
        'xgb_learning_rate': [1e-4, 1.0],   # Step size shrinkage (also called eta).
        'max_depth': [3, 12],               # Maximum depth of a tree.
        'min_child_weight': [1, 10],        # Minimum sum of instance weight (hessian) needed in a child.
        'subsample': [0.5, 1],              # Fraction of samples used for training each tree.
        'colsample_bytree': [0.5, 1],       # Fraction of features used for each tree.
        'gamma': [1e-8, 10.0],              # Minimum loss reduction to make a split.
        'reg_lambda': [1e-8, 10.0],         # L2 regularization term on weights.
        'alpha': [1e-8, 10.0],              # L1 regularization term on weights.
        'n_estimators': [50, 500],        # Number of boosting rounds (trees).

        # xgb extra_params that are optional user-specified
        'verbosity': 1,               # Verbosity of output (0 = silent, 1 = warnings, 2 = info).
        'booster': "gbtree",          # Type of booster ('gbtree', 'gblinear', or 'dart').
        'tree_method': "auto",        # Tree construction algorithm.
        'nthread': -1,                # Number of parallel threads.
        'colsample_bylevel': 1,       # Fraction of features used per tree level.
        'colsample_bynode': 1,        # Fraction of features used per tree node.
        'scale_pos_weight': 1,        # Balancing of positive and negative weights.
        'base_score': 0.5,            # Initial prediction score (global bias).
        'missing': np.nan,            # Value in the data to be treated as missing.
        'importance_type': "gain",    # Feature importance type ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
        'predictor': "auto",          # Type of predictor ('cpu_predictor', 'gpu_predictor').
        'enable_categorical': False,   # Whether to enable categorical data support.    

        # mlp numerical core hyperparameters optimized by optuna
        'n_layers': [1, 3],         # used to determine hiden_layer_sizes depth
        'units': [10, 200],         # used to determine hiden_layer_sizes width
        'alpha': [1e-6, 1e-1],      # L2 regularization term
        'learning_rate_init': [1e-5, 1e-1],  # initial learning rate

        # mlp numerical solver-specific hyperparameters optimized by optuna
        'batch_size': [32, 1024],   # batch size for stochastic optimizers
        'momentum': [0.5, 0.99],    # for gradient descent update
        'power_t': [0.1, 0.9],      # L2 regularization term
        'beta_1': [0.8, 0.99],      # decay for first moment adam
        'beta_2': [0.9, 0.9999],    # decay rate for second moment adam
        'epsilon': [1e-8, 1e-4],    # numerical stability in adam
        'max_fun': [15000, 50000],  # used for solver lbfgs max number function calls

        # mlp categorical hyperparameters optimized by optuna
        'activation': ["relu", "tanh", "logistic"],  # hidden layer activation method
        'solver': ["adam", "sgd", "lbfgs"],          # for weight optimization
        'mlp_learning_rate': ["constant", "invscaling", "adaptive"],  # for weight updates
        'early_stopping': [True, False],     # terminate when score not improving
        'nesterov': [True, False],           # used for nesterovs_momentum if solver is sgd

        # mlp extra_params that are optional user-specified for optuna
        'tol': 1e-4,
        'max_iter': 200,
        
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign objective depending on type of model
    if data['classify']:
        # objective for XGBClassifier
        num_class = y.nunique()
        if num_class == 2:
            # binomial response variable
            data['objective'] = 'binary:logistic'
        else:
            # multinomial response variable
            data['objective'] = 'multi:softmax'
            data['num_class'] = num_class

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import xgbmlp_objective
    study.optimize(lambda trial: xgbmlp_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['params_stage1'] = study.best_trial.user_attrs.get('params_stage1')
    model_outputs['model_stage1'] = study.best_trial.user_attrs.get('model_stage1')
    model_outputs['params_stage2'] = study.best_trial.user_attrs.get('params_stage2')
    model_outputs['model_stage2'] = study.best_trial.user_attrs.get('model_stage2')
    model_outputs['results_stage1'] = study.best_trial.user_attrs.get('results_stage1')
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial

    # get best_params from the optuna study
    best_params = study.best_trial.user_attrs.get('params_stage2')

    model_outputs['best_params'] = best_params

    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']
    if 'n_layers' in best_params:
        del best_params['n_layers']
    if 'units' in best_params:
        del best_params['units']
    if 'nesterov' in best_params:
        del best_params['nesterov']
    prefix = 'n_units_l'
    matching_keys = [key for key in best_params if key.startswith(prefix)]
    for key in matching_keys:
        del best_params[key]
    
    if data['classify']:
        print('Fitting MLPClassifier model with best parameters, please wait ...')    
        fitted_model = MLPClassifier(**best_params).fit(
            X[model_outputs['selected_features']],y)
    else:    
        print('Fitting MLPRegressor model with best parameters, please wait ...')    
        fitted_model = MLPRegressor(**best_params).fit(
            X[model_outputs['selected_features']],y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("MLPClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("MLPClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['MLPClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("MLPClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['MLPRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("MLPRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("MLPRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs
    
def xgbrfe_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by optuna 
    to find the optimum hyper-parameters for XGBoost
    XGBRegressor or XGBClassifier with Recursive Feature Elimination
    using a threshold of feature_importance in two stages:
    Stage 1: XGBoost for feature selection using threshold of feature_importance
    Stage 2: XGBoost using the selected features from Stage 1 
    for classification or regression
    '''
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.inspection import permutation_importance

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')
    
    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)
    
    # XGBoost params
    params_stage1 = {
        "learning_rate": trial.suggest_float("xgb_learning_rate", *kwargs["xgb_learning_rate"], log=True),
        "max_depth": trial.suggest_int("max_depth", *kwargs["max_depth"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *kwargs["min_child_weight"]),
        "subsample": trial.suggest_float("subsample", *kwargs["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *kwargs["colsample_bytree"]),
        "gamma": trial.suggest_float("gamma", *kwargs["gamma"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *kwargs["reg_lambda"], log=True),
        "alpha": trial.suggest_float("alpha", *kwargs["alpha"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *kwargs["n_estimators"]),
        'random_state': kwargs['random_state'],                
        'device': kwargs['device'],                 

        # xgb extra specified args
        "verbosity": kwargs["verbosity"],
        "objective": kwargs["objective"],
        "booster": kwargs["booster"],
        "tree_method": kwargs["tree_method"],
        "nthread": kwargs["nthread"],
        "colsample_bylevel": kwargs["colsample_bylevel"],
        "colsample_bynode": kwargs["colsample_bynode"],
        "base_score": kwargs["base_score"],
        "missing": kwargs["missing"],
        "importance_type": kwargs["importance_type"],
        "enable_categorical": kwargs["enable_categorical"],

    }

    if not kwargs['classify']:
        params_stage1['predictor'] = kwargs['predictor']
        params_stage1['scale_pos_weight'] = kwargs['scale_pos_weight']

    if kwargs['objective'] == 'multi:softmax':
        params_stage1['num_class'] = kwargs['num_class']

    # Stage 1: Fit XGBoost for feature selection
    # print(f'Trial {trial.number+1} stage 1 ...')
    if kwargs['classify']:
        model_stage1 = XGBClassifier(**params_stage1)
    else:
        model_stage1 = XGBRegressor(**params_stage1)
    model_stage1.fit(X, y)
    
    # absolute value of feature importances (not used for feature selection)
    feature_importances_raw = np.abs(model_stage1.feature_importances_)
    feature_importances_norm = feature_importances_raw / feature_importances_raw.sum()

    # absolute value of mean permutation importances
    if kwargs['use_permutation']:
        result = permutation_importance(model_stage1, X, y, n_repeats=5, random_state=seed)
        permutation_importances_raw = np.abs(result.importances_mean)
        permutation_importances_norm = permutation_importances_raw / permutation_importances_raw.sum()

    # feature selection
    threshold = trial.suggest_float("feature_threshold", *kwargs["feature_threshold"], log=True) 
    if kwargs['use_permutation']:
        if kwargs['use_normalized']:
            selected_idx = np.where(permutation_importances_norm > threshold)[0]
        else:
            selected_idx = np.where(permutation_importances_raw > threshold)[0]
    else:
        if kwargs['use_normalized']:
            selected_idx = np.where(feature_importances_norm > threshold)[0]
        else:
            selected_idx = np.where(feature_importances_raw > threshold)[0]
        
    # heavily penalize trials with no selected features
    if len(selected_idx) == 0:
        trial.set_user_attr("selected_features", [])
        return -1e6

    # selected_features
    feature_names = kwargs['feature_names']
    selected_features = [feature_names[i] for i in selected_idx]
    trial.set_user_attr("selected_features", selected_features)

    # dictionary to log results of stage 1
    if kwargs['use_permutation']:
        results_stage1 = {
            "selected_idx": selected_idx,
            "selected_features": selected_features,
            'feature_names': feature_names,
            'use_normalized': kwargs['use_normalized'],
            'use_permutation': kwargs['use_permutation'],
            'threshold': threshold,
            "feature_importances_raw": feature_importances_raw,
            "feature_importances_norm": feature_importances_norm,
            "permutation_importances_raw": permutation_importances_raw,
            "permutation_importances_norm": permutation_importances_norm,
        }
    else:
        results_stage1 = {
            "selected_idx": selected_idx,
            "selected_features": selected_features,
            'feature_names': feature_names,
            'use_normalized': kwargs['use_normalized'],
            'use_permutation': kwargs['use_permutation'],
            'threshold': threshold,
            "feature_importances_raw": feature_importances_raw,
            "feature_importances_norm": feature_importances_norm,
        }
    trial.set_user_attr("results_stage1", results_stage1)
    
    # Subset data
    X_selected = X[selected_features]

    # Stage 2: Fit XGBoost for classification or regression using selected_features
    # print(f'Trial {trial.number+1} stage 2 ...')
    if kwargs['classify']:
        model_stage2 = XGBClassifier(**params_stage1)
        # Cross-validated scoring
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            model_stage2, X_selected, y,
            cv=cv,
            # scoring="f1_weighted"
            scoring=kwargs["scoring"]
        )
    else:
        model_stage2 = XGBRegressor(**params_stage1)
        # Cross-validated scoring
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            model_stage2, X_selected, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )
    score_mean = np.mean(scores)

    # log params, models, and score
    trial.set_user_attr("params_stage1", params_stage1)
    trial.set_user_attr("model_stage1", model_stage1)
    trial.set_user_attr("model_stage2", model_stage2)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
        
    return score_mean
      
def xgbrfe_auto(X, y, **kwargs):

    """
    Autocalibration of hyperparameters for a hybrid model 
    of XGBoost with Recursive Feature Elimination
    Stage 1: XGBoost for feature selection using threshold of feature importance
    Stage 2: XGBoost for classification or regression using selected features 

    by
    Greg Pelletier
    gjpelletier@gmail.com
    20-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,               # number of optuna trials
        classify= False,            # True for MLPClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',              # 'on' to display all 
        gpu= True,                  # Autodetect to use gpu if present
        n_splits= 5,                # number of splits for KFold CV
        pruning= False,             # prune poor optuna trials

        # random seed for all functions 
        'random_state': 42,         # random seed for reproducibility

        # objective function options
        'show_trial_progress': True, # print each trial number and best cv score
        'use_permutation': False,    # True to use abs permutation importances for RFE
                                     # False to use abs .feature_importances_ for RFE
        'use_normalized': True,      # True to normalize the abs importances for RFE
                                     # False to use raw abs importances

        # xgb params that are optimized by optuna
        'feature_threshold': [0.001, 0.1],  # threshold for feature_importance
        'xgb_learning_rate': [1e-4, 1.0],   # Step size shrinkage (also called eta).
        'max_depth': [3, 12],               # Max depth of a tree.
        'min_child_weight': [1, 10],   # Min sum of instance weight (hessian) in a child.
        'subsample': [0.5, 1],         # Fraction of samples used for training each tree.
        'colsample_bytree': [0.5, 1],       # Fraction of features used for each tree.
        'gamma': [1e-8, 10.0],              # Minimum loss reduction to make a split.
        'reg_lambda': [1e-8, 10.0],         # L2 regularization term on weights.
        'alpha': [1e-8, 10.0],              # L1 regularization term on weights.
        'n_estimators': [50, 500],        # Number of boosting rounds (trees).

        # xgb extra_params that are optional user-specified
        'verbosity': 1,            # Verbosity of output (0=silent, 1=warnings, 2=info).
        'booster': "gbtree",          # Type of booster ('gbtree','gblinear','dart').
        'tree_method': "auto",        # Tree construction algorithm.
        'nthread': -1,                # Number of parallel threads.
        'colsample_bylevel': 1,       # Fraction of features used per tree level.
        'colsample_bynode': 1,        # Fraction of features used per tree node.
        'scale_pos_weight': 1,        # Balancing of positive and negative weights.
        'base_score': 0.5,            # Initial prediction score (global bias).
        'missing': np.nan,            # Value in the data to be treated as missing.
        'importance_type': "gain",    # Feature importance type 
                                  # ('weight','gain','cover','total_gain','total_cover').
        'predictor': "auto",      # Type of predictor ('cpu_predictor', 'gpu_predictor').
        'enable_categorical': False,  # Whether to enable categorical data support.    

        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'params_stage1' = study.best_trial.user_attrs.get('params_stage1')
                - 'model_stage1' = study.best_trial.user_attrs.get('model_stage1')
                - 'model_stage2' = study.best_trial.user_attrs.get('model_stage2')
                - 'results_stage1' = study.best_trial.user_attrs.get('results_stage1')
                - 'selected_features' = study.best_trial.user_attrs.get('selected_features')
                - 'score_mean' = study.best_trial.user_attrs.get('score_mean')
                - 'best_trial' = study.best_trial
                - 'optuna_study': optimzed optuna study object
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model, outputs = xgbrfe_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna
    from xgboost import XGBRegressor, XGBClassifier

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # True for MLPClassifier
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'verbose': 'on',
        'gpu': True,                         # Autodetect to use gpu if present
        'n_splits': 5,                       # number of splits for KFold CV

        'pruning': False,                    # prune poor optuna trials

        # objective function options
        'show_trial_progress': True,         # print trial numbers during execution
        'use_permutation': False,            # use permutation importances for RFE
        'use_normalized': True,              # normalize the importances for RFE
        'scoring': None,                     # cross_val_score scoring name
        
        # random seed for all functions 
        'random_state': 42,                 # random seed for reproducibility

        # xgb params that are optimized by optuna
        'feature_threshold': [0.001, 0.1],   # threshold for feature_importance
        'xgb_learning_rate': [1e-4, 1.0],   # Step size shrinkage (also called eta).
        'max_depth': [3, 12],               # Maximum depth of a tree.
        'min_child_weight': [1, 10],        # Minimum sum of instance weight (hessian) needed in a child.
        'subsample': [0.5, 1],              # Fraction of samples used for training each tree.
        'colsample_bytree': [0.5, 1],       # Fraction of features used for each tree.
        'gamma': [1e-8, 10.0],              # Minimum loss reduction to make a split.
        'reg_lambda': [1e-8, 10.0],         # L2 regularization term on weights.
        'alpha': [1e-8, 10.0],              # L1 regularization term on weights.
        'n_estimators': [50, 500],        # Number of boosting rounds (trees).

        # xgb extra_params that are optional user-specified
        'verbosity': 1,               # Verbosity of output (0 = silent, 1 = warnings, 2 = info).
        'booster': "gbtree",          # Type of booster ('gbtree', 'gblinear', or 'dart').
        'tree_method': "auto",        # Tree construction algorithm.
        'nthread': -1,                # Number of parallel threads.
        'colsample_bylevel': 1,       # Fraction of features used per tree level.
        'colsample_bynode': 1,        # Fraction of features used per tree node.
        'scale_pos_weight': 1,        # Balancing of positive and negative weights.
        'base_score': 0.5,            # Initial prediction score (global bias).
        'missing': np.nan,            # Value in the data to be treated as missing.
        'importance_type': "gain",    # Feature importance type ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
        'predictor': "auto",          # Type of predictor ('cpu_predictor', 'gpu_predictor').
        'enable_categorical': False,   # Whether to enable categorical data support.    
        
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")
        
    # assign objective depending on type of model
    if data['classify']:
        # objective for XGBClassifier
        num_class = y.nunique()
        if num_class == 2:
            # binomial response variable
            data['objective'] = 'binary:logistic'
        else:
            # multinomial response variable
            data['objective'] = 'multi:softmax'
            data['num_class'] = num_class
    else:
        # objective for XGBRegressor
        data['objective'] = 'reg:squarederror'

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'enable_categorical': data['enable_categorical'],
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import xgbrfe_objective
    study.optimize(lambda trial: xgbrfe_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['params_stage1'] = study.best_trial.user_attrs.get('params_stage1')
    model_outputs['model_stage1'] = study.best_trial.user_attrs.get('model_stage1')
    model_outputs['model_stage2'] = study.best_trial.user_attrs.get('model_stage2')
    model_outputs['results_stage1'] = study.best_trial.user_attrs.get('results_stage1')
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial

    # get best_params from the optuna study
    best_params = study.best_trial.user_attrs.get('params_stage1')

    model_outputs['best_params'] = best_params

    if 'num_features' in best_params:
        del best_params['num_features']
    if 'selector_type' in best_params:
        del best_params['selector_type']
    if 'n_layers' in best_params:
        del best_params['n_layers']
    if 'units' in best_params:
        del best_params['units']
    if 'nesterov' in best_params:
        del best_params['nesterov']
    prefix = 'n_units_l'
    matching_keys = [key for key in best_params if key.startswith(prefix)]
    for key in matching_keys:
        del best_params[key]
    
    if data['classify']:
        print('Fitting XGBClassifier model with best parameters, please wait ...')    
        fitted_model = XGBClassifier(**best_params).fit(
            X[model_outputs['selected_features']],y)
    else:    
        print('Fitting XGBRegressor model with best parameters, please wait ...')    
        fitted_model = XGBRegressor(**best_params).fit(
            X[model_outputs['selected_features']],y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("XGBClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("XGBClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['XGBClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("XGBClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['XGBRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("XGBRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("XGBRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs

def adarfe_objective(trial, X, y, study, **kwargs):
    '''
    Objective function used by optuna 
    to find the optimum hyper-parameters for XGBoost
    XGBRegressor or AdaBoostClassifier with Recursive Feature Elimination
    using a threshold of feature_importance in two stages:
    Stage 1: AdaBoost for feature selection using threshold of feature_importance
    Stage 2: AdaBoost using the selected features from Stage 1 
    for classification or regression
    '''
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
    from PyMLR import detect_gpu
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier 

    if kwargs['show_trial_progress'] and trial.number > 0:
        print(f'Trial {trial.number}, best cv test score so far: {study.best_value:.6f} ...')
    
    seed = kwargs.get("random_state", 42)
    rng = np.random.default_rng(seed)
    
    # AdaBoost params_stage1
    params_stage1 = {
        # AdaBoost params_stage1
        "n_estimators": trial.suggest_int("n_estimators",
            *kwargs['n_estimators']),
        'learning_rate': trial.suggest_float("learning_rate", 
            *kwargs['learning_rate'], log=True),        
        'random_state': kwargs['random_state'],                
    }

    # DecisionTree params for base_estimator of AdaBoost
    params_estimator = {
        "max_depth": trial.suggest_int("max_depth",
            *kwargs['max_depth']),
        "min_samples_split": trial.suggest_int("min_samples_split",
            *kwargs['min_samples_split'], log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf",
            *kwargs['min_samples_leaf'], log=True),
        "max_features": trial.suggest_float("max_features",
            *kwargs['max_features']),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes",
            *kwargs['max_leaf_nodes'], log=True),
    }

    # Use DecisionTree as estimator for AdaBoost
    if kwargs['classify']:
        params_stage1['estimator'] = DecisionTreeClassifier(
            max_depth= params_estimator['max_depth'], 
            min_samples_split= params_estimator['min_samples_split'], 
            min_samples_leaf= params_estimator['min_samples_leaf'], 
            max_features= params_estimator['max_features'], 
            max_leaf_nodes= params_estimator['max_leaf_nodes'], 
            random_state=seed)
    else:
        params_stage1['estimator'] = DecisionTreeRegressor(
            max_depth= params_estimator['max_depth'], 
            min_samples_split= params_estimator['min_samples_split'], 
            min_samples_leaf= params_estimator['min_samples_leaf'], 
            max_features= params_estimator['max_features'], 
            max_leaf_nodes= params_estimator['max_leaf_nodes'], 
            random_state=seed)
        params_stage1['loss'] = trial.suggest_categorical("loss", kwargs['loss'])

    # Stage 1: Fit XGBoost for feature selection
    # print(f'Trial {trial.number+1} stage 1 ...')
    if kwargs['classify']:
        model_stage1 = AdaBoostClassifier(**params_stage1)
    else:
        model_stage1 = AdaBoostRegressor(**params_stage1)
    model_stage1.fit(X, y)
    
    # absolute value of feature importances (not used for feature selection)
    feature_importances_raw = np.abs(model_stage1.feature_importances_)
    feature_importances_norm = feature_importances_raw / feature_importances_raw.sum()

    # absolute value of mean permutation importances
    if kwargs['use_permutation']:
        result = permutation_importance(
            model_stage1, X, y, n_repeats=5, random_state=seed, n_jobs=kwargs['n_jobs'])
        permutation_importances_raw = np.abs(result.importances_mean)
        permutation_importances_norm = permutation_importances_raw / permutation_importances_raw.sum()

    # feature selection
    threshold = trial.suggest_float("feature_threshold", *kwargs["feature_threshold"], log=True) 
    if kwargs['use_permutation']:
        if kwargs['use_normalized']:
            selected_idx = np.where(permutation_importances_norm > threshold)[0]
        else:
            selected_idx = np.where(permutation_importances_raw > threshold)[0]
    else:
        if kwargs['use_normalized']:
            selected_idx = np.where(feature_importances_norm > threshold)[0]
        else:
            selected_idx = np.where(feature_importances_raw > threshold)[0]
        
    # heavily penalize trials with no selected features
    if len(selected_idx) == 0:
        trial.set_user_attr("selected_features", [])
        return -1e6

    # selected_features
    feature_names = kwargs['feature_names']
    selected_features = [feature_names[i] for i in selected_idx]
    trial.set_user_attr("selected_features", selected_features)

    # dictionary to log results of stage 1
    if kwargs['use_permutation']:
        results_stage1 = {
            "selected_idx": selected_idx,
            "selected_features": selected_features,
            'feature_names': feature_names,
            'use_normalized': kwargs['use_normalized'],
            'use_permutation': kwargs['use_permutation'],
            'threshold': threshold,
            "feature_importances_raw": feature_importances_raw,
            "feature_importances_norm": feature_importances_norm,
            "permutation_importances_raw": permutation_importances_raw,
            "permutation_importances_norm": permutation_importances_norm,
        }
    else:
        results_stage1 = {
            "selected_idx": selected_idx,
            "selected_features": selected_features,
            'feature_names': feature_names,
            'use_normalized': kwargs['use_normalized'],
            'use_permutation': kwargs['use_permutation'],
            'threshold': threshold,
            "feature_importances_raw": feature_importances_raw,
            "feature_importances_norm": feature_importances_norm,
        }
    trial.set_user_attr("results_stage1", results_stage1)
    
    # Subset data
    X_selected = X[selected_features]

    # Stage 2: Fit AdaBoost for classification or regression using selected_features
    if kwargs['classify']:
        model_stage2 = AdaBoostClassifier(**params_stage1)
        # Cross-validated scoring
        cv = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        scores = cross_val_score(
            model_stage2, X_selected, y,
            cv=cv,
            # scoring="f1_weighted"
            scoring=kwargs["scoring"]
        )
    else:
        model_stage2 = AdaBoostRegressor(**params_stage1)
        # Cross-validated scoring
        cv = RepeatedKFold(n_splits=kwargs["n_splits"], n_repeats=2, random_state=seed)
        scores = cross_val_score(
            model_stage2, X_selected, y,
            cv=cv,
            # scoring="neg_root_mean_squared_error"
            scoring=kwargs["scoring"]
        )
    score_mean = np.mean(scores)

    # log params, models, and score
    trial.set_user_attr("params_estimator", params_estimator)
    trial.set_user_attr("params_stage1", params_stage1)
    trial.set_user_attr("model_stage1", model_stage1)
    trial.set_user_attr("model_stage2", model_stage2)
    trial.set_user_attr("scoring", kwargs["scoring"])
    trial.set_user_attr("score_mean", score_mean)
        
    return score_mean
 
def adarfe_auto(X, y, **kwargs):

    """
    Autocalibration of hyperparameters for a hybrid model 
    of AdaBoost with Recursive Feature Elimination
    Stage 1: AdaBoost for feature selection using threshold of feature importance
    Stage 2: AdaBoost for classification or regression using selected features 

    by
    Greg Pelletier
    gjpelletier@gmail.com
    20-Aug-2025

    REQUIRED INPUTS (X and y should have same number of rows and 
    only contain real numbers)
    X = dataframe of the candidate independent variables 
        (as many columns of data as needed)
    y = dataframe of the dependent variable (one column of data)

    OPTIONAL KEYWORD ARGUMENTS
    **kwargs (optional keyword arguments):
        n_trials= 50,               # number of optuna trials
        classify= False,            # True for MLPClassifier
        preprocess= True,           # Apply OneHotEncoder and StandardScaler
        preprocess_result= None,    # dict of the following result from 
                                    # preprocess_train if available:         
                                    # - encoder          (OneHotEncoder)
                                    # - scaler           (StandardScaler)
                                    # - categorical_cols (categorical cols)
                                    # - non_numeric_cats (non-num cat cols)
                                    # - continuous_cols  (continuous cols)
        verbose= 'on',              # 'on' to display all 
        gpu= True,                  # Autodetect to use gpu if present
        n_splits= 5,                # number of splits for KFold CV
        pruning= False,             # prune poor optuna trials

        # random seed for all functions 
        'random_state': 42,         # random seed for reproducibility

        # objective function options
        'show_trial_progress': True, # print each trial number and best cv score
        'use_permutation': False,    # True to use abs permutation importances for RFE
                                     # False to use abs .feature_importances_ for RFE
        'use_normalized': True,      # True to normalize the abs importances for RFE
                                     # False to use raw abs importances

        # param for for permutation_importance 
        'n_jobs': -1,                 # -1 to use all CPU cores with cross_val_score
        
        # params for AdaBoost optimized by optuna
        'feature_threshold': [0.001, 0.1],   # threshold for feature_importance
        'n_estimators': [50, 500],
        'learning_rate': [0.01, 1.0],
        'loss': ["linear", "square", "exponential"],  # loss fn for regressor
        
        # params for base_estimator (DecisionTree)
        'max_depth': [2, 30],               # max depth of a tree
        'min_samples_split': [2, 20],       # min samples to split internal node
        'min_samples_leaf': [1, 20],        # min samples to be at a leaf node
        'max_features': [0.1, 1.0],         # number of features to consider 
                                            # when looking for the best split
        'max_leaf_nodes': [10, 100],        # max number of leaf nodes 
        
        preprocessing options:
            use_encoder (bool): True (default) or False
            use_scaler (bool): True (default) or False
            threshold_cat (int): Max unique values for numeric columns 
                to be considered categorical (default: 12)
            scale (str): 'minmax' or 'standard' for scaler (default: 'standard')
            unskew_pos (bool): True: use log1p transform on features with 
                skewness greater than threshold_skew_pos (default: False)
            threshold_skew_pos: threshold skewness to log1p transform features
                used if unskew_pos=True (default: 0.5)
            unskew_neg (bool): True: use sqrt transform on features with 
                skewness less than threshold_skew_neg (default: False)
            threshold_skew_neg: threshold skewness to sqrt transform features
                used if unskew_neg=True (default: -0.5)

    RETURNS
        fitted_model, model_outputs
            model_objects is the fitted model object
            model_outputs is a dictionary of the following outputs: 
                - 'preprocess': True for OneHotEncoder and StandardScaler
                - 'preprocess_result': output or echo of the following:
                    - 'encoder': OneHotEncoder for categorical X
                    - 'scaler': StandardScaler for continuous X
                    - 'categorical_cols': categorical numerical columns 
                    - 'non_numeric_cats': non-numeric categorical columns 
                    - 'continous_cols': continuous numerical columns
                - 'params_stage1' = study.best_trial.user_attrs.get('params_stage1')
                - 'model_stage1' = study.best_trial.user_attrs.get('model_stage1')
                - 'model_stage2' = study.best_trial.user_attrs.get('model_stage2')
                - 'results_stage1' = study.best_trial.user_attrs.get('results_stage1')
                - 'selected_features' = study.best_trial.user_attrs.get('selected_features')
                - 'score_mean' = study.best_trial.user_attrs.get('score_mean')
                - 'best_trial' = study.best_trial
                - 'optuna_study': optimzed optuna study object
                - 'best_params': best model hyper-parameters found by optuna
                - 'y_pred': Predicted y values
                - 'residuals': Residuals (y-y_pred) for each of the four methods
                - 'stats': Regression statistics for each model

    NOTE
    Do any necessary/optional cleaning of the data before 
    passing the data to this function. X and y should have the same number of rows
    and contain only real numbers with no missing values. X can contain as many
    columns as needed, but y should only be one column. X should have unique
    column names for for each column

    EXAMPLE 
    model, outputs = xgbrfe_auto(X, y)

    """

    from PyMLR import stats_given_y_pred, detect_dummy_variables, detect_gpu
    from PyMLR import preprocess_train, preprocess_test, check_X_y, fitness_metrics
    from PyMLR import fitness_metrics_logistic, pseudo_r2
    from PyMLR import plot_confusion_matrix, plot_roc_auc
    import time
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone
    from sklearn.metrics import PredictionErrorDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    import sys
    import statsmodels.api as sm
    import optuna
    from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier 

    # Check if cwd has write permissions and change cwd to home if not
    import os
    from pathlib import Path
    if not os.access(os.getcwd(), os.W_OK):
        # Change the working directory to home if no write permissions
        print(f"Current working directory has no write permission: {os.getcwd()}")    
        os.chdir(Path.home())
        os.makedirs('pymlr', exist_ok=True)
        os.chdir('pymlr')
        print(f"Working directory changed to: {os.getcwd()}")    

    # Define default values of input data arguments
    defaults = {
        'n_trials': 50,                     # number of optuna trials
        'classify': False,            # True for MLPClassifier
        'preprocess': True,           # True for OneHotEncoder and StandardScaler
        'preprocess_result': None,    # dict of  the following result from 
                                      # preprocess_train if available:         
                                      # - encoder          (OneHotEncoder) 
                                      # - scaler           (StandardScaler)
                                      # - categorical_cols (categorical columns)
                                      # - non_numeric_cats (non-numeric cats)
                                      # - continuous_cols  (continuous columns)
        # --- preprocess_train ---
        'use_encoder': True, 
        'use_scaler': True, 
        'threshold_cat': 12,    # threshold number of unique items for categorical 
        'scale': 'standard', 
        'unskew_pos': False, 
        'threshold_skew_pos': 0.5,
        'unskew_neg': False, 
        'threshold_skew_neg': -0.5,        
        # ------------------------
        'verbose': 'on',
        'gpu': True,                         # Autodetect to use gpu if present
        'n_splits': 5,                       # number of splits for KFold CV

        'pruning': False,                    # prune poor optuna trials

        # objective function options
        'show_trial_progress': True,         # print trial numbers during execution
        'use_permutation': False,            # use permutation importances for RFE
        'use_normalized': True,              # normalize the importances for RFE
        'scoring': None,                     # cross_val_score scoring name
        
        # random seed for all functions 
        'random_state': 42,                 # random seed for reproducibility

        # param for for permutation_importance 
        'n_jobs': -1,                 # -1 to use all CPU cores with cross_val_score
        
        # params for AdaBoost optimized by optuna
        'feature_threshold': [0.001, 0.1],   # threshold for feature_importance
        'n_estimators': [50, 500],
        'learning_rate': [0.01, 1.0],
        'loss': ["linear", "square", "exponential"],  # loss fn for regressor
        
        # params for base_estimator (DecisionTree)
        'max_depth': [2, 30],               # max depth of a tree
        'min_samples_split': [2, 20],       # min samples to split internal node
        'min_samples_leaf': [1, 20],        # min samples to be at a leaf node
        'max_features': [0.1, 1.0],         # number of features to consider 
                                            # when looking for the best split
        'max_leaf_nodes': [10, 100],        # max number of leaf nodes        
    }

    # Update input data argumements with any provided keyword arguments in kwargs
    data = {**defaults, **kwargs}

    # print a warning for unexpected input kwargs
    unexpected = kwargs.keys() - defaults.keys()
    if unexpected:
        # raise ValueError(f"Unexpected argument(s): {unexpected}")
        print(f"Unexpected input kwargs: {unexpected}")

    # Auto-detect if GPU is present and use GPU if present
    if data['gpu']:
        use_gpu = detect_gpu()
        if use_gpu:
            data['device'] = 'gpu'
        else:
            data['device'] = 'cpu'
    else:
        data['device'] = 'cpu'

    # copy X and y to avoid altering the originals
    X = X.copy()
    y = y.copy()
    
    X, y = check_X_y(X,y)

    # Warn the user to consider using classify=True if y has < 12 classes
    if y.nunique() <= 12 and not data['classify']:
        print(f"Warning: y has {y.nunique()} classes, consider using optional argument classify=True")

    # assign scoring depending on type of model
    if data['classify']:
        if data['scoring'] == None:
            data['scoring'] = "f1_weighted"
    else:
        if data['scoring'] == None:
            data['scoring'] = "neg_root_mean_squared_error"
        
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Set start time for calculating run time
    start_time = time.time()

    # Set global random seed
    np.random.seed(data['random_state'])

    # check if X contains dummy variables
    X_has_dummies = detect_dummy_variables(X)

    # Initialize output dictionaries
    model_objects = {}
    model_outputs = {}

    # Pre-process X to apply OneHotEncoder and StandardScaler
    if data['preprocess']:
        if data['preprocess_result']!=None:
            # print('preprocess_test')
            X = preprocess_test(X, data['preprocess_result'])
        else:
            kwargs_pre = {
                'use_encoder': data['use_encoder'],
                'use_scaler': data['use_scaler'],
                'threshold_cat': data['threshold_cat'],
                'scale': data['scale'], 
                'unskew_pos': data['unskew_pos'], 
                'threshold_skew_pos': data['threshold_skew_pos'],
                'unskew_neg': data['unskew_neg'], 
                'threshold_skew_neg': data['threshold_skew_neg']        
            }
            data['preprocess_result'] = preprocess_train(X, **kwargs_pre)
            X = data['preprocess_result']['df_processed']

    data['feature_names'] = X.columns.to_list()

    print('Running optuna to find best parameters, could take a few minutes, please wait...')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # optional pruning
    if data['pruning']:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True),
            pruner=optuna.pruners.MedianPruner())
    else:
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=data['random_state'], multivariate=True))
    
    X_opt = X.copy()    # copy X to prevent altering the original

    from PyMLR import adarfe_objective
    study.optimize(lambda trial: adarfe_objective(trial, X_opt, y, study, **data), n_trials=data['n_trials'])

    # save outputs
    model_outputs['preprocess'] = data['preprocess']   
    model_outputs['preprocess_result'] = data['preprocess_result'] 
    model_outputs['X_processed'] = X.copy()
    model_outputs['pruning'] = data['pruning']
    model_outputs['optuna_study'] = study
    model_outputs['params_estimator'] = study.best_trial.user_attrs.get('params_estimator')
    model_outputs['params_stage1'] = study.best_trial.user_attrs.get('params_stage1')
    model_outputs['model_stage1'] = study.best_trial.user_attrs.get('model_stage1')
    model_outputs['model_stage2'] = study.best_trial.user_attrs.get('model_stage2')
    model_outputs['results_stage1'] = study.best_trial.user_attrs.get('results_stage1')
    model_outputs['selected_features'] = study.best_trial.user_attrs.get('selected_features')
    model_outputs['scoring'] = study.best_trial.user_attrs.get('scoring')
    model_outputs['score_mean'] = study.best_trial.user_attrs.get('score_mean')
    model_outputs['best_trial'] = study.best_trial

    # get best_params from the optuna study
    best_params = study.best_trial.user_attrs.get('params_stage1')

    model_outputs['best_params'] = best_params

    if data['classify']:
        print('Fitting AdaBoostClassifier model with best parameters, please wait ...')    
        fitted_model = AdaBoostClassifier(**best_params).fit(
            X[model_outputs['selected_features']],y)
    else:    
        print('Fitting AdaBoostRegressor model with best parameters, please wait ...')    
        fitted_model = AdaBoostRegressor(**best_params).fit(
            X[model_outputs['selected_features']],y)

    if data['classify']:
        if data['verbose'] == 'on':    
            # confusion matrix
            selected_features = model_outputs['selected_features']
            hfig = plot_confusion_matrix(fitted_model, X[selected_features], y)
            hfig.savefig("AdaBoostClassifier_confusion_matrix.png", dpi=300)            
            # ROC curve with AUC
            selected_features = model_outputs['selected_features']
            hfig = plot_roc_auc(fitted_model, X[selected_features], y)
            hfig.savefig("AdaBoostClassifier_ROC_curve.png", dpi=300)            
        # Goodness of fit statistics
        metrics = fitness_metrics_logistic(
            fitted_model, 
            X[model_outputs['selected_features']], y, brier=False)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['AdaBoostClassifier']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])    
        if data['verbose'] == 'on':
            print('')
            print("AdaBoostClassifier goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')    
    else:
    
        # check to see of the model has intercept and coefficients
        if (hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_') 
                and fitted_model.coef_.size==len(X[model_outputs['selected_features']].columns)):
            intercept = fitted_model.intercept_
            coefficients = fitted_model.coef_
            # dataframe of model parameters, intercept and coefficients, including zero coefs
            n_param = 1 + fitted_model.coef_.size               # number of parameters including intercept
            popt = [['' for i in range(n_param)], np.full(n_param,np.nan)]
            for i in range(n_param):
                if i == 0:
                    popt[0][i] = 'Intercept'
                    popt[1][i] = fitted_model.intercept_
                else:
                    popt[0][i] = X[model_outputs['selected_features']].columns[i-1]
                    popt[1][i] = fitted_model.coef_[i-1]
            popt = pd.DataFrame(popt).T
            popt.columns = ['Feature', 'Parameter']
            # Table of intercept and coef
            popt_table = pd.DataFrame({
                    "Feature": popt['Feature'],
                    "Parameter": popt['Parameter']
                })
            popt_table.set_index('Feature',inplace=True)
            model_outputs['popt_table'] = popt_table
        
        # Goodness of fit statistics
        metrics = fitness_metrics(
            fitted_model, 
            X[model_outputs['selected_features']], y)
        stats = pd.DataFrame([metrics]).T
        stats.index.name = 'Statistic'
        stats.columns = ['AdaBoostRegressor']
        model_outputs['metrics'] = metrics
        model_outputs['stats'] = stats
        model_outputs['y_pred'] = fitted_model.predict(X[model_outputs['selected_features']])
    
        if data['verbose'] == 'on':
            print('')
            print("AdaBoostRegressor goodness of fit to training data in model_outputs['stats']:")
            print('')
            print(model_outputs['stats'].to_markdown(index=True))
            print('')
    
        if hasattr(fitted_model, 'intercept_') and hasattr(fitted_model, 'coef_'):
            print("Parameters of fitted model in model_outputs['popt']:")
            print('')
            print(model_outputs['popt_table'].to_markdown(index=True))
            print('')
    
        # residual plot for training error
        if data['verbose'] == 'on':
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="actual_vs_predicted",
                ax=axs[0]
            )
            axs[0].set_title("Actual vs. Predicted")
            PredictionErrorDisplay.from_predictions(
                y,
                y_pred=model_outputs['y_pred'],
                kind="residual_vs_predicted",
                ax=axs[1]
            )
            axs[1].set_title("Residuals vs. Predicted")
            fig.suptitle(
                f"Predictions compared with actual values and residuals (RMSE={metrics['RMSE']:.3f})")
            plt.tight_layout()
            # plt.show()
            plt.savefig("AdaBoostRegressor_predictions.png", dpi=300)

    # Best score of CV test data
    print('')
    print(f"Best-fit score of CV test data: {study.best_value:.6f}")
    print('')

    # Print the run time
    fit_time = time.time() - start_time
    print('Done')
    print(f"Time elapsed: {fit_time:.2f} sec")
    print('')

    # Restore warnings to normal
    warnings.filterwarnings("default")

    return fitted_model, model_outputs


