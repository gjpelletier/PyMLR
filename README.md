# Python tools for Machine Learning Regression

The PyMLR module includes the following main Python functions to facilitate the use of sklearn, XGBoost, CatBoost, LightGBM, statsmodels, and optuna solvers for linear and logistic regression: 
 
 - **logistic_auto** - LogisticRegression with automatic cross-validated optimization of the hyperparameters
 - **logistic** - LogisticRegression with user-specified hyperparameters
 - **lasso** - LassoCV, LassoLarsCV, and LassoLarsIC using AIC/BIC
 - **ridge** - RidgeCV and Ridge using VIF
 - **elastic** - ElasticNetCV
 - **xgb_auto** - XGBoostRegressor with automatic cross-validated optimization of the hyperparameters
 - **xgb** - XGBoostRegressor with user-specified hyperparameters
 - **catboost_auto** - CatBoostRegressor with automatic cross-validated optimization of the hyperparameters
 - **catboost** - CatBoostRegressor with user-specified hyperparameters
 - **gbr_auto** - GradientBoostingRegressor with automatic cross-validated optimization of the hyperparameters
 - **gbr** - GradientBoostingRegressor with user-specified hyperparameters
 - **forest_auto** - RandomForestRegressor with automatic cross-validated optimization of the hyperparameters
 - **forest** - RandomForestRegressor with user-specified hyperparameters
 - **svr_auto** - SVR with automatic cross-validated optimization of the hyperparameters
 - **svr** - SVR with user-specified hyperparameters
 - **knn_auto** - KNeighborsRegressor with automatic cross-validated optimization of the hyperparameters and optional PCA transformation
 - **knn** - KNeighborsRegressor with user-specified hyperparameters and optional PCA transformation
 - **lgbm** - LGBMRegressor with user-specified hyperparameters
 - **stacking** - StackingRegressor
 - **stepwise** - Stepwise multiple linear regression to minimize MSE, AIC, or BIC
 - **model_agnostic** - Perform model-agnostic analysis (e.g. Shapely Additive Explantions (SHAP), Beeswarm and Bar plots, PDB-ICE plots, Permutation Importance plots)
 - **show_optuna** - Display the results of the hyperparameter optimization using optuna
 
The functions in PyMLR allow the user to write one line of high-level python code to provide the following:

- final fitted model object
- regression summary statistics of the best fit model (e.g. r-squared, adjusted r-squared, RMSE of residuals, p-value of the F-statistic, AIC, BIC, alpha, etc., if applicable)
- best fit intercept and model coefficients (if applicable)
- variance  inflation factors (if applicable)
- plots of predicted vs actual and predicted vs residuals
- figures showing other diagnostic descriptions of the analysis if applicable (e.g. coefficients vs alpha, MSE vs alpha, AIC and BIC vs alpha)
- show plots of optuna optimization history, hyperparameter importance, and contour plots of relationships between hyperparameters and MSE 
- automatic cross-validated Bayesian optimization of hyperparameters
- automatic PCA transform of X with PCA-KNN regression and output of the optimized fitted PCA transformer (optional, knn_auto and knn)
- automatic detection and use of GPU for computations (xgb, catboost)
- automatic prevention of overfitting the training data with KNN regression (knn)
- automatic detection of categorical features and application of OneHotEncoder (logistic, xgb)
- automatic standardization of X and output of the scaler (optional)
- optimization of feature selection (logistic, xgb)
- optimization of logistic regression for binomial or multinomial response variables

### Lasso

The **lasso** function in the PyMLR module provides output of regression models and summary statistics using the following methods using sklearn.linear_model:

- LassoCV: Lasso using Cross-Validation with coordinate descent to optimize alpha
- LassoLarsCV: Lasso using Cross-Validation with Least Angle Regression
- LassoLarsIC using AIC: Lasso using Least Angle Regression with Akaike Information Criterion
- LassoLarsIC using BIC: Lasso using Least Angle Regression with Bayesian Information Criterion

Lasso (Least Absolute Shrinkage and Selection Operator) adds a penalty to the loss function. This penalty encourages sparsity in the model, meaning that some coefficients will be exactly zero, effectively removing the corresponding predictors from the model. 

Lasso linear regression includes an L1 penalty term to the standard least squares objective function. The penalty term is a sum of the absolute values of the regression coefficients multiplied by a hyperparameter, denoted as "alpha". The **lasso** function finds the optimum value of alpha for each of the methods listed above. The alpha determines the amount of shrinkage applied to the model coefficients. As alpha increases, the coefficients are pushed towards zero, and some may become exactly zero, effectively eliminating those features from the model. 

Lasso regression is useful for dealing with multicollinearity, where predictors are highly correlated, and when an optimal subset of the candidate features should be included in the model. 

### Ridge

The **ridge** function in the PyMLR module provides output of regression models and summary statistics using the following methods using sklearn.linear_model:

- RidgeCV: RidgeCV regression with cross-validation using the MSE as the scoring criterion to optimize alpha
- RidgeVIF: Ridge regression using target VIF to optimize alpha by trial

Ridge regression adds an L2 penalty to the loss function, which is the product of the regularization hyperparameter and the sum of the squares of the coefficients. This penalty shrinks the coefficients towards zero but does not force them to be exactly zero. 

Ridge regression is useful for dealing with multicollinearity, where predictors are highly correlated, and when all candidate features should be included in the model. 

### Elastic Net

The **elastic** function in the PyMLR module provides output of the fitted regression model and summary statistics using the following method using sklearn.linear_model:

- ElasticNetCV: Elastic regression with cross-validation using the MSE as the scoring criterion to optimize alpha and the L1-ratio that balances between L1 and L2 regularization.

Elastic regression, also know as Elastic Net, is a regularization technique that combines the strengths of Lasso (L1) and Ridge (L2) regression methods. It is particularly useful for handling datasets with high-dimensional features and multicollinearity (correlated features). By blending the penalties of L1 and L2, Elastic Net provides a balance between feature selection (Lasso) and coefficient shrinkage (Ridge).

### Stacking

The **stacking** function in the PyMLR module provides output of a fitted regression model and summary statistics using the sklearn StackingRegressor function for ensemble modeling with any combination of sklearn base regressors which can be turned 'on' or 'off' with the following optional keyword arguments:

- lasso= 'on' (default) or 'off'        uses LassoCV
- ridge= 'on' (default) or 'off'        uses RidgeCV
- elastic= 'on' (default) or 'off'      uses ElasticNetCV
- sgd= 'on' (default) or 'off'          uses SGDRegressor
- knr= 'on' (default) or 'off'          uses KNeighborsRegressor
- svr= 'on' (default) or 'off'          uses SVR(kernel='rbf')
- mlp= 'on' or 'off' (default)          uses MLPRegressor
- gbr= 'on' (default) or 'off'          uses GradientBoostingRegressor
- tree= 'on' (default) or 'off'         uses DecisionTreeRegressor
- forest= 'on' (default) or 'off'       uses RandomForestRegressor

The meta-model may be specifed using the optional keyword argument meta:

- meta= 'linear', 'lasso', 'ridge' (default), or 'elastic' 

where 'linear' uses LinearRegression, 'lasso' uses LassoCV,  'ridge' uses RidgeCV (default), and 'elastic' uses ElasticNetCV as the meta-model for the final estimator.

Stacking regression is an **ensemble** machine learning technique that improves predictive accuracy by combining multiple base regression models. Instead of selecting a single best model, stacking leverages multiple models to generate a more robust final prediction.

**How Stacking Regression Works**

Base Regressors (Level 0 Models):  
- Several regression models (e.g., Lasso, Ridge, Elastic, DecisionTree, RandomForest, etc.) are trained independently on the dataset.
- Each model learns different aspects of the data.

Meta-Model (Level 1 Model):  
- A separate model (e.g. linear regression, Lasso, or Elastic) is trained to **learn from the outputs of base models**.
- It assigns strength weights to each base model’s predictions, determining which models contribute the most to final accuracy.

Final Prediction: 
- The meta-model makes a final prediction based on the base models' combined outputs.

**Advantages of Stacking Regression**

– Utilizes the strengths of multiple models  
– Helps mitigate overfitting or underfitting 
– Works well for datasets with nonlinear relationships

### Stepwise

The **stepwise** function in the PyMLR module is intended to be a Python alternative to the R stepAIC function. The **stepwise** function has the option to use either forward selection (default), backward selection, or all subsets of possible combinations for the optimum set of predictor variables as follows using statsmodels OLS:

- Forward selection (default) starts with no predictors and adds predictors as long as it improves the model (reduces AIC or BIC, or increases adjusted rsquared) 
- Backward selection starts with all predictors and removes predictors as long as it improves the model (reduces AIC or BIC, or increases adjusted rsquared)
- All subsets of possible combinations of predictor features to find the best of all possible models (up to 20 candidate predictors)

Either the AIC, BIC, or adjusted rsquared may be used as the criterion with forward, backward, or all subsets. In addition, there is an option to find all features with p-values less than a signficance threshold through backward elimination based only on the p-values of the coefficients. The stepwise algorithm also has the option (default) to remove any non-signficant predictors after either a forward, backward, or all subsets search using the AIC, BIC, or adjusted rsquared criterion. 

The choice between Lasso, Ridge, Elastic Net, Stacking, or Stepwise regression depends on the specific context and requirements of the analysis. Stepwise regression is widely used ([e.g. Murtaugh, 2009](https://doi.org/10.1111/j.1461-0248.2009.01361.x)), but often criticized ([e.g. Flom and Cassell, 2007](https://www.lexjansen.com/pnwsug/2008/DavidCassell-StoppingStepwise.pdf)). Machine Learning methods such as Lasso, Ridge, and Elastic Net regression are generally preferred for their efficiency and ability to handle large datasets without overfitting. However, Stepwise regression can be more suitable for exploratory data analysis and when the goal is to identify the most influential predictors. Ultimately, the best choice depends on the data characteristics and the researcher's objectives. Stacking regression is an ensemble machine learning technique that improves predictive accuracy by combining multiple base regression models.

### Comparison of Stepwise, Lasso, Ridge, and Elastic Net

- Feature selection: Lasso performs explicit feature selection by setting some coefficients to zero, while Ridge shrinks coefficients but retains all predictors. Elastic Net balances the regularization methods of Lasso and Ridge and is able to do feature selection. Stepwise regression also performs feature selection but can be less stable than Lasso. 
- Regularization: Lasso, Ridge, and Elastic Net are regularization techniques that prevent overfitting, but they do so differently. Lasso is more likely to produce sparse models, while Ridge is more likely to shrink coefficients smoothly. Elastic Net balances the capabilities of Lasso and Ridge.
- Computational cost: Stepwise regression can be computationally expensive, especially for large datasets. Lasso, Ridge, and Elastic Net can be solved more efficiently using optimization algorithms. 

### AIC vs BIC

Using AIC as the criterion is the default in the **stepwise** fuction. The user also has the option to use the BIC as the criterion instead. AIC is considered to be a useful critierion in stepwise regression. However, BIC is generally considered to be better than AIC for several reasons:

- Penalty for Complexity: BIC penalizes models more heavily for the number of parameters, making it more conservative and less likely to overfit, especially with larger sample sizes.
- Model Selection: BIC is particularly useful when the sample size is large, as it encourages simpler models that are less likely to capture noise.
- Model Recovery: Studies suggest that BIC tends to recover the true model more effectively than AIC, particularly in scenarios where the sample size is large.
While both criteria are useful for model selection, BIC is often preferred for its stricter criteria, which helps in avoiding overfitting and improving model interpretability

### Acceptable VIF as the target for Ridge regression

Ridge regression reduces the Variance Inflation Factors of the features by adding a penalty term to the ordinary least squares regression. The magnitude of the penalty term is related to the regularization paramter (alpha) and the sum of the squared coefficients. At very low values of alpha there is negligible penalty and the Ridge regression results are practically the same as OLS. As the alpha value is increased, the penalty increases and the VIF values of the features decreases, which decreases the magnitude of the coefficients to mitigates the problem of multicollinearity.

Cross-validated ridge regression (e.g. using RidgeCV) does not always result in acceptable multicollinearity as indicated by VIF. While cross-validation helps in fine-tuning the regression coefficients, it does not always result in VIF values close to 1. Ideally the VIF of all features should be as close as possibe to VIF=1. This can be achieved using a trial and error method of evaluating the VIF values of the model features over a range of alpha values. 

The **ridge** function in PyMLR includes an algorithm (RidgeVIF) to find the model with the optimum value of alpha that will result in VIF values as close as possible to a user-specified target VIF (default target VIF=1.0). This assures that there will be acceptable multicollinearity for all features. The trade-off is that this algorithm reduces the model coefficients such that the target VIF will be achieved. The user has the option to specify any target for VIF to explore the balance between VIF and coefficient values.  

## Installation for Python or Jupyter Notebook

The PyMLR module require that you have already installed numpy, pandas, scikit-learn, tabulate, matplotlib, seaborn, statsmodels, xgboost, lightgbm, mlxtend, and optuna packages. In addition, the auto-optimization functions, because they use optuna, require that you have already installed and enabled ipywidgets in your Python environment if you are working in a Jupyter notebook. 

If you have not already installed PyMLR, enter the following with pip or !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/PyMLR.git
```

if you are upgrading from a previous installation of PyMLR, enter the following with pip pr !pip in your notebook or terminal:<br>
```
pip install git+https://github.com/gjpelletier/PyMLR.git --upgrade
```

Next import the main functions from the PyMLR module as needed using the following code in your notebook or python code:<br>
```
from PyMLR import logistic, logistic_auto
from PyMLR import lasso, ridge, elastic
from PyMLR import xgb, xgb_auto
from PyMLR import catboost, catboost_auto,
from PyMLR import gbr, gbr_auto,
from PyMLR import forest, forest_auto,
from PyMLR import svr, svr_auto,
from PyMLR import knn, knn_auto,
from PyMLR import lgbm, stacking, stepwise
from PyMLR import show_optuna
from PyMLR import plot_predictions_from_test
from PyMLR import plot_logistic_results_test
```


---

## Example 1. Automatic cross-validated optimization of multinomial logistic regression hyperparameters

In this example we will use **logistic_auto** to automatically optimize the LogisticRegression hyperparameters using cross-validation. The logistic_auto function encodes and standardizes the X values by default. 

Run the following code:
```
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True, as_frame=True)

# optional kwargs for logistic_auto
kwargs = {
    # user-specified params
    'random_state': 42,                        # random seed
    'n_trials': 50,                            # number of optuna trials
    # [min,max] model params that are optimized by optuna
    'C': [1e-4, 10],                           # Inverse regularization
    # categorical model params optimized by optuna
    'solver': ['liblinear', 'lbfgs', 'saga'],  # optimization algorithm
    'penalty': ['l1', 'l2'],                   # norm of the penalty
}    

from PyMLR import logistic_auto
train_model, train_output = logistic_auto(X, y, **kwargs)
```

Running the code above produces the following display of regression statistics:
```
Running optuna to find best parameters, could take a few minutes, please wait...
Fitting LogisticRegression model with best parameters, please wait ...

LogisticRegression goodness of fit to training data in model_outputs['stats']:

| Statistic          |   LogisticRegression |
|:-------------------|---------------------:|
| mcfadden_pseudo_r2 |            0.971809  |
| accuracy           |            0.994435  |
| f1_score           |            0.994415  |
| precision          |            0.994468  |
| recall             |            0.994386  |
| log_loss           |            0.0649097 |
| n_classes          |           10         |
| n_samples          |         1797         |


Best-fit accuracy of CV test data: 0.9722

Done
Time elapsed: 120.75 sec
```

![LogisticRegression_confusion_matrix](https://github.com/user-attachments/assets/7688dcc5-6949-403e-bcdb-13e5bfd4b13a)
![LogisticRegression_ROC_curve](https://github.com/user-attachments/assets/defee605-e62a-4882-8665-fed1cc38cc1f)


---

## Example 2. Automatic cross-validated optimization of XGBoost hyperparameters

In this example we will use **xgb_auto** to automatically optimize the XGBoost hyperparameters using cross-validation. The xgb_auto function standardizes the X values by default. 

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the xgb_auto function in the PyMLR module
from PyMLR import xgb_auto
model_objects, model_outputs = xgb_auto(X, y,
    # [min, max] range of params that are optimized by optuna
    learning_rate= [0.01, 0.3],       # step size shrinkage (also called eta)
    max_depth= [3, 10],               # maximum depth of a tree
    min_child_weight= [1, 10],        # minimum sum of instance weight (hessian) needed in a child
    subsample= [0.5, 1],              # fraction of samples used for training each tree
    colsample_bytree= [0.5, 1],       # fraction of features used for each tree
    gamma= [0, 10],                   # minimum loss reduction to make a split
    reg_lambda= [0, 10],              # L2 regularization term on weights
    alpha= [0, 10],                   # L1 regularization term on weights
    n_estimators= [100, 1000]         # number of boosting rounds (trees)
    )
```

Running the code above produces the following display of regression statistics:
```
Running optuna to find best parameters, could take a few minutes, please wait...
Fitting XGBRegressor model with best parameters, please wait ...
XGBRegressor statistics of fitted model in model_outputs['stats']:

| Statistic   |   XGBRegressor |
|:------------|---------------:|
| r-squared   |       0.751082 |
| RMSE        |      38.4195   |
| n_samples   |     442        |

Done
Time elapsed: 335.19 sec
```

![XGBRegressor_predictions](https://github.com/user-attachments/assets/6b7b2716-d80a-4607-bd5e-fc0ebe0a61ad)


---

## Example 3. Automatic cross-validated optimization of CatBoost hyperparameters

In this example we will use **catboost_auto** to automatically optimize the CatBoost hyperparameters using cross-validation. The catboost_auto function standardizes the X values by default. 

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the catboost_auto function in the PyMLR module
from PyMLR import catboost_auto
model_objects, model_outputs = catboost_auto(X, y,
    # [min, max] range of params that are optimized by optuna
    random_state= 42,                 # random seed for reproducibility
    n_trials= 50,                     # number of optuna trials
    learning_rate= [0.01, 0.3],       # Balances step size in gradient updates.
    depth= [4, 10],                   # Controls tree depth
    iterations= [100, 3000],          # Number of boosting iterations
    l2_leaf_reg= [1, 10],             # Regularization strength       
    random_strength= [0, 1],          # Adds noise for diversity
    bagging_temperature= [0.1, 1.0],  # Controls randomness in sampling
    border_count= [32, 255],          # Number of bins for feature discretization
    min_data_in_leaf= [1, 100],       # Minimum samples per leaf         
    max_bin= [64, 255],               # Number of bins for feature quantization
    )
```

Running the code above produces the following display of regression statistics:
```
Running optuna to find best parameters, could take a few minutes, please wait...
Fitting CatBoostRegressor model with best parameters, please wait ...
CatBoostRegressor statistics of fitted model in model_outputs['stats']:

| Statistic   |   CatBoostRegressor |
|:------------|--------------------:|
| r-squared   |            0.740263 |
| RMSE        |           39.2455   |
| n_samples   |          442        |

Done
Time elapsed: 859.84 sec
```

![CatBoostRegressor_predictions](https://github.com/user-attachments/assets/2daf64fe-8a16-49a7-b3a6-d623cea1501d)


---

## Example 4. Automatic cross-validated optimization of GradientBoostingRegressor hyperparameters

In this example we will use **gbr_auto** to automatically optimize the GradientBoostingRegressor hyperparameters using cross-validation. The gbr_auto function standardizes the X values by default. 

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the xgb_auto function in the PyMLR module
from PyMLR import gbr_auto
model_objects, model_outputs = gbr_auto(X, y,
    # [min, max] range of params that are optimized by optuna
    random_state= 42,                      # random seed for reproducibility
    n_trials= 50,                          # number of optuna trials
    learning_rate= [0.01, 0.3],            # shrinks the contribution of each tree
    n_estimators= [100, 1000],             # number of boosting stages (trees)
    max_depth= [3, 10],                    # max depth of the individual regression estimators
    min_samples_split= [2, 10],            # min samples to split an internal node
    min_samples_leaf= [1, 10],             # min samples to be a leaf node
    subsample= [0.5, 1.0],                 # fraction of samples to fit each tree
    max_features= [None, "sqrt", "log2"]   # number of features for best split
)
```

Running the code above produces the following display of regression statistics:
```
Running optuna to find best parameters, could take a few minutes, please wait...
Fitting GradientBoostingRegressor model with best parameters, please wait ...
GradientBoostingRegressor statistics of fitted model in model_outputs['stats']:

| Statistic   |   GradientBoostingRegressor |
|:------------|----------------------------:|
| r-squared   |                    0.850365 |
| RMSE        |                   29.7879   |
| n_samples   |                  442        |

Done
Time elapsed: 97.19 sec
```

![GradientBoostingRegressor_predictions](https://github.com/user-attachments/assets/328be74b-a6e0-4770-b31d-4118ca1279d7)


---

## Example 5. Automatic cross-validated optimization of SVR hyperparameters

In this example we will use **svr_auto** to automatically optimize the XGBoost hyperparameters using cross-validation. The svr_auto function standardizes the X values by default. 

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the svr_auto function in the PyMLR module
from PyMLR import svr_auto
model_objects, model_outputs = svr_auto(X, y,
    # [min, max] range of params that are optimized by optuna
    C= [0.1, 1000],           # C Regularization parameter
    epsilon= [0.01, 1.0],     # Epsilon in the epsilon-SVR model
    gamma= 'scale'            # 'scale', 'auto', or [min, max]
    )
```

Running the code above produces the following display of regression statistics:
```
Running optuna to find best parameters, could take a few minutes, please wait...
Fitting SVR model with best parameters, please wait ...
SVR statistics of fitted model in model_outputs['stats']:

| Statistic   |        SVR |
|:------------|-----------:|
| r-squared   |   0.564601 |
| RMSE        |  50.812    |
| n_samples   | 442        |

Done
Time elapsed: 8.63 sec
```

![SVR_predictions](https://github.com/user-attachments/assets/b4a15cd6-47d7-49d2-9ec4-d6bf677a14c0)


---

## Example 6. Use Lasso regression to analyze diabetes data

In this example we will use Lasso regression to analyze the diabetes data available from sklearn. The lasso function uses the sklearn.processing StandardScaler to standardize the X values by default. Then the lasso function uses the standardized X values to find each of the best fit models using LassoCV, LassoLarsCV, LassoLarsIC using AIC, and LassoLarsIC using BIC.

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the lasso function in the PyMLR module
from PyMLR import lasso
model_objects, model_outputs = lasso(X, y)
```

Running the code above produces the following display of output tables with regression statistics and best-fit coefficients for each model (LassoCV, LassoLarsCV, LassoLarsIC using AIC, LassoLarsIC using BIC):
```
Fitting Lasso regression models, please wait ...

Lasso regression statistics of best models in model_outputs['stats']:

| Statistic          |         LassoCV |     LassoLarsCV |    LassoLarsAIC |    LassoLarsBIC |
|:-------------------|----------------:|----------------:|----------------:|----------------:|
| alpha              |     1.11865     |     1.10767     |     0.950407    |     0.950407    |
| r-squared          |     0.512957    |     0.512989    |     0.51341     |     0.51341     |
| adjusted r-squared |     0.503959    |     0.503991    |     0.50442     |     0.50442     |
| n_samples          |   442           |   442           |   442           |   442           |
| df residuals       |   434           |   434           |   434           |   434           |
| df model           |     7           |     7           |     7           |     7           |
| F-statistic        |    65.2989      |    65.3073      |    65.4173      |    65.4173      |
| Prob (F-statistic) |     1.11022e-16 |     1.11022e-16 |     1.11022e-16 |     1.11022e-16 |
| RMSE               |    53.7411      |    53.7393      |    53.7161      |    53.7161      |
| Log-Likelihood     | -2388.18        | -2388.16        | -2387.97        | -2387.97        |
| AIC                |  4792.36        |  4792.33        |  4790           |  4790           |
| BIC                |  4825.09        |  4825.06        |  4818.64        |  4818.64        |

Coefficients of best models in model_outputs['popt']:

| Feature   |   LassoCV |   LassoLarsCV |   LassoLarsAIC |   LassoLarsBIC |
|:----------|----------:|--------------:|---------------:|---------------:|
| const     | 152.133   |     152.133   |      152.133   |      152.133   |
| age       |  -0       |       0       |        0       |        0       |
| sex       |  -9.11162 |      -9.13079 |       -9.40617 |       -9.40617 |
| bmi       |  24.8066  |      24.809   |       24.8419  |       24.8419  |
| bp        |  13.9806  |      13.9909  |       14.1342  |       14.1342  |
| s1        |  -4.58713 |      -4.61047 |       -4.94418 |       -4.94418 |
| s2        |  -0       |       0       |        0       |        0       |
| s3        | -10.5553  |     -10.5615  |      -10.651   |      -10.651   |
| s4        |   0       |       0       |        0       |        0       |
| s5        |  24.2697  |      24.2839  |       24.4841  |       24.4841  |
| s6        |   2.45869 |       2.46804 |        2.6051  |        2.6051  |

Variance Inflation Factors model_outputs['vif']:
Note: VIF>5 indicates excessive collinearity

| Feature   |   LassoCV |   LassoLarsCV |   LassoLarsAIC |   LassoLarsBIC |
|:----------|----------:|--------------:|---------------:|---------------:|
| const     |   1       |       1       |        1       |        1       |
| age       | nan       |     nan       |      nan       |      nan       |
| sex       |   1.25594 |       1.25594 |        1.25594 |        1.25594 |
| bmi       |   1.49419 |       1.49419 |        1.49419 |        1.49419 |
| bp        |   1.39388 |       1.39388 |        1.39388 |        1.39388 |
| s1        |   1.58065 |       1.58065 |        1.58065 |        1.58065 |
| s2        | nan       |     nan       |      nan       |      nan       |
| s3        |   1.66245 |       1.66245 |        1.66245 |        1.66245 |
| s4        | nan       |     nan       |      nan       |      nan       |
| s5        |   2.07745 |       2.07745 |        2.07745 |        2.07745 |
| s6        |   1.45162 |       1.45162 |        1.45162 |        1.45162 |

Done
Time elapsed: 1.79 sec
```

![Lasso_predictions_vs_actual](https://github.com/user-attachments/assets/e60dc714-62f8-4c11-99eb-9d242be769bd)
![Lasso_predictions_vs_residuals](https://github.com/user-attachments/assets/2963604a-95b9-4cb7-8556-12a9f488bb21)


---

## Example 7. Use Ridge regression to analyze diabetes data

In this example we will use Ridge regression to analyze the diabetes data available from sklearn. The ridge function uses the sklearn.processing StandardScaler to standardize the X values by default. Then the ridge function uses the standardized X values to find each of the best fit models using RidgeCV, and Ridge using AIC, BIC, and VIF for optimization.

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the ridge function in the PyMLR module
from PyMLR import ridge
model_objects, model_outputs = ridge(X, y)
```

Running the code above produces the following output display of the best fit model:
```
Fitting Ridge regression models, please wait ...

Ridge regression statistics of best models in model_outputs['stats']:

| Statistic          |         RidgeCV |        RidgeVIF |
|:-------------------|----------------:|----------------:|
| alpha              |     1.87382     |    53.367       |
| r-squared          |     0.51733     |     0.511683    |
| adjusted r-squared |     0.504982    |     0.499191    |
| n_samples          |   442           |   442           |
| df residuals       |   431           |   431           |
| df model           |    10           |    10           |
| F-statistic        |    46.1949      |    45.1624      |
| Prob (F-statistic) |     1.11022e-16 |     1.11022e-16 |
| RMSE               |    53.4993      |    53.8114      |
| Log-Likelihood     | -2386.18        | -2388.76        |
| AIC                |  4794.37        |  4799.51        |
| BIC                |  4839.37        |  4844.51        |

Coefficients of best models in model_outputs['popt']:

| Feature   |    RidgeCV |   RidgeVIF |
|:----------|-----------:|-----------:|
| const     | 152.133    | 152.133    |
| age       |  -0.402061 |   0.132284 |
| sex       | -11.2811   |  -9.59744  |
| bmi       |  24.7848   |  22.9527   |
| bp        |  15.3345   |  14.173    |
| s1        | -25.6503   |  -3.44889  |
| s2        |  13.1337   |  -3.5854   |
| s3        |  -0.47979  |  -9.04181  |
| s4        |   7.01112  |   5.55365  |
| s5        |  31.1369   |  20.5852   |
| s6        |   3.30159  |   4.25116  |

Variance Inflation Factors model_outputs['vif']:
Note: VIF>5 indicates excessive collinearity

| Feature   |   RidgeCV |   RidgeVIF |
|:----------|----------:|-----------:|
| age       |   1.20293 |   0.904903 |
| sex       |   1.26072 |   0.918176 |
| bmi       |   1.48134 |   1.02315  |
| bp        |   1.43679 |   1.00172  |
| s1        |  26.672   |   0.532883 |
| s2        |  18.5444  |   0.7917   |
| s3        |   8.59212 |   1.00853  |
| s4        |   7.59308 |   1.40875  |
| s5        |   5.49131 |   1.06848  |
| s6        |   1.46335 |   1.0328   |

Done
Time elapsed: 1.75 sec
```

![Ridge_predicted_vs_actual](https://github.com/user-attachments/assets/d282e056-b0ea-4661-953d-c45c4b8993a3)
![Ridge_predicted_vs_residuals](https://github.com/user-attachments/assets/98ec5184-6829-4efd-9149-6b21a2289b4c)


---

## Example 8. Use Elastic Net regression to analyze diabetes data

In this example we will use Elastic Net regression to analyze the diabetes data available from sklearn. The elastic function uses the sklearn.processing StandardScaler to standardize the X values by default. Then the elastic function uses the standardized X values to find the best fit model using ElasticNetCV, and Ridge using MSE as the scoring criterion.

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the elastic function in the PyMLR module
from PyMLR import elastic
model_objects, model_outputs = elastic(X, y)
```

Running the code above produces the following output display of the best fit model:

```
Fitting Elastic Net regression model, please wait ...

ElasticNetCV regression statistics of best model in model_outputs['stats']:

| Statistic          |    ElasticNetCV |
|:-------------------|----------------:|
| alpha              |     1.11865     |
| r-squared          |     0.512957    |
| adjusted r-squared |     0.503959    |
| n_samples          |   442           |
| df residuals       |   434           |
| df model           |     7           |
| F-statistic        |    65.2989      |
| Prob (F-statistic) |     1.11022e-16 |
| RMSE               |    53.7411      |
| Log-Likelihood     | -2388.18        |
| AIC                |  4792.36        |
| BIC                |  4825.09        |
| L1-ratio           |     1           |

Coefficients of best model in model_outputs['popt']:

| Feature   |   ElasticNetCV |
|:----------|---------------:|
| const     |      152.133   |
| age       |       -0       |
| sex       |       -9.11162 |
| bmi       |       24.8066  |
| bp        |       13.9806  |
| s1        |       -4.58713 |
| s2        |       -0       |
| s3        |      -10.5553  |
| s4        |        0       |
| s5        |       24.2697  |
| s6        |        2.45869 |

Variance Inflation Factors model_outputs['vif']:
Note: VIF>5 indicates excessive collinearity

| Feature   |   ElasticNetCV |
|:----------|---------------:|
| const     |        1       |
| age       |      nan       |
| sex       |        1.25594 |
| bmi       |        1.49419 |
| bp        |        1.39388 |
| s1        |        1.58065 |
| s2        |      nan       |
| s3        |        1.66245 |
| s4        |      nan       |
| s5        |        2.07745 |
| s6        |        1.45162 |

Done
Time elapsed: 8.83 sec
```

![ElasticNetCV_predictions](https://github.com/user-attachments/assets/3dc966a6-56c9-4044-8e86-55c3693a6b60)


---

## Example 9. Use Stacking regression to analyze diabetes data

In this example we will use Stacking regression to analyze the diabetes data available from sklearn. The **stacking** function uses the sklearn StackingRegressor with an ensemble of models. The **stacking** function standardizes the X values by default. 

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the stacking function in the PyMLR module
from PyMLR import stacking
model_objects, model_outputs = stacking(X, y)
```

Running the code above produces the following output display of the best fit model:
```
Fitting StackingRegressor models, please wait ...

StackingRegressor statistics of fitted ensemble model in model_outputs['stats']:

| Statistic   |   StackingRegressor |
|:------------|--------------------:|
| r-squared   |            0.603124 |
| RMSE        |           48.5121   |
| n_samples   |          442        |

Meta-model coefficients of base_regressors in model_outputs['meta_params']:

- positive intercept suggests base models under-predict target
- negative intercept suggests base models over-predict target
- positive coefficients have high importance
- coefficients near zero have low importance
- negative coefficients have counteracting importance

| Coefficient               |   StackingRegressor |
|:--------------------------|--------------------:|
| Intercept                 |          -1.16457   |
| LassoCV                   |           1.70732   |
| RidgeCV                   |           0.409702  |
| ElasticNetCV              |          -1.59604   |
| SGDRegressor              |           0.195497  |
| KNeighborsRegressor       |           0.134988  |
| GradientBoostingRegressor |           0.0692347 |
| DecisionTreeRegressor     |          -0.0585657 |
| RandomForestRegressor     |           0.152548  |

Done
Time elapsed: 13.66 sec
```

![StackingRegressor_predictions](https://github.com/user-attachments/assets/14a9e6d7-55e9-4a41-834d-9383d10a04b7)


---

## Example 10. Use Stepwise regression to analyze diabetes data

In this example we will use Stepwise regression to analyze the diabetes data available from sklearn.

We will use the following optional arguments:

- criterion='bic' to use BIC as the criterion
- direction='all' to search for the best model from all possible combinations of features
- standardize='on' to use the sklearn.processing StandardScaler to standardize the X values

Run the following code:
```
# Read X and y from the sklearn diabetes data set
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Use the stepwise function in the PyMLR module
from PyMLR import stepwise
model_object, model_output = stepwise(X, y, 
    criterion='bic', direction='all', standardize='on')
```

Running the code above produces the following output display of the best fit model:
```
Best 10 subsets of features in model_outputs['step_features']:

|   Rank |     AIC |     BIC |   rsq_adj | Features                               |
|-------:|--------:|--------:|----------:|:---------------------------------------|
|      0 | 4792.26 | 4816.81 |  0.502997 | ['sex' 'bmi' 'bp' 's3' 's5']           |
|      1 | 4792.26 | 4816.81 |  0.502997 | ['sex' 'bmi' 'bp' 's3' 's5']           |
|      2 | 4788.6  | 4817.24 |  0.508193 | ['sex' 'bmi' 'bp' 's1' 's2' 's5']      |
|      3 | 4789.92 | 4818.56 |  0.506728 | ['sex' 'bmi' 'bp' 's1' 's4' 's5']      |
|      4 | 4790.12 | 4818.76 |  0.5065   | ['sex' 'bmi' 'bp' 's1' 's3' 's5']      |
|      5 | 4791.09 | 4819.73 |  0.505419 | ['sex' 'bmi' 'bp' 's2' 's3' 's5']      |
|      6 | 4793.18 | 4821.82 |  0.503078 | ['sex' 'bmi' 'bp' 's3' 's4' 's5']      |
|      7 | 4789.32 | 4822.05 |  0.508488 | ['sex' 'bmi' 'bp' 's1' 's2' 's4' 's5'] |
|      8 | 4789.37 | 4822.1  |  0.508429 | ['sex' 'bmi' 'bp' 's1' 's2' 's5' 's6'] |
|      9 | 4793.56 | 4822.2  |  0.502647 | ['sex' 'bmi' 'bp' 's3' 's5' 's6']      |

Best of all possible models after removing insignficant features if any:
Best features:  ['sex', 'bmi', 'bp', 's3', 's5'] 

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 target   R-squared:                       0.509
Model:                            OLS   Adj. R-squared:                  0.503
Method:                 Least Squares   F-statistic:                     90.26
Date:                Tue, 27 May 2025   Prob (F-statistic):           4.75e-65
Time:                        17:53:07   Log-Likelihood:                -2390.1
No. Observations:                 442   AIC:                             4792.
Df Residuals:                     436   BIC:                             4817.
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        152.1335      2.585     58.849      0.000     147.053     157.214
sex          -11.2146      2.876     -3.899      0.000     -16.868      -5.562
bmi           24.9036      3.106      8.019      0.000      18.800      31.008
bp            15.5172      3.001      5.171      0.000       9.620      21.415
s3           -13.7518      3.122     -4.404      0.000     -19.889      -7.615
s5            22.5597      3.124      7.221      0.000      16.419      28.700
==============================================================================
Omnibus:                        2.465   Durbin-Watson:                   1.990
Prob(Omnibus):                  0.291   Jarque-Bera (JB):                2.099
Skew:                           0.051   Prob(JB):                        0.350
Kurtosis:                       2.678   Cond. No.                         2.33
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Variance Inflation Factors of selected_features:
Note: VIF>5 indicates excessive collinearity

| Feature   |     VIF |
|:----------|--------:|
| const     | 1       |
| sex       | 1.23788 |
| bmi       | 1.44328 |
| bp        | 1.34724 |
| s3        | 1.45888 |
| s5        | 1.46057 |
```



