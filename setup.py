from setuptools import setup
import sys
sys.path.insert(0, ".")
from PyMLR import __version__

setup(
    name='PyMLR',
    version=__version__,
    author='Greg Pelletier',
    py_modules=['PyMLR'], 
    install_requires=[
        'numpy','pandas','statsmodels','seaborn',
        'scikit-learn','tabulate','matplotlib',
        'xgboost','lightgbm','mlxtend','optuna',
        'shap'],
)

