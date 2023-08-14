# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        # "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
        "D:\\IT Job\\mlflow\\first_ml_flow\\winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]


    remote_server_url="https://dagshub.com/saikumar2616/first_ml_flow.mlflow"
    mlflow.set_tracking_uri(remote_server_url)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print("tracking_url_type_store is :" , tracking_url_type_store)


    logged_model = 'runs:/a7af652c54644d2bb77105a181d5c9c8/updated_model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    loaded_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/ElasticnetWineModel/1"
        )

    # Predict on a Pandas DataFrame.
    import pandas as pd
    predicted_output = loaded_model.predict(pd.DataFrame(test_x))

    print(predicted_output)