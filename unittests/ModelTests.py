#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
       
    def test_01_train(self):
        """
        test the train functionality
        """

        ## data ingestion
        df = load_data()

        df_processed = get_preprocessor(df)

        ts1, ts2 = prepare_timeseries(df_processed)

        ## train the model
        model_train(ts1, "", test=False)
        self.assertTrue(os.path.exists(os.path.join("models", "test.joblib")))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        model = model_load(test=True)
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

       
    # def test_03_predict(self):
    #     """
    #     test the predict function input
    #     """

    #     ## data ingestion
    #     df = load_data()

    #     df_processed = get_preprocessor(df)

    #     ts1, ts2 = prepare_timeseries(df_processed)
    #     target, features = model_train(ts1, "", test=False)

    #     ## load model first
    #     model = model_load(test=True)
    
    #     ## ensure that a list can be passed
    #     query = pd.DataFrame({'country': [""],
    #                       'date': ["2019-08-01"]
    #             })

    #     result = model_predict(query, target, features, "", "2019-08-01", model, 30, test=True)
    #     y_pred = result['y_pred']
    #     self.assertTrue(y_pred[0] in [0,1])

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
