"""
Data Loader file
Class is used to get Prediction data inputted by the user and Model will be called on this data
Created by: Anshul Mehta
"""
import pandas as pd
import numpy as np
class Data_Getter:
    # Constructor
    def __init__(self,file_object,logger_object):
        # Source of thw csv file
        self.training_file='data_prediction_DB/InputFile.csv'
        self.file_object=file_object
        self.logger_object=logger_object
    def getData(self):
        """
        Method name: getData 
        Description: Gets data from the Source
        Output: Data in a Pandas dataframe
        On failure: Raise Exception
        """
        self.logger_object.log(self.file_object,"Entry into the method for sourcing data")
        try:

            # Try to get the data
            self.data=pd.read_csv(self.training_file)
            self.logger_object.log(self.file_object,"The data has been read and loaded into a Pandas Df")
            return self.data

        except Exception:
            self.logger_object.log(self.file_object,"An exception occured in the get data method")
            self.logger_object.log(self.file_object,"Data load unsuccessful and exited getData from Data_Getter class")
            raise Exception()

