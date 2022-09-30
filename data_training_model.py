"""
Doc String:
Entry Point for Model training.
After the data is Validated and Pre-Processed, it needs to be trained.
Written by: Anshul Mehta

"""
# Here we are importing from the diretories an are importing files
from tkinter import EXCEPTION
from sklearn.model_selection import train_test_split
# from code.data_ingestion.data_loader import Data_Getter
from data_ingestion import data_loader
from data_preprocessing import clustering
from data_preprocessing import preprocessing
from application_logging import logger
from file_operations import file_methods

# Creating a Logging Object for Model Training

class train_model:
    #  Creating a constructor
    def __init__(self):
        self.log_writer=logger.app_logger()
        # This will create a file start logging in the data
        self.file_object= open("Training_Logs/ModelTrainingLog.txt", 'a+')
    def data_training_model(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        """
        Function docstrin:
        Get the data from the data_loader
        Preprocess the data: Cluster and Preprocess the Data
        Solit the Data into training and test sets 
        Find the best model for each of the clusters
        Then save the best models

        """
        try:
            # The complete training Process
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.getData()

            # Preprocessing on the data
            

        except Exception:
            # This means that the training was unsuccessful
            # Log that as well
            self.log_writer.log(self.file_object,"The training stopped")
            self.file_object.close()
            raise Exception()
    

