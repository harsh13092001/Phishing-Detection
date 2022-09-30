"""
This class is used to Preprocess the data
Finding Missing Values, Imputing Data, Over and Under Sampling and making the data palatable for the Model
"""
import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
class Preprocessor:
    """
        Class that will transform the data and make it ready for training.
    """

    def __init__(self,logger_object,file_object):
        self.file_object=file_object
        self.logger_object=logger_object

    def remove_columns(self,data,columns):
        """
        The method removes the specified columns and returns the Df with the remaining columns 
        """
        self.data=data
        self.columns=columns
        self.logger_object.log(self.file_object,"Entered the Remove columns method in the Preprocessing class")
        try:
            # drop the labels specified in the columns
            self.new_data=self.data.drop(labels=self.columns, axis=1) 
            self.logger_object.log(self.file_object,'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.new_data

        except Exception as e:
            self.logger_object.log(self.file_object,"Not able to remove the columns from the dataframe with Exception"+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
            Method Name: separate_label_feature
            Description: This method separates the features and a Label Coulmns.
            Output: Returns two separate Dataframes, one containing features and the other containing Labels .
        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()
    def dropUnnecessaryColumns(self,data,columnNameList):
      
        data = data.drop(columnNameList,axis=1)
        return data
    def replaceInvalidValuesWithNull(self,data):
        """
            This method will replace all Invalid values with Null so that Null values can 
            be imputed in the subsequent steps 
        """
        for column in data.columns:
            # Check where the char '?' is being found
            count = data[column][data[column] == '?'].count()
            if count!=0:
                data[column]=data[column].replace('?',np.nan)
        
        return data
    
    def is_null_present(self,data):
        """
        This method checks all the columns in the DF and then returns true with a list of cols that 
        contain Null or return false
        """
        self.logger_object.log(self.file_object,"Entered the Is_Null_Present method to check columns in Preprocessing class")
        self.cols_with_missing_vals=[]
        self.null_present=False
        self.cols=data.columns
        try:
            # This null counts returns a List of Columns and their null counts: Column names are in the Index
            self.null_counts=data.isna().sum() # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                    self.null_present=True
                    # Appennd that Indexed column in the cols_With_missing_values list
                    self.cols_with_missing_vals.append(self.cols[i])
            # Logs to populate which columns have missing Values
            if(self.null_present):
                # If the flag has been turned into true
                self.dataframe_with_null=pd.DataFrame()
                # The column names
                self.dataframe_with_null['columns']=data.columns
                # Count of the missing values corresponding to the coluns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv')
            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
    


        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occured in is_null_present method of the Preprocessor class"+str(e))
            self.logger_object.log(self.file_object,'Finding missing value process exited of Preprocessing class')
            raise Exception()

    def encodeCategoricalValues(self,data):
        """
            This method encodes all the categorical values in the training set 
        """
        # Dictionary method to encode the values according to their corresponding class
        data["class"] = data["class"].map({'p': 1, 'e': 2})

        for column in data.drop(['class'],axis=1).columns:
                data = pd.get_dummies(data, columns=[column])

        return data
    
    def encodeCategoricalValues(self,data):
        """
            This method encodes all the categorical values in the training set 
        """
        # Dictionary method to encode the values according to their corresponding class
        # data["class"] = data["class"].map({'p': 1, 'e': 2})

        for column in data.columns:
            data = pd.get_dummies(data, columns=[column])

        return data

    def handleImbalance(self,X,y):
        """
        This method will handle the Imbalance of the data with method like Oversampling, SMOTE etc etc.
        """
        # Try two method oversampling and SMOTE
        sm = SMOTE(random_state = 42)
        x_sampled, y_sampled = sm.fit_sample(X, y)
        return x_sampled,y_sampled
    
    def impute_missing_values(self,data,cols_with_missing_values):

        """
            This will go over all the columns and Impute them with the suitable Imputations
        """
        self.data=data
        self.cols_with_missing_values=cols_with_missing_values
        self.logger_object.log(self.file_object,"Enetred into the method for Imputing the Values") 
        try:
            self.Imputer=SimpleImputer(missing_values=np.nan)
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,"Exception while Imputing the missing values"+str(e))
            self.file_object.log(self.file_object,"Error in Imputing the Null Values, exiting the imputer method of Preprocessing class")   

    def get_columns_wtih_zero_std_deviation(self,data):
        """
            Checks if there are any columns that only have identical values and they can be omitted in classification.
            As they add no value but will increase feature dimensionality
        """
        self.logger_object.log(self.file_object, 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns=data.columns
        self.data_n = data.describe()
        self.col_to_drop=[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0): # check if standard deviation is zero
                    self.col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()


        

