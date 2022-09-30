import pickle 
import os
import shutil
# Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script. # bytes that can be written to any file-like object.
class File_Operation:
    """
    This class serves 3 purpose
    1. Model Saving
    2. Model Loading 
    AND
    3. Select the correct model for the correct cluster
    """
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object
        self.model_directory="/models"

    # Method to save the model
    def save_model(self,model,filename):
        self.logger_object.log(self.file_object,"Entered the Save Model method")

        try:
            # Create a separate directory for each of the cluster
            path=os.path.join(self.model_directory,filename)
            # Remove the previously exisitng models for each cluster, if any
            if os.path.isdir(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            # Else make a directory anway
            else:
                os.makedirs(path)
            # .sav is a generic file extension for saving Models and Progress
            with open(path +'/' + filename+'.sav','wb') as f:
                # Save the model in the file
                pickle.dump(model,f)
            self.logger_object.log(self.file_object,"Saved the Mode with the "+filename+"of File Operation")
            return 'success'

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Model File '+filename+' could not be saved. Exited the save_model method of the Model_Finder class')

            raise Exception()

    def load_model(self,filename):
        self.logger_object.log(self.file_object,"Entered the Load Model method in the File Class Operations Class")
    
        try:
            with open(self.model_directory + filename + '/' + filename + '.sav','rb') as f:
                self.logger_object.log(self.file_object,'Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
                return pickle.load(f)

        except Exception as e:
            self.logger_object.log(self.file_object,"Not able to load  the model and encountered Exception"+str(e))
            self.logger_object.log(self.file_object,"Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class")
            raise Exception()
    
    def select_correct_model(self,cluster_number):
        self.logger_object.log(self.file_object,"Entered the Selecting model method")

        try:
            self.cluster_number=cluster_number
            self.folder__name=self.model_directory
            self.list_of_model_files=[]
            self.list_of_files=os.listdir(self.folder__name)
            for self.file in self.list_of_files:
                try:
                    # Make sure to understand this step
                    if(self.file.index(str( self.cluster_number))!=-1):
                        self.model_name=self.file
                except:
                    continue
            self.model_name=self.model_name.split('.')[0]
            self.logger_object.log(self.file_object,'Exited the find_correct_model_file method of the Model_Finder class')
            return self.model_name

        except Exception as e:

            self.logger_object.log(self.file_object,
                                   'Exception occured in find_correct_model_file method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Exited the find_correct_model_file method of the Model_Finder class with Failure')
            raise Exception()
    