"""
This class will find the best Model and tune the Hyperparameters for the models and
"""
from email.contentmanager import raw_data_manager
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
# GridSearchCV is a technique to search through the best parameter values from the given set of the grid of parameters.
class Model_Finder:
    """
    This class will find the AUC ROC Score of each of the Model and return the best Model
    """
    def __init__(self,logger_object,file_object):
        self.file_object=file_object
        self.logger_object=logger_object
        self.svc=SVC()
        self.xgb=XGBClassifier(objective='binary:logistic',n_jobs=-1)
        self.rf=RandomForestClassifier(max_depth=5,random_state=42)
        # An important hyperparameter for Adaboost is n_estimator. Often by changing the number of base models or weak learners we can adjust the accuracy of the model
        self.adaBoost=AdaBoostClassifier(n_estimators=100, random_state=0)
        self.dt=DecisionTreeClassifier(random_state=42)

    def get_best_params_from_svc(self,X_train,y_train):
        self.logger_object.log(self.file_object,"Entered the method for Finding best hyperparameters from Support Vector Classifier")
        try:
            self.param_grid_svc = {"kernel": ['rbf', 'sigmoid'],
                          "C": [0.1, 0.5, 1.0],
                          "random_state": [0, 100, 200, 300]}
            """ A General format for GridSearchCV:
                1. Instantiate a GridSearchCV object
                2. Fit the data
                3. The GridSearchCV will check for the permutations and combinations of the hyperparameters
                4. Get the best set of hyperparameters for the list specified
                5. Create a new model with these best hyperparameters obtained
                6. Fit the Data and calculate the AUC ROC Score
            """
            self.grid=GridSearchCV(SVC(),param_grid=self.param_grid_svc,cv=5,verbose=3)
            # Fitting the grid to get the best hyperparameters
            self.grid.fit(X_train,y_train)
            # Extracting the best hyperparameters
            self.kernel=self.grid.best_params_['kernel']
            self.C=self.grid.best_params_['C']
            self.random_state=self.grid.best_params_['random_state']

            # Create a model with the best obtained parameters and then fit and log and return
            self.sv_classifier = SVC(kernel=self.kernel,C=self.C,random_state=self.random_state)
            self.sv_classifier.fit(X_train,y_train)
            self.logger_object.log(self.file_object,'SVM best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_svm method of the Model_Finder class')
            
            # Return the best Model
            return self.sv_classifier


        except Exception as e:
            self.logger_object.log(self.file_object," Exception occured in finding the best hyperparameters for the Support Vector Classifier with "+str(e))
            self.logger_object.log(self.file_object,"SVM, Model Training failed, exited the SVC Mehtod")
            raise Exception()

    def get_best_params_from_xgb(self,X_train,y_train):
        self.logger_object.log(self.file_object,"Entered the method for Finding best hyperparameters from XGBoost Classifier")
        try:
            """ A General format for GridSearchCV:
                1. Instantiate a GridSearchCV object
                2. Fit the data
                3. The GridSearchCV will check for the permutations and combinations of the hyperparameters
                4. Get the best set of hyperparameters for the list specified
                5. Create a new model with these best hyperparameters obtained
                6. Fit the Data and calculate the AUC ROC Score
            """
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                "n_estimators": [100, 150], "criterion": ['gini', 'entropy'],"max_depth": range(8, 10, 1)

            } 
            self.grid=GridSearchCV(XGBClassifier(objective='binary:logistic'),param_grid=self.param_grid_xgboost,cv=5,verbose=3)
            # Fitting the grid to get the best hyperparameters
            self.grid.fit(X_train,y_train)
            # Extracting the best parameters
            self.n_estimators=self.grid.best_params_['n_estimators']
            self.criterion=self.grid.best_params_['criterion']
            self.max_depth=self.grid.best_params_['max_depth']

            # Creating a new Model with the best obtained hyperparameters
            self.xgb_classifier=XGBClassifier(objective='binary:logistic',n_estimators=self.n_estimators,criterion=self.criterion,max_depth=self.max_depth)
            # Fit the model
            self.xgb_classifier.fit(X_train,y_train)
            self.logger_object.log(self.file_object,'XGB best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_xgb method of the Model_Finder class')
            # Return the best model
            return self.xgb_classifier


        except Exception as e:
            self.logger_object.log(self.file_object," Exception occured in finding the best hyperparameters for the XGBoost Classifier with "+str(e))
            self.logger_object.log(self.file_object,"Xgb, Model Training failed, exited the XGBoost Mehtod")
            raise Exception()


    def get_best_params_from_rf(self,X_train,y_train):
        self.logger_object.log(self.file_object,"Entered the method for Finding best hyperparameters from Random Forest Classifier")
        try:
            """ A General format for GridSearchCV:
                1. Instantiate a GridSearchCV object
                2. Fit the data
                3. The GridSearchCV will check for the permutations and combinations of the hyperparameters
                4. Get the best set of hyperparameters for the list specified
                5. Create a new model with these best hyperparameters obtained
                6. Fit the Data and calculate the AUC ROC Score
            """
            self.param_grid_rf = { 
                'n_estimators': [200,300,500],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [4,5,6,7,8],
                'criterion' :['gini', 'entropy']
            }
            self.grid=GridSearchCV(RandomForestClassifier(),param_grid=self.param_grid_rf,cv=5)
            # Fitting the grid to get the best hyperparameters
            self.grid.fit(X_train,y_train)
            # Extracting the best hyperparameters
            self.n_estimators=self.grid.best_params_['n_estimators']
            self.max_features=self.grid.best_params_['max_features']
            self.max_depth=self.grid.best_params_['max_depth']
            self.criterion=self.grid.best_params_['criterion']
            # Create a new model with the best obtained hyperparameters
            self.rf_classifier=RandomForestClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,max_features=self.max_features,criterion=self.criterion)
            # Fit the data
            self.rf_classifier.fit(X_train,y_train)
            self.logger_object.log(self.file_object,'RF best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_rfc method of the Model_Finder class')
            return self.rf_classifier

        except Exception as e:
            self.logger_object.log(self.file_object," Exception occured in finding the best hyperparameters for the Random Forest Classifier with "+str(e))
            self.logger_object.log(self.file_object,"RF, Model Training failed, exited the Random Forest Classifier Mehtod")
            raise Exception()


    def get_best_params_from_ada(self,X_train,y_train):
        self.logger_object.log(self.file_object,"Entered the method for Finding best hyperparameters from Ada Boost Classifier")
        try:
            """ A General format for GridSearchCV:
                1. Instantiate a GridSearchCV object
                2. Fit the data
                3. The GridSearchCV will check for the permutations and combinations of the hyperparameters
                4. Get the best set of hyperparameters for the list specified
                5. Create a new model with these best hyperparameters obtained
                6. Fit the Data and calculate the AUC ROC Score
            """
            self.param_grid_ada={
                'n_estimators':[50,100,150,200],
                'learning_rate':[1.0,2.0,3.0],
                'random_state':[0,100,200]

            }
            self.grid=GridSearchCV(AdaBoostClassifier(),param_grid=self.param_grid_dt,cv=5)
            # Fitting the grid to get the best hyperparameters
            self.grid.fit(X_train,y_train)
            # Extracting the best hyperparameters
            self.n_estimators=self.grid.best_params_['n_estimators']
            self.learning_rate=self.grid.best_params_['learning_rate']
            self.random_state=self.grid.best_params_['random_state']
            # Create a model with the best obtained hyperparameters
            self.ada_classifier=AdaBoostClassifier(n_estimators=self.n_estimators,learning_rate=self.learning_rate,random_state=self.random_state)
            # Fit the data
            self.ada_classifier.fit(X_train,y_train)
            self.logger_object.log(self.file_object,'ADA best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_ada_classifier method of the Model_Finder class')
            # Return the model
            return self.ada_classifier

        except Exception as e:
            self.logger_object.log(self.file_object," Exception occured in finding the best hyperparameters for the Ada Boost Classifier with "+str(e))
            self.logger_object.log(self.file_object,"ADA, Model Training failed, exited the AdaBoosts Mehtod")
            raise Exception()

    
    def get_best_params_from_dt(self,X_train,y_train):
        self.logger_object.log(self.file_object,"Entered the method for Finding best hyperparameters from Decision Tree Classifier")
        try:
            """ A General format for GridSearchCV:
                1. Instantiate a GridSearchCV object
                2. Fit the data
                3. The GridSearchCV will check for the permutations and combinations of the hyperparameters
                4. Get the best set of hyperparameters for the list specified
                5. Create a new model with these best hyperparameters obtained
                6. Fit the Data and calculate the AUC ROC Score
            """
            self.param_grid_dt={
                'criterion':['gini','entropy','log_loss'],
                'max_depth':[2,3,4],
                'max_features':['auto','sqrt','log2']
            }
            self.grid=GridSearchCV(DecisionTreeClassifier(),param_grid=self.param_grid_dt,cv=5)
            # Fitting the grid to get the best hyperparameters
            self.grid.fit(X_train,y_train)

            # Extracting the best hyperparameters
            self.criterion=self.grid.best_params_['criterion']
            self.max_depth=self.grid.best_params_['max_depth']
            self.max_features=self.grid.best_params_['max_features']
            # Create a new model with the best obtained hyperparameters
            self.dt_classifier=DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth,max_features=self.max_features)
            # Fit the model
            self.dt_classifier.fit(X_train,y_train)
            self.logger_object.log(self.file_object,'DT best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_decisionTreeClassifier method of the Model_Finder class')
            return self.dt_classifier
            

        except Exception as e:
            self.logger_object.log(self.file_object," Exception occured in finding the best hyperparameters for the Decision Tree Classifier with "+str(e))
            self.logger_object.log(self.file_object,"DT, Model Training failed, exited the Decision tree Mehtod")
            raise Exception()
    
    def get_best_model(self,X_train,y_train,X_test,y_test):
        """
        Returns the model with the best AUC ROC Score out of the following models 
        """
        self.logger_object.log(self.file_object,"Entered the method to find the best Model")
        """
            1. Sequentially Create the best models by calling the methods
            2. Check if the length is 1, return the accuracy score
            3. Else predict and get thee AUC Score
            4. Get list of all the AUC Scores
            5. Compare and return the best model
        """
        # Append the Scores in a dictionary with the model names and scores
        # Then select the model with the highest value
        model_scores=dict()
        try:
            # Create the best model for XGBoost
            
            self.xgboost= self.get_best_params_from_xgb(X_train,y_train)
            # Make Predictions and check for the conditions
            #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
            self.prediction_xgboost=self.xgboost.predict(X_test)
            if len(y_test.unique()) == 1:
                self.xgboost_score=accuracy_score(y_test,self.prediction_xgboost)
                self.logger_object.log(self.file_object,"Accuracy score for XGBoost is"+str(self.xgboost_score))
            else:
                self.xgboost_score=roc_auc_score(y_test,self.prediction_xgboost)
                self.logger_object.log(self.file_object,"AUC ROC Score for XGBoost is"+str(self.xgboost_score))
            
            model_scores['self.xgboost'].append[self.xgboost_score]

            # Create the best model for Random Forest  
            self.rfc=self.get_best_params_from_rf(X_train,y_train)
            # Make prediction and check for label consition and accoridngly give the accuracy /roc auc score
            self.prediction_rfc=self.rfc.predict(X_test)
            if len(y_test.unique())==1:
                self.rfc_score=accuracy_score(y_test,self.prediction_rfc)
                self.logger_object.log(self.file_object,"Accuracy score for Random Forest Classifier is"+str(self.rfc_score))
            
            else:
                self.rfc_score=roc_auc_score(y_test,self.prediction_rfc)
                self.logger_object.log(self.file_object,"AUC ROC Score for Random Forest Classifier is"+str(self.rfc_score))

            model_scores['self.rfc'].append[self.rfc_score]
            
            # Create the best model for SVC
            self.svm=self.get_best_params_for_svm(X_train,y_train)
            self.prediction_svm=self.svm.predict(X_test) # prediction using the SVM Algorithm

            if len(y_test.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.svm_score = accuracy_score(y_test,self.prediction_svm)
                self.logger_object.log(self.file_object, 'Accuracy for SVM:' + str(self.svm_score))
            else:
                self.svm_score = roc_auc_score(y_test, self.prediction_svm) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for SVM:' + str(self.svm_score))

            model_scores['self.svm'].append[self.svm_score]

            # Create the best model for AdaBoost
            self.ada=self.get_best_params_for_ada(X_train,y_train)
            self.prediction_ada=self.ada.predict(X_test) # prediction using the SVM Algorithm

            if len(y_test.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.ada_score = accuracy_score(y_test,self.prediction_ada)
                self.logger_object.log(self.file_object, 'Accuracy for AdaBoost:' + str(self.ada_score))
            else:
                self.ada_score = roc_auc_score(y_test, self.prediction_ada) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for AdaBoost:' + str(self.ada_score))

            model_scores['self.ada'].append[self.ada_score]

            # Create the best mdoel for Decision Tree Classifier
            self.dtc=self.get_best_params_for_dt(X_train,y_train)
            self.prediction_dtc=self.dtc.predict(X_test) # prediction using the SVM Algorithm

            if len(y_test.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.dtc_score = accuracy_score(y_test,self.prediction_dtc)
                self.logger_object.log(self.file_object, 'Accuracy for Decision Tree Classifier:' + str(self.dtc_score))
            else:
                self.dtc_score = roc_auc_score(y_test, self.prediction_dtc) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for Decision Tree Classifier:' + str(self.dtc_score))
            model_scores['self.dtc'].append[self.dtc_score]
        
            Keymax = max(zip(model_scores.values(), model_scores.keys()))[1]
            # Return the model with the highest AUC ROC Score
            return Keymax
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()




