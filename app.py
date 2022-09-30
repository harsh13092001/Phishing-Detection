# The wsgiref server is much better than the Flask server
#------importing all libraries
from shutil import ExecError
from wsgiref import simple_server
from flask import Flask
from flask import  request
from flask import Response
# Cross-origin resource sharing is a mechanism that allows restricted resources on a web page to be requested from another domain outside the domain from which the first resource was served
# These 4 files are kept 
from flask_cors import CORS,cross_origin
from data_training_validation_insertion import train_validation
from data_training_model import train_model
from data_prediction_validation_insertion import predict_validation
from data_prediction_from_model import Prediction
import flask_monitoringdashboard as dashboard
import os
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
# Initialize an app
# Bind the app to the dashboard and initialize CORS

app=Flask(__name__)
dashboard.bind(app)
CORS(app)
# Managing the routes

@app.route("/predict",methods=['POST'])
@cross_origin()
#------Creating  function for Route client
def predictRouteClient():
    try:
        # If the requested folder path is not None
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            # #object initialization
            pred_val = predict_validation(path) 
            #calling the prediction_validation function
            pred_val.predictionValidation() 
            #object initialization
            pred = Prediction(path) 
            # predicting for dataset present in database
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % path)
    # Handling all the exceptions
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/train",methods=['POST'])
@cross_origin
def trainRouteClient():
    try:
        if request.json['folderPath'] is not None:
            # Data in JSON Format
            path=request.json['folderPath']
            # Create a Validation Object
            train_obj=train_validation(path)
            # Calling the train validation function for this Object
            train_obj.trainValidation()
            # Initialize an Object for Model Training from the class
            train_model_obj=train_model()
            # Calling the function to train the Model
            train_model_obj.trainingModel()
    except ValueError:
        return Response("Error Occured %s" %ValueError)
    except KeyError:
        return Response("Error Occured %s" %KeyError)
    except Exception as e:
        return Response("Error Occurreded! %s" % e)

# Initializing a server
port = int(os.getenv('PORT'))
if __name__=="__main__":
    host="127.0.0.1"
    # Make a server with thee wsgrief httpd method
    httpd=simple_server.make_server(host,port,app)
    httpd.serve_forever()
