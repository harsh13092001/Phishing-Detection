"""
Cluster the data into some clusters so that each cluster can have a best model according to it's datapoints

"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from file_operations import  file_methods

class KMeansClustering:
    """"
    Class Doc String - Use the class to divide the data into clusters
    """
    # Constructor
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object
    
    """
    This method is used to get an elbow plot which shows what are the optimum number of clusters that 
    need to be formed, whenever it plateaus it means that optimum number of clusters are reached
    """
    def elbow_plot(self,data):
        """
        WCSS is the sum of squared distance between each point and the centroid in a cluster. When we plot the WCSS with the K value, the plot looks like an Elbow. As the number of clusters increases, the WCSS value will start to decrease. WCSS value is largest when K = 1.
        """
        self.logger_object.log(self.file_object,"Entry into the Elbow Plot method")
        # Initialize an empty list for wcss
        wcss=[]
        try:
            for i in range(1,11):
                # Initializing an KMeans Object with iterator number of clusters
                kMeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
                # Fit the object to the data
                kMeans.fit(data)
                # Inertia measures how well a dataset was clustered by K-Means. It is calculated by measuring the distance between each data point and its centroid, squaring this distance, and summing these squares across one cluster. A good model is one with low inertia AND a low number of clusters ( K ).
                # Get the Inertia for the current fit and append it to the list
                wcss.appeand(kMeans.inertia_)
            # Plot the curve and Locate the Knee using the kneed KneeLocator and Data Generator
            # Creating a plot Object
            plt.plot(range(1,11),wcss)  
            plt.title("Elbow Plot")
            plt.xlabel("The number of clusters")
            plt.ylabel("The WCSS Distance")
            # Saving the plot
            plt.savefig('Clustering_Plot/K-Means_Elbow.PNG')
            # Finding the optimum number of clusters
            self.kn=KneeLocator(range(1,11),wcss,curve='convex',direction='decreasing')
            self.logger_object.log(self.file_object,'The optimum number of clusters is: '+str(self.kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
            return self.kn.knee
        
        except Exception as e:
            self.logger_object.log(self.file_object,"Finding the elbow plot method exited with Execption" + str(e))
            self.logger_object.log(self.file_object,"Finding the right number of clusters exited with Exception")
            raise Exception()
    
    def create_clusters(self,data,num_of_clusters):
        """
        Method: create clusters 
        Create a new dataframe which consists of the cluster infnormation for datapoints
        """
        self.logger_object.log(self.file_object,"Entry into the clustering method")
        self.data=data
        try:
            # A KMeans for the self Object
            self.kmeans=KMeans(n_clusters=num_of_clusters,init="kmeans++",random_state=42)
            # Dividing the data into clusters
            self.y_kmeans=self.kmeans.fit_predict(data)
            # Keep the naming consistent in the File Operations method
            self.file_op = file_methods.File_Operation(self.file_object,self.logger_object)
            # Saving the KMeans model to the directory
            self.save_model=self.file_op.save_model(self.kmeans,'KMeans')
            # Now for each data point to know which cluster it belongs to need to create a new column
            self.data['cluster']=self.y_kmeans
            self.logger_object.log(self.file_object, 'succesfully created '+str(self.kn.knee)+ 'clusters and appeneded to the DataFrame.')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()
