"""
This file has a class for Logging the logs 
Logs are being maintained for majority of the processes
"""
from datetime import datetime
class App_Logger:
    # Constructor
    def __init__(self):
        pass
    def log(self,file_object,log_message):
        self.now=datetime.now()
        self.date=datetime.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")

