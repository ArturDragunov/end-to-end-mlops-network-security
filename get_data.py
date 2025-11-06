import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")

# To define security protocol when we connect to cloud from local especially for MongoDB
import certifi
# Setting up a trusted SSL certificate bundle so your HTTPS requests are safe and verified.
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging


class NetworkDataExtract():
    # you don't need to define __init__ constructor if you assume it empty
    # it will be added automatically in the background
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def pushing_data_to_mongodb(self,records, database,collection):
        try:
            self.database=database # we save parameter to object variable, so that later we could use it
            self.collection=collection
            self.records=records
            
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
                
            self.collection.insert_many(self.records)
                
            return len(self.records)
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
if __name__ == '__main__':
    FILE_PATH="./Network_Data/NetworkData.csv"
    DATABASE="ArturPortfolio" # can be any name
    COLLECTION="NetworkData"
    networobj = NetworkDataExtract()
    records = networobj.csv_to_json_convertor(FILE_PATH)
    noofrecords=networobj.pushing_data_to_mongodb(records,DATABASE,COLLECTION)
    print(f'{noofrecords=}')