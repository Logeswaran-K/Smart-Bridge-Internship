# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:26:56 2021

@author: Logu
"""
from ibm_watson_machine_learning import APIClient
wml_credentials = { "url": "https://us-south.ml.cloud.ibm.com",
                  "apikey": "JHO6rBiA8a_BOg-j0meLSIoRdy7nnI62ooaoqxGaPfWo"
                  }
client = APIClient(wml_credentials)
print(client)

def guide_from_space_name(client,space_name):
    space = client.spaces.get_details()
    return(next(item for item in space['resources']if item['entity']["name"]== space_name)['metadata']['id'])

space_uid = guide_from_space_name(client,'CNN_Breast_Cancer_Prediction')
print("Space UID = " + space_uid)


client.set.default_space(space_uid)

client.repository.download('d5cf8374-00ff-4545-b8d6-a1571230cb80','my_model.tar.gz')

