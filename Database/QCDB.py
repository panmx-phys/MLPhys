from http.client import NON_AUTHORITATIVE_INFORMATION
import json
from datetime import datetime
from pickle import NONE
from tkinter.messagebox import NO

from sympy import re

descriptions = [
        "Full unitary using Jax",
        "Rotation based quantum computing, without noise",
        "Rotation based quantum computing"
]

jsonFile = "~/data/QCDB.json"

# defaults
_H = 0.5
_J = 0.5
_N = 6

def saveToDB(description:str, beta,res, h=_H, J=_J,N = _N, additional = dict()):
    import pymongo
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mlDB = myclient["mlDB"]
    QCcollection = mlDB["QCDB"]
    newDict = {"description":description, "labels" :{"beta" : beta,"h" :h,"J" : J, "N" : N},"result" : res, "expTime": datetime.now(), "valid": True}
    newDict = {**newDict,**additional}
    QCcollection.insert_one(newDict)


def saveTojson(description:str, beta,res, h=_H, J=_J,N = _N, additional = dict()):
    newDict = {"description":description, "labels" :{"beta" : beta,"h" :h,"J" : J, "N" : N},"result" : res, "expTime": datetime.now(), "valid": True}
    newDict = {**newDict,**additional}
    newJson = json.dumps(newDict)
    with open(jsonFile) as data_file:
        old_data = json.load(data_file)
    old_data += newJson
    with open(jsonFile, 'w') as outfile:
        json.dump(old_data, outfile)
    return
    



    
