import pymongo
import json
from datetime import datetime

descriptions = [
        "Full unitary using Jax",
        "Rotation based quantum computing, without noise",
        "Rotation based quantum computing"
]

jsonFile = "/Users/qwe/data/QCDB.json"

# defaults
_H = 0.5
_J = 0.5
_N = 6

def saveToDB(description:str, beta,res, h=_H, J=_J,N = _N, additional = dict()):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mlDB = myclient["mlDB"]
    QCcollection = mlDB["QCDB"]
    newDict = {"description":description, "labels" :{"beta" : beta,"h" :h,"J" : J, "N" : N},"result" : res, "expTime": datetime.now(), "valid": True}
    newDict = {**newDict,**additional}
    QCcollection.insert_one(newDict)


def saveTojson(description:str, beta,res, h=_H, J=_J,N = _N, additional = dict()):
    t = str(datetime.now())
    newDict = {"description":description, "labels" :{"beta" : beta,"h" :h,"J" : J, "N" : N},"result" : res, "expTime": t, "valid": True}
    newDict = {**newDict,**additional}
    newJson = json.dumps(newDict)
    with open(jsonFile) as data_file:
        old_data = json.loads(data_file)
    old_data += newJson
    with open(jsonFile, 'w') as outfile:
        json.dumps(old_data, outfile)
    return
    
