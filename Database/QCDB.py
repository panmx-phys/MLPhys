import pymongo
from datetime import datetime
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mlDB = myclient["mlDB"]
QCcollection = mlDB["QCDB"]

descriptions = [
        "Full unitary using Jax",
        "Rotation based quantum computing, without noise",
        "Rotation based quantum computing"
]

# defaults
_H = 0.5
_J = 0.5
_N = 6

def saveToDB(description:str, beta,res, h=_H, J=_J,N = _N, additional = dict()):
    newDict = {"description":description, "labels" :{"beta" : beta,"h" :h,"J" : J, "N" : N},"result" : res, "expTime": datetime.now(), "valid": True}
    newDict = {**newDict,**additional}
    QCcollection.insert_one(newDict)

