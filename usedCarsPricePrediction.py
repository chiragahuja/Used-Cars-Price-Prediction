import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Utility Functions

def getDataBatch(dataFile):
    """Generator function which generates data(from a csv file) in chunks of 1000"""
    dataset=pd.read_csv(dataFile, encoding='ISO-8859-1',chunksize=1000,iterator=True)
    for dataBatch in dataset:
        yield(dataBatch)
    
    
def getUniqueFeatureValues(dataFile):
    """Iterates over the dataset and returns unique values of each columns in the dataset"""
    uniqueColumnValues = {}
    for dataBatch in getDataBatch(dataFile):
        dataBatch = cleanDataset(dataBatch)
        for col in dataBatch:
            if(col != "price"):
                if col in uniqueColumnValues:
                    uniqueColumnValues[col] = np.append(uniqueColumnValues[col], dataBatch[col].unique())
                    uniqueColumnValues[col] = np.unique(uniqueColumnValues[col])
                else:
                    uniqueColumnValues[col] = dataBatch[col].unique()
    
    return uniqueColumnValues

def getFeatureEncoder(uniqueColumnValues):
    """Creates and returns LabelEncoder(which maps strings to integer) of each column/feature"""
    columnEncoder = {}
    for key in uniqueColumnValues:
        columnEncoder[key] = LabelEncoder().fit(uniqueColumnValues[key])
    
    return columnEncoder

def cleanDataset(dataset):
    """Cleans the dataset. 
    
    Removes unnecessary rows and columns.
    Replaces nan values.
    """
    dataset = dataset.fillna("NAN")
    dataset = dataset.drop(columns=['dateCrawled', 'dateCreated', 'nrOfPictures', 'name', 'lastSeen'])
    dataset = dataset[(dataset.yearOfRegistration <= 2018) 
                      & (dataset.yearOfRegistration >= 1950)  
                      & (dataset.powerPS >0)
                      & (dataset.price > 100)]

    return dataset
        

def encodeDataset(dataset, columnEncoderDict):
    """Uses labelEncoder to convert value of each column in dataset to a categorical value"""
    for key in columnEncoderDict:
        pass
        encodedValue = columnEncoderDict[key].transform(dataset[key])
        dataset.loc[:,key] = encodedValue
    return dataset
            
            
def getFeatureEncoderForData(dataFile):
    """Iterates over the dataset to get unique values of each features 
    and creates a LabelEncoder for each feature using the unique values.
    """
    uniqueFeatureValues = getUniqueFeatureValues(dataFile)
    featureEncoderDict = getFeatureEncoder(uniqueFeatureValues)
    return featureEncoderDict

def preprocessDataset(batchDataset,featureEncoderDict):
    """Cleans data, encodes Data and converts the label 
    which ensures RMSLE loss function is used during training
    """
    batchDataset = cleanDataset(batchDataset) 
    batchDataset = encodeDataset(batchDataset,featureEncoderDict)
    
    Y = np.log1p(batchDataset['price'])  
    batchDataset = batchDataset.drop(columns=['price'])
    return batchDataset,Y
    
    
def getBatchedFeaturesAndLabel(dataFile,featureEncoderDict, forTraining=True,splitRatio=0.2):
    """Generator for returning prerpocessed train/test dataset in chunks of 1000"""
    for batchDataset in getDataBatch(dataFile):
        batchDataset,Y = preprocessDataset(batchDataset,featureEncoderDict)
        if forTraining:
            trainX,_,trainY,_ = train_test_split(batchDataset,Y, test_size=splitRatio, shuffle=False) 
            yield trainX,trainY
        else:
            _,testX,_,testY = train_test_split(batchDataset,Y, test_size=splitRatio, shuffle=False)
            yield testX,testY

def trainModel(dataFile,featureEncoderDict,model):
    """Trains a random forest model in batches and returns the model"""
    print("=> Training Process Started")
    for trainX,trainY in getBatchedFeaturesAndLabel(dataFile,featureEncoderDict):
        model.fit(trainX, trainY)  
        model.n_estimators += 3
    print("=> Training Process Completed")
    return model


def testModelAndSavePredictions(dataFile,resultsFile,featureEncoderDict,model,splitRatio=0.2):
    """Predicts used car price using trained random forest model on test data and stores result 
    in a csv file along with carSoldOnLesserValue column(which signifies whether a car is 
    sold at <90% of its average value)
    """
    
    print("=> Model Testing Started")
    firstTime = True
    carsBelowsAverage = 0
    totalCars = 0
    for batchDataset in getDataBatch(dataFile):
        
        batchDataset = cleanDataset(batchDataset)
        batchDatasetCopy = batchDataset.copy()
        
        batchDataset = encodeDataset(batchDataset,featureEncoderDict)
        Y = np.log1p(batchDataset['price'])  
        batchDataset = batchDataset.drop(columns=['price'])
        
        _,testX,_,testY = train_test_split(batchDataset,Y, test_size=splitRatio,shuffle=False)
        _,batchDatasetCopy,_,_ = train_test_split(batchDatasetCopy,Y, test_size=splitRatio,shuffle=False)
        
        predY = model.predict(testX)
        predY = np.expm1(predY)
        
        batchDatasetCopy["predictedPrice"] = predY
        batchDatasetCopy["carSoldOnLesserValue"] = (batchDatasetCopy["price"]/batchDatasetCopy["predictedPrice"])<.9
        
        carsBelowsAverage = carsBelowsAverage + sum(batchDatasetCopy["carSoldOnLesserValue"])
        totalCars = totalCars + batchDatasetCopy.shape[0]
        if firstTime:
            batchDatasetCopy.to_csv(resultsFile, index=False)
            firstTime = False
        else:
            batchDatasetCopy.to_csv(resultsFile, header=False ,index=False, mode="a")
    
    print("=> Model Testing Completed")
    
    print(f"\n% of Cars in Testset sold below 90% of Average Value: {carsBelowsAverage*100/totalCars}")
    print(f"=> Results for test set are stores in \"{resultsFile}\"")


def getTestRSquaredValue(dataFile,featureEncoderDict,model):
    "Returns average r squared values of test data"
    totalSamples = 0
    r2Score=0  
    for testX,testY in getBatchedFeaturesAndLabel(dataFile,featureEncoderDict,False):
        r2Score = r2Score + testX.shape[0]*model.score(testX,testY)
        totalSamples = totalSamples + testX.shape[0]
    return r2Score/totalSamples



if __name__ == "__main__":

    dataFile = "data/autos.csv"

    featureEncoderDict= getFeatureEncoderForData(dataFile)

    rForest = RandomForestRegressor(n_estimators=10,warm_start=True)

    trainModel(dataFile,featureEncoderDict,rForest)


    testRSquaredValue = getTestRSquaredValue(dataFile,featureEncoderDict,rForest)
    print(f"\nrSquared Value for Test Data : {testRSquaredValue}")

    resultsFile = "data/results.csv"
    testModelAndSavePredictions(dataFile,resultsFile,featureEncoderDict,rForest)