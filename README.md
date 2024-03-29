# Problem Statement
Build a Machine Learning model that can predict whether a car is sold below 90% of its average value.

# Files/Folders
* **data**: Contains the dataset to be used for training and testing.
* **DataAnalysis_UsedCarsDataset.ipynb**: Python notebook in which data analysis is done.
* **usedCarsPricePrediction.py**: Defines the predictive model.
* **Dockerfile**: Used for building and executing the docker image.
* **requirements.txt**: Defines the packages to be installed.

# Dataset
* Kaggle Dataset : [Used Cars Prediction](https://www.kaggle.com/orgesleka/used-cars-database)

# Execution Steps
* docker image build -t usedcarsdocker .
* docker run -v \<fullPathToDataset\>:/data/ usedcarsdocker

# Result
* Of every 1000 data samples selected from dataset, last 200 data samples act as test data.
* A CSV file "results.csv"(which is stored alongside dataset file) is generated which contains entire test data set along with predicted price for each
  data sample (using trained random forest model) and "carSoldOnLesserValue" column (which signifies
  whether a car is sold at <90% of its average value).
