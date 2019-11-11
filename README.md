# Problem Statement
Build a Machine Learning model that can predict whether a car is sold below 90% of its average value.

# Dataset
* Kaggle Dataset : [Used Cars Prediction](https://www.kaggle.com/orgesleka/used-cars-database)

# Execution Steps
* docker image build -t usedcarsdocker .
* docker run -v \<fullPathToDataset\>:/data/ usedcarsdocker
