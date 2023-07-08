import comet_ml
#from comet_ml import Experiment, Artifact
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,classification_report,precision_score,recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle
import json

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ['MLFLOW_TRACKING_USERNAME'] =  os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")


os.environ['MLFLOW_TRACKING_URI'] = f'https://dagshub.com/Bwan109/mlops-poc.mlflow'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])


#mlflow.set_tracking_uri("https://dagshub.com/Bwan109/mlops-poc.mlflow")
mlflow.sklearn.autolog()

data = pd.read_csv('water_preprocessed.csv',index_col=0)

# # Splitting X and Y
X = data.drop(columns=['is_safe'])
y = data['is_safe']

# #Train_Test Split
train_x,test_x,train_y,test_y=train_test_split(X,y,random_state=42,test_size=0.20)
clf_knn = KNeighborsClassifier() 

clf_knn.fit(train_x, train_y) # fit the model
y_pred = clf_knn.predict(test_x) # then predict on the test set
accuracy= accuracy_score(test_y, y_pred) # this gives us how often the algorithm predicted correctly
clf_report= classification_report(test_y, y_pred, output_dict=True) # with the report, we have a bigger picture, with precision and recall for each class

precision =  clf_report['weighted avg']['precision'] 
recall = clf_report['weighted avg']['recall']    
f1_score = clf_report['weighted avg']['f1-score']

# Now print to file metrics.json
with open("metrics.json", 'w') as outfile:
    json.dump({ "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}, outfile)



# Export model pickle
with open("./app/modelknn.pkl", "wb") as f:
    pickle.dump(clf_knn,f)


