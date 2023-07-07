import comet_ml
from comet_ml import Experiment, Artifact
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,classification_report,precision_score,recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle
import json

# experiment = comet_ml.Experiment(
#     api_key="jNBCAtiZnHlP2dsxXyGOTiJc2",
#     project_name="VS MLOps Water Quality Prediction PoC"
# )

# Tracking dataset CometML
# artifact = Artifact(name="VS MLOps Water Quality Prediction Dataset PoC", artifact_type="dataset")
# artifact.add("./data/water_preprocessed.csv")
# experiment.log_artifact(artifact)


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

# Experiment Tracking
# def log_classification_report(y_true, y_pred):
#     report = classification_report(y_true, y_pred, output_dict=True)
#     for key, value in report.items():
#         if key == "accuracy":
#             experiment.log_metric(key, value)
#         else:
#             experiment.log_metrics(value, prefix=f"{key}")


# with experiment.train():
#     log_classification_report(train_y, clf_knn.predict(train_x))

# with experiment.test():
#     log_classification_report(test_y, clf_knn.predict(test_x))


# Export model pickle
with open("./app/modelknn.pkl", "wb") as f:
    pickle.dump(clf_knn,f)


# experiment.log_model("VS MLOps Water Quality Prediction Model PoC", "./app/modelknn.pkl")
