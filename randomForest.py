from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
import csv

# Number of attributes
attNums = 15

# List save all attribute names
attList = ['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE',
           'FATIGUE' ,'ALLERGY' ,'WHEEZING','ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH',
           'SWALLOWING DIFFICULTY','CHEST PAIN','LUNG_CANCER']

        

if __name__ == '__main__':
    # X is data, y is result of predictions
    X, y = [[] for _ in range(2)]

    # open file data
    with open('db_lung_cancer.csv', 'r') as f:
        title = f.readline() # title 
        line = f.readline()
        
        while line: # Loop over lines in database
            arr = line.split(',') # Handle , and store attributes of 1 instance in a list
            arr[0] = 0 if arr[0] == 'M' else 1 # Convert non-numeric attribute to boolean
            X.append([int(x) for x in arr[:-1]]) # Add all attributes except the last one to X
            y.append(0 if arr[-1] == 'NO\n' else 1) # Add the last attribute to Y
            
            line = f.readline() # Move to next line

    # Split by percent and store data in lists
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=25)
    rf.fit(X_train, y_train) # Train the Random Forest model



    # Make predictions and evaluate
    y_pred = rf.predict(X_test) # Predicts the class labels for the test dataset X_test.
    y_prob = rf.predict_proba(X_test)[:, 1]  # Predicts the probability of the positive class (class 1) for each sample in X_test

    acc = accuracy_score(y_test, y_pred)  # Proportion of correct predictions out of total predictions.
    prec = precision_score(y_test, y_pred) # Proportion of correctly predicted positive samples out of all predicted positives.
    rec = recall_score(y_test, y_pred) # Proportion of actual positive samples correctly identified as positive.
    f1 = f1_score(y_test, y_pred) # Harmonic mean of precision and recall.
    roc_auc = roc_auc_score(y_test, y_prob) # Measures the model's ability to distinguish between classes
    cm = confusion_matrix(y_test, y_pred) # Provides a matrix summarizing the prediction results:
    report = classification_report(y_test, y_pred) # Generates a detailed text summary of precision, recall, F1-score, and support for each class.

    # Print out metrics
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
