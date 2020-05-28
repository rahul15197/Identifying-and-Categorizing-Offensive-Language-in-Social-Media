# completed
import warnings
import pickle as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

olid_data = open("training.tsv", encoding='utf-8')
taska_test = open("taska_test.tsv", encoding='utf-8')
taska_test_labels = open("labels_a.csv", encoding='utf-8')

training_dataframe = pd.read_table(olid_data, sep="\t")
test_dataframe = pd.read_table(taska_test, sep="\t")
test_labels_dataframe = pd.read_csv(taska_test_labels)

training_tweet_list = training_dataframe['tweet'].tolist()
test_tweet_list = test_dataframe['tweet'].tolist()

training_task_a_labels_list = training_dataframe['subtask_a'].tolist()
test_task_a_labels_list = test_labels_dataframe['subtask_a'].tolist()

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(training_tweet_list)
test_vectors = vectorizer.transform(test_tweet_list)

# svm
model_svm = svm.SVC(kernel='linear')
model_svm.fit(train_vectors, training_task_a_labels_list)
prediction_svm = model_svm.predict(test_vectors)

print("Classification report for SVM")
print(classification_report(test_task_a_labels_list, prediction_svm))
accuracy_svm = round(accuracy_score(test_task_a_labels_list, prediction_svm) * 100, 2)
print("Accuracy (SVM) = " + str(accuracy_svm) + " %")
print("\nConfusion Matrix (SVM)")
cf_matrix_svm = confusion_matrix(test_task_a_labels_list, prediction_svm)
print(cf_matrix_svm)

# decision tree
model_dt = DecisionTreeClassifier(max_depth=6)
model_dt.fit(train_vectors, training_task_a_labels_list)
prediction_dt = model_dt.predict(test_vectors)

print("\nClassification report for Decision Tree")
print(classification_report(test_task_a_labels_list, prediction_dt))
accuracy_dt = round(accuracy_score(test_task_a_labels_list, prediction_dt) * 100, 2)
print("Accuracy (DT) = " + str(accuracy_dt) + " %")
print("\nConfusion Matrix (DT)")
cf_matrix_dt = confusion_matrix(test_task_a_labels_list, prediction_dt)
print(cf_matrix_dt)

# knn
model_knn = KNeighborsClassifier(n_neighbors=10)
model_knn.fit(train_vectors, training_task_a_labels_list)
prediction_knn = model_knn.predict(test_vectors)

print("\nClassification report for K Nearest Neighbour")
print(classification_report(test_task_a_labels_list, prediction_knn))
accuracy_knn = round(accuracy_score(test_task_a_labels_list, prediction_knn) * 100, 2)
print("Accuracy (KNN) = " + str(accuracy_knn) + " %")
print("\nConfusion Matrix (KNN)")
cf_matrix_knn = confusion_matrix(test_task_a_labels_list, prediction_knn)
print(cf_matrix_knn)

# logistic regression
model_lr = LogisticRegression()
model_lr.fit(train_vectors, training_task_a_labels_list)
prediction_lr = model_lr.predict(test_vectors)

print("\nClassification report for Logistic Regression")
print(classification_report(test_task_a_labels_list, prediction_lr))
accuracy_lr = round(accuracy_score(test_task_a_labels_list, prediction_lr) * 100, 2)
print("Accuracy (LR)= " + str(accuracy_lr) + " %")
print("\nConfusion Matrix (LR)")
cf_matrix_lr = confusion_matrix(test_task_a_labels_list, prediction_lr)
print(cf_matrix_lr)

labels = ['NOT', 'OFF']

# graph plot svm
fig = plt.figure(1)
ax = fig.add_subplot()
cax = ax.matshow(cf_matrix_svm)
plt.title('Confusion matrix of SubTask A (SVM)')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')

# graph plot dt
fig = plt.figure(2)
ax = fig.add_subplot()
cax = ax.matshow(cf_matrix_dt)
plt.title('Confusion matrix of SubTask A (DT)')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')

# graph plot knn
fig = plt.figure(3)
ax = fig.add_subplot()
cax = ax.matshow(cf_matrix_knn)
plt.title('Confusion matrix of SubTask A (KNN)')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')

# graph plot lr
fig = plt.figure(4)
ax = fig.add_subplot()
cax = ax.matshow(cf_matrix_lr)
plt.title('Confusion matrix of SubTask A (LR)')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')

classifier_list = ['SVM', 'DT', 'KNN', 'LR']
accuracy_list = [accuracy_svm, accuracy_dt, accuracy_knn, accuracy_lr]

fig = plt.figure(5)
ax = fig.add_subplot()
plt.title("Accuracy Comparison for SubTask A")
plt.ylabel('Classifier')
plt.xlabel('Accuracy (%)')
plt.barh(classifier_list, accuracy_list)
for i, j in enumerate(accuracy_list):
    ax.text(j + 1, i + .10, str(j), color='blue')
plt.show()

file = open('classifier_dump_a', 'wb')
file1 = open('accuracy_list_dump_a', 'wb')
pl.dump(classifier_list, file)
pl.dump(accuracy_list, file1)

