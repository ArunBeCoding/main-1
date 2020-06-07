import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn import svm
import nltk
import string
import sklearn.metrics as metrics
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#############
# Functions #
#############

# Function to append two dicts


def joinDicts(dict1, dict2):
    for value in dict2:
        dict1[value] = dict1[value] + dict2[value]

    return dict1


def dateAdd(datetochange, daystoadd):
    date_object = datetime.strptime(datetochange, "%Y-%m-%d")
    toReturn = date_object + timedelta(days=daystoadd)
    return toReturn.strftime("%Y-%m-%d")


def dateMinus(datetochange, daystoadd):
    date_object = datetime.strptime(datetochange, "%Y-%m-%d")
    toReturn = date_object - timedelta(days=daystoadd)
    return toReturn.strftime("%Y-%m-%d")


# Read CSV
df = pd.read_csv(r'finalfinaldata.csv')
frame = pd.DataFrame(df, columns=['Header'])
# header_1 = frame.iat[1, 0]
# print(header_1)

# Create dict to store all words in it
dict = {}

# Create array to store all tokenized sentences
headerArr = []

# Tokenize each header and add words to dict
for i in range(1596):
    sentence = frame.iat[i, 0]
    tokens = word_tokenize(sentence)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    headerArr.append(words)
    for word in words:
        if word in dict:
            dict[word] = dict[word] + 1
        else:
            dict[word] = 1

# Remove all words that has a occurrence of 1
arrayToRem = []

for entry in dict:
    if dict[entry] == 1:
        arrayToRem.append(entry)

for rem in arrayToRem:
    dict.pop(rem)

# print(len(dict))
# print(dict)

# Create a zero value copy of dict
dictZero = dict.copy()
for entry2 in dictZero:
    dictZero[entry2] = 0

# Create a dict of dates corresponding to dict of words used
finalDict = {}
dates = pd.DataFrame(df, columns=['Date'])
for j in range(1596):
    date = dates.iat[j, 0]
    # Create a new dictionary for this header
    header = headerArr[j]
    headerDict = dictZero.copy()
    for word in header:
        if word in dictZero:
            headerDict[word] = headerDict[word] + 1

    # Add created header dict to final dict
    if date in finalDict:
        finalDict[date] = joinDicts(finalDict[date], headerDict)
        print("")
    else:
        finalDict[date] = headerDict

# Create pandas dataframe
arrToConv = []
for val in finalDict:
    arrToConv.append(finalDict[val])

df = pd.DataFrame(arrToConv)

# Create date dataframe to merge
arrToMerge = []
for y in finalDict:
    arrToMerge.append(y)

df2 = pd.DataFrame(arrToMerge, columns=['Date'])
result = pd.concat([df2, df], axis=1)
# print(result)

# Create dataframe from stock prices
df3 = pd.read_csv(r'AAPL.csv')
frame2 = pd.DataFrame(df3)

# Function to check whether date exists in NLP dataframe
listToConv = finalDict.keys()
modList = []
for key in listToConv:
    newOne = dateAdd(key, 1)
    modList.append(newOne)


def isExists(dat):
    if dat in modList:
        return True


# Cleanup AAPL dataframe by deleting all dates that are not in NLP dataframe
arrToDel = []
for v in range(773):
    if not (isExists(frame2.loc[v, 'Date'])):
        arrToDel.append(v)
        # print(v)

modFrame2 = frame2.drop(arrToDel)

# Cleanup result to delete dates that are not in AAPL
dateArr = modFrame2['Date'].tolist()
olderDateArr = []
for key in modFrame2['Date']:
    newOne = dateMinus(key, 1)
    olderDateArr.append(newOne)


# Function to check whether date exists in stock Dates
def isExists2(dat):
    if dat in olderDateArr:
        return True


arrToDel2 = []
for g in range(686):
    if not (isExists2(result.loc[g, 'Date'])):
        arrToDel2.append(g)

finalResult = result.drop(arrToDel2)
# print(result)
# print(finalResult)
modFrame2.drop(['Date'], axis=1, inplace=True)
# print(modFrame2)

# change all index names to merge later - finalResult
finalResultDict = finalResult.to_dict('index')
newFinalDict = {}
count1 = 0
for dic in finalResultDict:
    newFinalDict[count1] = finalResultDict[dic]
    count1 = count1 + 1

arrToConv2 = []
for val in newFinalDict:
    arrToConv2.append(newFinalDict[val])

finalResult = pd.DataFrame(arrToConv2)
# print(finalResult)

# change all index names to merge later - modFrame2
modFrame2Dict = modFrame2.to_dict('index')
# print(modFrame2Dict)
newModDict = {}
count2 = 0
for dic2 in modFrame2Dict:
    newModDict[count2] = modFrame2Dict[dic2]
    count2 = count2 + 1

# print(newModDict)
arrToConv3 = []
for val in newModDict:
    arrToConv3.append(newModDict[val])

modFrame2 = pd.DataFrame(arrToConv3)
# print(modFrame2)

withAAPL = pd.concat([finalResult, modFrame2], axis=1)
# print(withAAPL)

# Calculate whether stock increased or decreased

sign = []
for index, i in withAAPL.iterrows():
    if i['Close'] >= i['Open']:
        sign.append(1)
    elif i['Close'] < i['Open']:
        sign.append(-1)

withAAPL['Score'] = sign

# to get array of just the numbers
dupFrame = withAAPL.copy()
# dupFrame.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Score'], axis=1, inplace=True)
# dupFrame.drop(['Date', 'Open', 'Close', 'Adj Close', 'Score'], axis=1, inplace=True)
dupFrame.drop(['Date', 'Score'], axis=1, inplace=True)
# dupFrame.drop(['Date'], axis=1, inplace=True)
dupArr = dupFrame.to_numpy()
# print(dupArr)

# End Result: dupArr is arr of just values, and sign is array of Scores (X & Y arrays respectively)
# withAAPL.to_csv('out_mark2.csv')
print("number of param: " + str(dupFrame.columns))

############################
### K-NEAREST NEIGHBOURS ###
############################

X = dupArr
Y = sign
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
# x_train2, x_test2, y_train2, y_test2 = train_test_split(X, Y, train_size=200)

KNNresult = []
# n_neigh = 10
# for i in range(1, 30):
#     model = KNeighborsClassifier(n_neighbors=i)
#     model.fit(x_train, y_train)
#     predicted = model.predict(x_test)
#     accuracy = metrics.accuracy_score(y_test, predicted)
#     KNNresult.append(accuracy)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)
knn_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, knn_pred)
KNNresult.append(accuracy)

auc = metrics.roc_auc_score(y_test, knn_pred)
print("Accuracy of KNN: " + str(KNNresult))
print('AUC of KNN: %.3f' % auc)


#####################
### RANDOM FOREST ###
#####################

RFresult = []
# for j in range(1, 11):
#     clf = RandomForestClassifier(n_estimators=j, max_depth=500)
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     accuracy2 = metrics.accuracy_score(y_test, y_pred)
#     RFresult.append(accuracy2)

clf = RandomForestClassifier(n_estimators=6, max_depth=500)
clf.fit(x_train, y_train)
rf_pred = clf.predict(x_test)
accuracy2 = metrics.accuracy_score(y_test, rf_pred)
RFresult.append(accuracy2)

auc2 = metrics.roc_auc_score(y_test, rf_pred)
print("Accuracy of RF: " + str(RFresult))
print('AUC of RF: %.3f' % auc2)


# for g in range(len(RFresult)):
#     print(str(g) + ": " + str(RFresult[g]))


###########
### SVM ###
###########

clf = svm.SVC(kernel='poly')
clf.fit(x_train, y_train)
svm_pred = clf.predict(x_test)
accuracy3 = metrics.accuracy_score(y_test, svm_pred)
auc3 = metrics.roc_auc_score(y_test, svm_pred)
print("Accuracy of SVM: " + str(accuracy3))
print('AUC of SVM: %.3f' % auc3)


###########################
### Logistic Regression ###
###########################

logisticRegr = LogisticRegression(penalty='l2', solver='sag')
logisticRegr.fit(x_train, y_train)
lg_pred = logisticRegr.predict(x_test)
# accuracy4 = logisticRegr.score(x_test, y_test)
accuracy4 = metrics.accuracy_score(y_test, lg_pred)
auc4 = metrics.roc_auc_score(y_test, lg_pred)
print("Accuracy of LogReg: " + str(accuracy4))
print('AUC of LogReg: %.3f' % auc4)



################
### Adaboost ###
################

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model = abc.fit(x_train, y_train)
ab_pred = model.predict(x_test)
accuracy5 = metrics.accuracy_score(y_test, ab_pred)
auc5 = metrics.roc_auc_score(y_test, ab_pred)
print("Accuracy of Adaboost: " + str(accuracy5))
print('AUC of Adaboost: %.3f' % auc5)



################
### Ensemble ###
################

ens_pred = []
for i in range(len(knn_pred)):
    # val = knn_pred[i] + rf_pred[i] + ab_pred[i] + lg_pred[i] + svm_pred[i]
    val = knn_pred[i] + rf_pred[i] + ab_pred[i]
    if val >= 0:
        ens_pred.append(1)
    else:
        ens_pred.append(-1)

accuracy6 = metrics.accuracy_score(y_test, ens_pred)
auc6 = metrics.roc_auc_score(y_test, ens_pred)
print("Accuracy of Ensemble: " + str(accuracy6))
print('AUC of Ensemble: %.3f' % auc6)




########################
### Confusion Matrix ###
########################

# # cm = metrics.confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(9, 9))
# sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# all_sample_title = 'Accuracy Score: {0}'.format(accuracy4)
# plt.title(all_sample_title, size=15)
# # print(cm)
