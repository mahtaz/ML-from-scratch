def euclidean_distance(x1, x2):
    return (np.sum((x1 - x2) ** 2)) ** 0.5


def train_test_split(dataset):
    l = int(0.8 * len(dataset))
    training_data = dataset[:l]
    testing_data = dataset[l:]
    return training_data, testing_data

def accuracy(actaull_y,predicted_y):
    m=0
    for i in range(len(actaull_y)):
        if actaull_y[i] == predicted_y[i]:
                m+=1
    accuracy = m / len(predicted_y)
    return accuracy


def read_data(file_name):   #addressing by giving the file name
    df=pd.read_csv(file_name)
    xs=df[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
    ys=df["target"]
    x =xs.values #data_frame to np.array
    y=ys.values
    return x,y


def shuffle(a, b):
    s = np.random.permutation(len(a))
    return a[s], b[s]
def k_fold_validation(x,y,k): #x an y must be shuffled

    l=int(len(X)/5)
    #print(len(X))
    m,n=0,0
    for i in range(1,6):
        if i==5:
            break
        x1=np.split(X, [i*l, (i+1)*l])#spliting array into 3 parts the middle part is 1/5 of data and beside it are therest
        y1=np.split(y, [i*l, (i+1)*l])
        training_x,testing_x=np.concatenate((x1[0], x1[2])),x1[1]
        training_y,testing_y=np.concatenate((y1[0], y1[2])),y1[1]
       # print(len(y1[1]))
        model=KNN(k=k)
        model.fit(training_x,training_y)
        predictions = model.predict(testing_x)
        predictionss = model.predict(training_x)
        m+=accuracy(testing_y, predictions)
        n+=accuracy(training_y, predictionss)
        #print(accuracy(testing_y, predictions),accuracy(training_y, predictionss))
    training_x,testing_x=[X[:4*l], X[4*l+3:]]#+3 is for the split size to be 60 like othe four test_splits
    training_y,testing_y=[y[:4*l], y[4*l+3:]]
    #print(len(testing_y))
    predictions = model.predict(testing_x)
    predictionss = model.predict(training_x)
    m+=accuracy(testing_y, predictions)
    n+=accuracy(training_y, predictionss)
   # print("mean test accuracy is:",m/5,"and mean training accuracy is:",n/5)
    return m/5 #mean of the 5 time spliting of data


def confusion_matrix(y_test, y_pred):
    TN, TP, FN, FP = 0, 0, 0, 0
    for i in range(len(y_test)):
        if y_test[i] == 0 and clf.predict(X_test)[i] == 0:
            TN += 1
        elif y_test[i] == 1 and clf.predict(X_test)[i] == 1:
            TP += 1
        elif y_test[i] == 1 and clf.predict(X_test)[i] == 0:
            FN += 1
        else:
            FP += 1

    return TN, TP, FN, FP

def classification_report(y_test, y_pred):#accuracy, precision, recall, specificity, f1Score
    TN,TP,FN,FP =confusion_matrix(y_test, y_pred)
    totall=TN+TP+FN+FP
    specificity=TN/(TN+FP)#True Negative)/(True Negative + False Positive
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    accuracy=(TP+TN)/totall
    f1score=(2*precision*recall)/(precision+recall)
    return accuracy, precision, recall, specificity, f1score

def entropy(labels):
    probs = np.bincount(labels) / len(labels)
   # print(probs)
    c=0
    for prob in probs:
        if prob>0:
            c+=-1*(prob * np.log2(prob))
    return c
