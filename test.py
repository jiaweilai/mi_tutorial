import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from sklearn import linear_model
from matplotlib import style

data = pd.read_csv("./student/student-mat.csv",sep=";")

print(data.head())
data = data[["G1","G2","G3","studytime","failures","absences"]]
print(data.head())

predict = "G3"

X = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

'''best = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)

    acc = linear.score(x_test,y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear, f)

print(best)'''
pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

'''
p = 'absences'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()'''

style.use("ggplot")
pyplot.scatter(y_test,predictions)
pyplot.xlabel("Exp")
pyplot.ylabel("Prediction")
pyplot.xlim([0,20])
pyplot.ylim([0,20])
pyplot.show()