import seaborn as sea
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

setStuff = pd.read_csv("C:\\Users\\shado_000\\hw8_data.csv")

Y = setStuff.iloc[:,0].values
X = setStuff.iloc[:,1:38]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

model = BaggingClassifier(DecisionTreeClassifier(max_depth = None), n_estimators=500, random_state=0)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
