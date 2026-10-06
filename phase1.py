import pandas as pd

data={
    "study_hours" : [4,5,3,2],
    "attendence" : [12,45,87,56],
    "sleep_hours" : [2,6,7,8],
    "previous_marks" : [45,56,15,78],
    "final_marks" : [78,56,89,98],

}

df=pd.DataFrame(data)


X=df[["study_hours","attendence","sleep_hours","previous_marks"]]
Y=df["final_marks"]

print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2)
print("traning data: ")
print(X_train)

print("testing data: ")
print(X_test)

print("traning data: ")
print(Y_train)

print("testing data: ")
print(y_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,Y_train)

model.predict(X_test)
Y_predict = model.predict(X_test)

#using  MSE which is mean squared erroe
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
import math

mae=mean_absolute_error(y_test, Y_predict)
mse=mean_squared_error(y_test,Y_predict)
rmse=math.sqrt(mse)
print("MSE:",mse)
print("MAE:",mae)
from sklearn.metrics import r2_score
r2=r2_score(y_test,Y_predict)
print("r2: ",r2)
print("RMSE:",rmse)


