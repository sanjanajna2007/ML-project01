import pandas as pd

data={
    "study_hours" : [4,5,3,2],
    "attendence" : [12,45,87,56],
    "sleep_hours" : [2,6,7,8],
    "previous_marks" : [45,56,15,78],
    "final_marks" : [78,56,89,98],

}

df=pd.DataFrame(data)
#print(df)
#print(df.head(2))
#print(df.shape)
#print(df.info)
#print(df.columns)

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
