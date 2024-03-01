import pandas as pd
import pickle

df=pd.read_csv('mushrooms.csv')

df.info()
df.describe()
df=df.dropna() # Preprocessing
df.isnull().sum()

from sklearn.preprocessing import LabelEncoder # fun defination for str to int
def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    print(feat.name,le.classes_)
#     print(le.classes_)
    return le.transform(feat)

for col in df.columns:# fun call
    df[str(col)] = label_encoded(df[str(col)])


import seaborn as sns # Corelation Matrix related or depended of columns
import matplotlib.pyplot as plt 
plt.figure(figsize=(12,10))
ax = sns.heatmap(df.corr())

fig = plt.figure(figsize = (20,15)) # Histogram - Destribution of dataset
ax = fig.gca()
df.hist(ax=ax)
plt.show()

#copy of Dataset
cdf=df.drop(['class','veil-type','gill-attachment','ring-type','gill-color','bruises'],axis=1)

#X = df.drop(['class','veil-type','gill-attachment','ring-type','gill-color','bruises'],axis=1)

# independent 
X = df[['cap-shape','cap-surface','cap-color','odor','gill-spacing','gill-size','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-color','ring-number','spore-print-color','population']]
X.info()

#dependent
y = df['class']


#40% Test data 60% Train Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.4,random_state=42)

from sklearn.ensemble import RandomForestClassifier

model_1 = RandomForestClassifier(max_depth=10, random_state=10)
model_1.fit(X_train, y_train)
y_pred = model_1.predict(X_test)

y_pred = model_1.predict([[5,3,0,5,0,0,0,0,2,2,2,7,2,2,7,1]])

y_pred = model_1.predict([[5,2,4,6,0,1,0,3,2,2,7,7,2,1,2,3]])

y_pred = model_1.predict([[5,2,3,5,1,0,1,3,2,2,7,7,2,1,3,0]])

print(y_pred)

from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)
y_pred = linearRegression.predict([[5,2,4,6,0,1,0,3,2,2,7,7,2,1,2,3]])
print(y_pred)
lracc = linearRegression.score(X_train, y_train)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=44)
model.fit(X_train, y_train)
y_pred = model.predict([[5,2,4,6,0,1,0,3,2,2,7,7,2,1,2,3]])
print(y_pred)
dtacc = model.score(X_train, y_train)

from sklearn.svm import SVR
SVM = SVR()
SVM.fit(X_train, y_train)
y_pred = SVM.predict([[5,2,3,5,1,0,1,3,2,2,7,7,2,1,3,0]])
SVMacc = SVM.score(X_train, y_train)


# Accuracy
Linear_Regression=round(linearRegression.score(X_train, y_train), 4)
Random_Forest=round(model.score(X_train, y_train), 4)
Support_Vector_Machine=round(SVM.score(X_train, y_train), 4)


data = {'LogisticRegression':Linear_Regression, 'SVC':Support_Vector_Machine, 'Random Forest':Random_Forest}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color =['black', 'red', 'green'], 
        width = 0.4)
 
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Accuracy of Algorithms")
plt.show()


file=open('my_model.pkl','wb')
pickle.dump(model_1,file,protocol=3)
file.close()