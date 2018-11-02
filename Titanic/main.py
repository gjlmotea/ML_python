import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def splitname(data):
    ret={}
    for v in data:
        tmp=re.split('[,.]',v)
        if len(tmp)<2:
            continue
        key=tmp[1].strip()
        ret[key]=ret.get(key,0)+1
    
def Sympol_id(name):
    ret={'Sir': 1, 'Major': 2, 'the Countess': 1, 'Mlle': 2, 'Capt':1, 'Dr': 7, 'Lady': 1, 'Master': 40, 'Mme': 1,}
    tmp=re.split('[,.]',name)
    if len(tmp)<2:
        return 0
    key=tmp[1].strip()
    if key in ret:
        return 1
    return 0


train= pd.read_csv('train.csv')
test2= pd.read_csv('test_ti.csv')
test= pd.read_csv('test.csv')

train = train.drop(["Ticket"], axis=1)
test = test.drop(["Ticket"], axis=1)
test2 = test2.drop(["Ticket"], axis=1)

train["Name"] = splitname(train["Name"])
train["Name"] = Sympol_id(str(train["Name"]))
test["Name"] = splitname(test["Name"])
test["Name"] = Sympol_id(str(test["Name"]))
test2["Name"] = splitname(test2["Name"])
test2["Name"] = Sympol_id(str(test2["Name"]))

##train = train.drop(["Cabin"], axis=1)
##test = test.drop(["Cabin"], axis=1)
##test2 = test2.drop(["Cabin"], axis=1)

train["Cabin"] = train['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin').astype('category').cat.codes
test["Cabin"] = test['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin').astype('category').cat.codes
test2["Cabin"] = test2['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin').astype('category').cat.codes


train.loc[train["Sex"]=="male", "Sex"] = 1
train.loc[train["Sex"]=="female", "Sex"] = 0
test.loc[test["Sex"]=="male", "Sex"] = 1
test.loc[test["Sex"]=="female", "Sex"] = 0
test2.loc[test2["Sex"]=="male", "Sex"] = 1
test2.loc[test2["Sex"]=="female", "Sex"] = 0

train["Embarked"] = train["Embarked"].fillna("S")
test2["Embarked"] = test2["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

train.loc[train["Embarked"]=="S", "Embarked"] = 0
train.loc[train["Embarked"]=="C", "Embarked"] = 1
train.loc[train["Embarked"]=="Q", "Embarked"] = 2
test.loc[test["Embarked"]=="S", "Embarked"] = 0
test.loc[test["Embarked"]=="C", "Embarked"] = 1
test.loc[test["Embarked"]=="Q", "Embarked"] = 2
test2.loc[test2["Embarked"]=="S", "Embarked"] = 0
test2.loc[test2["Embarked"]=="C", "Embarked"] = 1
test2.loc[test2["Embarked"]=="Q", "Embarked"] = 2
test['Fare'].fillna(test['Fare'].median(),inplace=True)


average_age_train   = train["Age"].mean()
std_age_train       = train["Age"].std()
count_nan_age_train = train["Age"].isnull().sum()


average_age_test   = test["Age"].mean()
std_age_test       = test["Age"].std()
count_nan_age_test = test["Age"].isnull().sum()

average_age_test2  = test2["Age"].mean()
std_age_test2      = test2["Age"].std()
count_nan_age_test2 = test2["Age"].isnull().sum()



rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)
rand_3 = np.random.randint(average_age_test2 - std_age_test2, average_age_test2 + std_age_test2, size = count_nan_age_test2)


train.loc[np.isnan(train["Age"]), "Age"] = rand_1
test.loc[np.isnan(test["Age"]), "Age"] = rand_2
test2.loc[np.isnan(test2["Age"]), "Age"] = rand_3

train['Age'] = train['Age'].astype(int)
test['Age']  = test['Age'].astype(int)
test2['Age'] = test2['Age'].astype(int)



from sklearn import preprocessing
from keras.utils import np_utils
X_train = train.drop(["PassengerId", "Survived"], axis=1)
y_train = train["Survived"]

X_test2 = test2.drop(["PassengerId", "Survived"], axis=1)
y_test2 = test2["Survived"]

X_test  = test.drop(["PassengerId"],axis=1)

X_train = np.array(X_train).astype('float32')
y_train = np.array(y_train).astype('int32')

X_test2 = np.array(X_test2).astype('float32')
y_test2 = np.array(y_test2).astype('int32')

X_test = np.array(X_test).astype('float32')

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
X_train_norm=minmax_scale.fit_transform(X_train)
y_train_onehot = np_utils.to_categorical(y_train, 2)

X_test2_norm=minmax_scale.fit_transform(X_test2)
y_test2_onehot = np_utils.to_categorical(y_test2, 2)

X_test_norm = minmax_scale.fit_transform(X_test)


from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense,Dropout
model = Sequential()
model.add(Dense(units=512,input_dim=9,kernel_initializer='random_uniform',activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(units=128,kernel_initializer='random_uniform',activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(units=32,kernel_initializer='random_uniform',activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(units=2,kernel_initializer='random_uniform',activation='softmax'))
model.summary()



from keras import optimizers
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])



import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
checkpointer = ModelCheckpoint(filepath='NNModel.h5', verbose=1, save_best_only=True)
early=EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)
call_list=[checkpointer,early]
train_history = model.fit(x=X_train_norm,
                         y=y_train_onehot,
                         validation_split=0.1,
                         epochs=100,
                         batch_size=128,verbose=1)



import matplotlib.pyplot as plt  
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()


show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
scores = model.evaluate(x=X_test2_norm,
                        y=y_test2_onehot)
scores[1]
prediction = model.predict_classes(X_test2_norm)  # Making prediction and save result to prediction
import pandas as pd  
print("\t[Info] Display Confusion Matrix:")  
print("%s\n" % pd.crosstab(y_test2, prediction, rownames=['label'], colnames=['predict']))
Y_pred=model.predict_classes(X_test_norm)
s=({"PassengerId":test["PassengerId"],"Survived":Y_pred})
submit=pd.DataFrame(data=s)
submit.to_csv('titanic.csv',index=False)
