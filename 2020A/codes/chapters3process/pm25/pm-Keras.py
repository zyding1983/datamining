import pandas as pd
from keras.layers import Dense,Dropout,Activation,Input
from keras.models import Sequential,Model
from sklearn.model_selection import train_test_split
from keras import metrics

def load_data():
    df_train=pd.read_csv("train.csv")

    #"D:\Class\Data Mining\King County House Price Predection\Pretreatment\FinalVersion\\train.csv")
    df_test=pd.read_csv("test.csv")

    #"D:Class\Data Mining\King County House Price Predection\Pretreatment\FinalVersion\\test.csv")
    return df_train,df_test

def make_model(InputSize):
    model=Sequential()
    model.add(Dense(units=90,activation='relu',input_shape=(InputSize,)))
    model.add(Dropout(0.1))
    model.add(Dense(units=50,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1,activation=None))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=[metrics.mae])
    print(model.summary())
    return model

if __name__ == '__main__':
    df_train,df_test=load_data()
    X_train,X_test,y_train,y_test=train_test_split(df_train.drop('pm2.5',axis=1)
                  .values,df_train['pm2.5'].values,test_size=0.1,random_state=0)

    model=make_model(12)
    model.fit(X_train,y_train,batch_size=32,epochs=60000,verbose=1,validation_data=
                                                         (X_test,y_test),shuffle=True)
    #print("111223211111111132")
    pred=model.predict(df_test.values)
    submissionFile=pd.DataFrame({'pm2.5':pred.reshape(1,-1)[0]})
    submissionFile.to_csv("Keras60000.csv",index=False)