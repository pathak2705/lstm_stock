from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import math
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# partid,locid,stampid,date,amount

df=pd.read_csv(r"C:\Users\pathak\python\Machine Learning\lstm-stock\ge.us.txt",index_col=0)
new_df=pd.DataFrame(df,columns=('partid','locid','stampid','amount'))
new_df['partid']=df['Open']
new_df['locid']=df['High']
new_df['stampid']=df['Low']
new_df['amount']=df['Close']
#making index as date
new_df.index=df.index
new_df.index.name='date'
new_df
#new_df.to_csv(r"C:\Users\pathak\python\Machine Learning\lstm-stock\new_df.csv")

#converting to numpy array
new_df=new_df.values
new_df
len(new_df)

#function to create lookback dataset
def create_dataset(dataset, look_back=1):
  dataX, dataY = [],[]
  for i in range(look_back,len(dataset)):
      dataX.append(dataset[i-look_back:i,:])
      #print("1")
      dataY.append(dataset[i,:])
  return np.array(dataX), np.array(dataY)

#dividing dataset into train and test
# n_train_time = 365*24  # not so good prediction
n_train_time = 365*30    # excellent prediction
train = new_df[:n_train_time, :]
train
test = new_df[n_train_time:, :]
test

# passing train and test into lookback function
train_X, train_y = create_dataset(train,look_back=1)
train_y
test_X, test_y = create_dataset(test,look_back=1)
test_X

# reshape input to be 3D [samples, timesteps, features]
train_X.shape[0]
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], train_X.shape[2]))#shape[0]=no of rows, shape[2]=no of columns
train_X.shape[1]
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], test_X.shape[2]))
train_X.shape[2]

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
#    model.add(LSTM(70))
#    model.add(Dropout(0.3))
model.add(Dense(train_y.shape[1])) #train_y.shape[1] because output units are 4
model.compile(loss='mean_squared_error', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)


#predicted results
predicted=model.predict(test_X)
#considering only 4th column(i.e. closing price) for comparison
predicted_new=predicted[:,3:]
predicted
test_y_new=test_y[:,3:]
test_y

#visualising predicted vs actual
plt.plot(test_y_new,color="red",label="actual data")
plt.plot(predicted_new,color="blue",label="predicted data")
plt.title("stock prediction")
plt.xlabel("Time")
plt.ylabel("stock price")
plt.legend()
plt.show()