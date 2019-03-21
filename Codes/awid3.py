# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:08:55 2018

@author: User
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:59:39 2018

@author: User
"""

import pandas as pd
import numpy as np
seed = 42
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)
col_names=pd.read_csv("F:\\DATASET\\AWID-CLS-R-Trn\\Featurenames.csv")
#5	"frame.time_delta_displayed"
#6	"frame.time_relative"
#7	"frame.len"
#13	"radiotap.length"
#14	"radiotap.present.tsft"
#15	"radiotap.present.flags"
#17	"radiotap.present.channel"
#18	"radiotap.present.fhss"
#19	"radiotap.present.dbm_antsignal"
#25	"radiotap.present.antenna"
#28	"radiotap.present.rxflags"
#63	"wlan.fc.type_subtype"
#64	"wlan.fc.version"
#65	"wlan.fc.type"
#66	"wlan.fc.subtype"
#67	"wlan.fc.ds"
#68	"wlan.fc.frag"
#69	"wlan.fc.retry"
#70	"wlan.fc.pwrmgt"
#71	"wlan.fc.moredata"
#72	"wlan.fc.protected"
#91	"wlan_mgt.fixed.capabilities.cfpoll.ap"
#103	"wlan_mgt.fixed.listen_ival"
#105	"wlan_mgt.fixed.status_code"
#106	"wlan_mgt.fixed.timestamp"
#108	"wlan_mgt.fixed.aid"
#109	"wlan_mgt.fixed.reason_code"
#111	"wlan_mgt.fixed.auth_seq"
#114	"wlan_mgt.fixed.chanwidth"
#123	"wlan_mgt.tim.bmapctl.offset"
#124	"wlan_mgt.country_info.environment"
#132	"wlan_mgt.rsn.capabilities.ptksa_replay_counter"
#133	"wlan_mgt.rsn.capabilities.gtksa_replay_counter"
#147	"wlan.qos.ack"
#154	"class"

to_int16=['"radiotap.present.reserved"',
            '"wlan.fc.type_subtype"',
            '"wlan.fc.ds"',
            '"wlan_mgt.fixed.capabilities.cfpoll.ap"',
           '"wlan_mgt.fixed.listen_ival"',
            '"wlan_mgt.fixed.status_code"', 
            '"wlan_mgt.fixed.timestamp"', 
            '"wlan_mgt.fixed.aid"', 
            '"wlan_mgt.fixed.reason_code"',
            '"wlan_mgt.fixed.auth_seq"',
            '"wlan_mgt.fixed.htact"', 
            '"wlan_mgt.fixed.chanwidth"',
            '"wlan_mgt.tim.bmapctl.offset"',
            '"wlan_mgt.country_info.environment"',
            '"wlan_mgt.rsn.capabilities.ptksa_replay_counter"', 
            '"wlan_mgt.rsn.capabilities.gtksa_replay_counter"',
                        '"wlan.qos.ack"' ]

train = pd.read_csv("F:\\DATASET\\AWID-CLS-R-Trn\\trn.csv",names =col_names)
test = pd.read_csv("F:\\DATASET\\AWID-CLS-R-Trn\\tst.csv",names =col_names)
train = train.replace(['?'], [0])

test=test.replace(['?'], [0])



from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for column in train:
	#print(column)
	#print(dataframe[column].dtype, column)	
	if column in to_int16:
		train[column] = train[column].apply(lambda x: int(str(x), base=16) if x != '?' else x)
		test[column] = test[column].apply(lambda x: int(str(x), base=16) if x != '?' else x)
	if column == '"class"':
		test[column] = encoder.fit_transform(test[column])
		train[column] = encoder.fit_transform(train[column])
   
    
#    p2=print(encoder.classes_)
#0: 'flooding' 1: 'impersonation'  2: 'injection' 3: 'normal'

#Observe the statistical values     
p=test.describe()

p1=train.describe()
#index_list=[3,6,7,28, 37,46,61,65,66,67,69,71,72,76,79,81,87,92,93,97, 103, 106,107,111,112,121,124,125,126,139,140,141,143,147]
#35 variable with nonzero statistical measurement
stat_nonzero=[5,6,7,13,14,15,17,18,19,25,28,33, 63,64,65,66,67,68,69,70,71,72,91
,103,105,106,108,109,111,113, 114,123,124,132,133,147,154]
train1=train.iloc[:,stat_nonzero]
test1=test.iloc[:,stat_nonzero]
float_col=['"frame.time_relative"','"wlan_mgt.fixed.timestamp"','"wlan_mgt.fixed.reason_code"']
def log_trns(df, col):
    return df[col].apply(np.log1p)
for col in float_col:
    train1[col] = log_trns(train1, col)
    test1[col] = log_trns(test1, col)

train1 = train1.apply(lambda col:pd.to_numeric(col, errors='coerce'))
test1 =test1.apply(lambda col:pd.to_numeric(col, errors='coerce'))
for column in train1:
	print(train1[column].dtype, column)	

#frames=[train1,test1]
#result = pd.concat(frames)
#result = result.sample(frac=1).reset_index(drop=True)
#result.to_csv('exp_dataset.csv')
#dataset = result.values
#X = dataset[:,0:-1]
#y = dataset[:,-1]
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test =train_test_split (X, y, test_size = 0.25, random_state = 0)



X_train = train1.iloc[:,0:-1].values
y_train = train1.iloc[:,-1]


X_test=test1.iloc[:,0:-1].values
y_test=test1.iloc[:,-1]


#print(pd.DataFrame(y_train).hist(bins=4))
pd.Series(y_train).value_counts(bins=4)
pd.Series(y_test).value_counts(bins=4)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#one-hot encoded output classes
y_train= pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values
#test1=test.values
#X_test=test1[:,0:-1]
#y_test=test1[:,-1]


# Designing of the model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adadelta,SGD,Adam,RMSprop

encoding_dim =24
input_img = Input(shape=(36,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(36, activation='sigmoid')(encoded)
encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(loss='binary_crossentropy', optimizer = 'adadelta')
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=50,
                validation_split=0.2)

autoencoder.save_weights('autoencoder.h5')

def fc(enco):
    
    den = Dense(36, activation = 'relu')(enco)
    den1=Dense(24,activation = 'relu')(den)
    in1=BatchNormalization()(den1)
    out = Dense(4, activation='softmax')(in1)
    return out
encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

for l1,l2 in zip(full_model.layers[:10],autoencoder.layers[0:10]):
    l1.set_weights(l2.get_weights())
    
for layer in full_model.layers[0:10]:
    layer.trainable = False
    
full_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
full_model.summary()   
    

history=full_model.fit(x=X_train, y=y_train, epochs=50, validation_split=0.2, batch_size=50)
full_model.save_weights('autoencoder_classification.h5')


for layer in full_model.layers[0:5]:
    layer.trainable = True
full_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
full_model.summary()   
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')
history=full_model.fit(x=X_train, y=y_train, epochs=50, validation_split=0.2, batch_size=50)
full_model.save_weights('classification_complete.h5')

#NN.fit(x=train, y=target, epochs=100, validation_split=0.1, batch_size=128, callbacks=[early_stopping])
#plot accuracy graph for different iterations
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 6), dpi=80)

print(history.history.keys())
accuracy=[ history.history['acc'], history.history['val_acc'],history.history['loss'],history.history['val_loss']]
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'], '--')
plt.title('Model accuracy with Dense ANN',fontsize=16)
plt.ylabel('Average Accuracy [0-1]',fontsize=16)
plt.xlabel('No. of Epoch',fontsize=16)
plt.ylim((.85,1))
plt.legend(['Training Acuracy', 'Validation Accuracy'], loc='lower right')
plt.tight_layout()
plt.show()



from sklearn.metrics import confusion_matrix,classification_report,roc_curve

# With the confusion matrix, we can aggregate model predictions
# This helps to understand the mistakes and refine the model
preds = full_model.predict(X_test)
y_pred = (preds > 0.5).astype(int)
pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test, axis=1)

cm= confusion_matrix(true_lbls, pred_lbls)
full_model.evaluate(X_test, y_test)
target_names =['flooding','impersonation' ,'injection' ,'Normal']
p2=print(classification_report(true_lbls, pred_lbls,target_names=target_names))



