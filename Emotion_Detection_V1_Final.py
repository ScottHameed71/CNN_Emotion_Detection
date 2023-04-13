#INITIALIZING THE LIBRARIES

import pandas as pda
import numpy as npy
import scikitplot
import seaborn as sns
import keras
import tensorflow as tfw
from tensorflow.keras.utils import to_categorical
import warnings
from tensorfxlow.keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
from tensorflow.keras.utils import load_img
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization,MaxPooling2D,Activation,Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.simplefilter("ignore")
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.regularizers import l1, l2
import plotly.express as pxs
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


##LOADING THE DATA


data = pda.read_csv("/Users/saadhameed/PycharmProjects/Capstone3/fer2013.csv")
data.shape


##DATA CHECK
data_check = data.isnull().sum()
print(data_check)


##DATA HEAD VALIDATION
data_head = data.head()
print (data_head)


##PRE-PROCESSING OF THE DATA
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
fig = pxs.bar(x = CLASS_LABELS,
            y = [list(data['emotion']).count(i) for i in npy.unique(data['emotion'])] ,
            color = npy.unique(data['emotion']) ,
            color_continuous_scale="Emrld")
fig.update_xaxes(title="Emotions")
fig.update_yaxes(title = "Number of Images")
fig.update_layout(showlegend = True,
   title = {
       'text': 'Train Data Distribution ',
       'y':0.95,
       'x':0.5,
       'xanchor': 'center',
       'yanchor': 'top'})
fig.show()


##APPLYING SAMPLING FUNCTION TO SHUFFLE THE DATA


data = data.sample(frac=1)


#CONVERTING CATEGORICAL LABELS INTO NUMERICAL TO OPTIMIZE THE ALGORITHM


labels = to_categorical(data[['emotion']], num_classes=7)




#NUMPY ARRAY CONVERSION OF IMAGE PIXELS


train_pixels = data["pixels"].astype(str).str.split(" ").tolist()
train_pixels = npy.uint8(train_pixels)




#ARRAY STANDARDIZATION TO MAKE MEAN = 0 & SD AS UNIT


pixels = train_pixels.reshape((35887*2304,1))


scaler = StandardScaler()
pixels = scaler.fit_transform(pixels)




#DATA POPULATION SPLIT
pixels = train_pixels.reshape((35887, 48, 48,1))


X_train, X_test, y_train, y_test = train_test_split(pixels, labels, test_size=0.1, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)


#DATA SAMPLE VALIDATION


plt.figure(figsize=(15,23))
label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}
i = 1
for i in range (7):
   img = npy.squeeze(X_train[i])
   plt.subplot(1,7,i+1)
   plt.imshow(img)
   index = npy.argmax(y_train[i])
   plt.title(label_dict[index])
   plt.axis('off')
   i += 1
plt.show()


#DATA AUGMENTATION TO PREVENT OVERFITTING USING IMAGEDATAGENERATOR


datagen = ImageDataGenerator(  width_shift_range = 0.1,
                              height_shift_range = 0.1,
                              horizontal_flip = True,
                              zoom_range = 0.2)
valgen = ImageDataGenerator(   width_shift_range = 0.1,
                              height_shift_range = 0.1,
                              horizontal_flip = True,
                              zoom_range = 0.2)


datagen.fit(X_train)
valgen.fit(X_val)


train_generator = datagen.flow(X_train, y_train, batch_size=64)
val_generator = datagen.flow(X_val, y_val, batch_size=64)


#MODEL CREATION USING CONVULOTIONAL NEURAL NETWORK CNN


def cnn_model():
   model = tfw.keras.models.Sequential()
   model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
   model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
   model.add(BatchNormalization())
   model.add(MaxPool2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))


   model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
   model.add(BatchNormalization())
   model.add(MaxPool2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))


   model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
   model.add(BatchNormalization())
   model.add(MaxPool2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))


   model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
   model.add(BatchNormalization())
   model.add(MaxPool2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))


   model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
   model.add(BatchNormalization())
   model.add(MaxPool2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))


   model.add(Flatten())
   model.add(Dense(256, activation='relu'))
   model.add(BatchNormalization())
   model.add(Dropout(0.25))


   model.add(Dense(512, activation='relu'))
   model.add(BatchNormalization())
   model.add(Dropout(0.25))


   model.add(Dense(7, activation='softmax'))
   model.compile(
       optimizer=Adam(lr=0.0001),
       loss='categorical_crossentropy',
       metrics=['accuracy'])
   return model


model = cnn_model()


model.compile(
   optimizer = Adam(lr=0.0001),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


model.summary()




# PREVENT OVERFITTING BY ADDING EARLY STOPPING


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 5),
               ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                               save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=30,
                   batch_size=64,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


###VISUALIZING RESULTS FOR BASELINE CNN MODEL


plt.plot(history.history["loss"],'r', label="Training Loss")
plt.plot(history.history["val_loss"],'b', label="Validation Loss")
plt.legend()


plt.plot(history.history["accuracy"],'r',label="Training Accuracy")
plt.plot(history.history["val_accuracy"],'b',label="Validation Accuracy")
plt.legend()


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )


label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(npy.random.choice(X_test.shape[0], size=24, replace=False)):
   ax = figure.add_subplot(4, 6, i + 1, xticks=[], yticks=[])
   ax.imshow(npy.squeeze(X_test[index]))
   predict_index = label_dict[(y_pred[index])]
   true_index = label_dict[npy.argmax(y_test, axis=1)[index]]


   ax.set_title("{} ({})".format((predict_index),
                                 (true_index)),
                color=("green" if predict_index == true_index else "red"))






#CONFUSION MATRIX FOR THE BASELINE CNN MODEL

CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis=1), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index=CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize=(15, 10))
plt.title('Confusion Matrix', fontsize=20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


#PERFORMANCE METRICS FOR THE BASELINE CNN MODEL

from sklearn.metrics import classification_report
print(classification_report(npy.argmax(y_test, axis=1), y_pred, digits=3))


#BASELINE MODEL COMPARISON: USING STOCHASTIC GRADIENT DESCENT (SGD) WITH THE AIM OF COMPARING TO ADAM

model = cnn_model()


model.compile(optimizer=tfw.keras.optimizers.SGD(0.001),
               loss='categorical_crossentropy',
               metrics = ['accuracy'])


history = model.fit(train_generator,
                   epochs=30,
                   batch_size=64,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


plt.plot(history.history["loss"],'r', label="Training Loss")
plt.plot(history.history["val_loss"],'b', label="Validation Loss")
plt.legend()


plt.plot(history.history["accuracy"],'r',label="Training Accuracy")
plt.plot(history.history["val_accuracy"],'b',label="Validation Accuracy")
plt.legend()


#OPTIMIZATION- PARAMETRIC TUNING USING ADAM & INCREASING EPOCHS FROM 30 TO 50


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.0001),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=50,
                   batch_size=64,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )




#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM & INCREASING EPOCHS FROM 30 TO 50
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))


#OPTIMIZATION- PARAMETRIC TUNING USING ADAM & INCREASING EPOCHS FROM 50 TO 75


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.0001),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=64,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )




#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM & INCREASING EPOCHS FROM 50 TO 75
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

#OPTIMIZATION- PARAMETRIC TUNING USING ADAM & DECREASING BATCH SIZE FROM 64 TO 32


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.0001),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=32,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )




#CONFUSION MATRIX FOR UOPTIMIZATION- PARAMETRIC TUNING USING ADAM & DECREASING BATCH SIZE FROM 64 TO 32
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

#OPTIMIZATION- PARAMETRIC TUNING USING ADAM & DECREASING BATCH SIZE FROM 32 TO 16


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.0001),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=16,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )

#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM & DECREASING BATCH SIZE FROM 32 TO 16
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

#OPTIMIZATION- PARAMETRIC TUNING USING ADAM & DECREASING BATCH SIZE FROM 16 TO 8


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.0001),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=8,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )

#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM & DECREASING BATCH SIZE FROM 16 TO 8
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

#OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.0002


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.0002),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=16,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )

#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.0002
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

#OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.0009


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.0009),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=16,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )

#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.0009
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

#OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.001


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.001),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=16,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )

#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.001
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

#OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.002


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.002),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=16,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )

#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.002
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

#OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.0025


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.0025),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=16,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )

#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.0025
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

#OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.0021


model = cnn_model()
model.compile(
   optimizer = Adam(lr=0.0021),
   loss='categorical_crossentropy',
   metrics=['accuracy'])


checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                             restore_best_weights=True,mode="max",patience = 10),
                             ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                             save_best_only=True,mode="max")]


history = model.fit(train_generator,
                   epochs=75,
                   batch_size=16,
                   verbose=1,
                   callbacks=[checkpointer],
                   validation_data=val_generator)


loss = model.evaluate(X_test,y_test)
print("Test Acc: " + str(loss[1]))


preds = model.predict(X_test)
y_pred = npy.argmax(preds , axis = 1 )

#CONFUSION MATRIX FOR OPTIMIZATION- PARAMETRIC TUNING USING ADAM, BATCH SIZE 16 AND INCREASED LR APROACH FROM 0.00001 TO LR 0.0021
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


cm_data = confusion_matrix(npy.argmax(y_test, axis = 1 ), y_pred)
cm = pda.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (20,10))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')


print(classification_report(npy.argmax(y_test, axis = 1 ),y_pred,digits=3))

