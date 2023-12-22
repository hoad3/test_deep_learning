!pip install tensorflow

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn import preprocessing

import os
dataset_path = []

for dirname, _, filenames in os.walk('/content/drive/MyDrive/archive(1)/train'):
    for filename in filenames:
        if filename.endswith('.csv'):
            dfp = os.path.join(dirname, filename)
            dataset_path.append(dfp)

# Load từ thư mục test
for dirname, _, filenames in os.walk('/content/drive/MyDrive/archive(1)/test'):
    for filename in filenames:
        if filename.endswith('.csv'):
            dfp = os.path.join(dirname, filename)
            dataset_path.append(dfp)


cols = list(pd.read_csv(dataset_path[1], nrows=1))

def load_file(path):
    # data = pd.read_csv(path, sep=',')
    data = pd.read_csv(path,
                   usecols =[i for i in cols if i != " Source IP"
                             and i != ' Destination IP' and i != 'Flow ID'
                             and i != 'SimillarHTTP' and i != 'Unnamed: 0'])

    return data

samples = pd.concat([load_file(dfp) for dfp in dataset_path], ignore_index=True)
print(samples.info())

import pandas as pd
import matplotlib.pyplot as plt


label_counts = samples[' Label'].value_counts()


plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar')
plt.title('Comparison of Label Column')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

def string2numeric_hash(text):
    import hashlib
    return int(hashlib.md5(text).hexdigest()[:8], 16)
samples = samples.replace('Infinity','0')
samples = samples.replace(np.inf,0)

samples[' Flow Packets/s'] = pd.to_numeric(samples[' Flow Packets/s'])

samples['Flow Bytes/s'] = samples['Flow Bytes/s'].fillna(0)
samples['Flow Bytes/s'] = pd.to_numeric(samples['Flow Bytes/s'])


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


samples[' Label'] = label_encoder.fit_transform(samples[' Label'])


label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

for label, encoded_value in label_mapping.items():
    print(f"Label: {label} - Encoded Value: {encoded_value}")

colunaTime = pd.DataFrame(samples[' Timestamp'].str.split(' ',1).tolist(), columns = ['dia','horas'])
colunaTime = pd.DataFrame(colunaTime['horas'].str.split('.',1).tolist(),columns = ['horas','milisec'])
stringHoras = pd.DataFrame(colunaTime['horas'].str.encode('utf-8'))
samples[' Timestamp'] = pd.DataFrame(stringHoras['horas'].apply(string2numeric_hash))
del colunaTime,stringHoras


print('Training data processed')


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

features = samples.drop(' Label', axis=1)

labels = samples[' Label']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

import tensorflow as tf


# In danh sách các GPU có sẵn
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Embedding,Dropout,Flatten,MaxPooling1D,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

# Load and preprocess your data (replace this with your actual data loading and preprocessing)
# Assuming 'samples' contains your features and labels
features = samples.drop(' Label', axis=1)
labels = samples[' Label']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Callback to print training progress
class PrintTrainingProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
        print(f" - Loss: {logs['loss']:.4f}")
        print(f" - Accuracy: {logs['accuracy']:.4f}")
        print(f" - Validation Loss: {logs['val_loss']:.4f}")
        print(f" - Validation Accuracy: {logs['val_accuracy']:.4f}")

# Model configuration
input_shape = (X_train.shape[1], 1)
model = Sequential()
# Thêm lớp Conv1D với 256 bộ lọc, kích thước cửa số là 3 và hàm kích hoạt là 'relu'
model.add(Conv1D(256, kernel_size=3, activation='relu', input_shape=input_shape))

# Thêm lớp LSTM với 128 đơn vị
model.add(LSTM(256))
model.add(Dropout(0.2))

# Lớp kết nối đầy đủ với 1 đơn vị và hàm kích hoạt softmax (cho bài toán đa lớp)
model.add(Dense(18, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with tf.device('/GPU:1'):
    history = model.fit(X_train, y_train_one_hot, epochs=40, batch_size=128, validation_data=(X_test, y_test_one_hot), verbose=1, callbacks=[PrintTrainingProgress()])
# Training the model with verbose

import matplotlib.pyplot as plt

model.save('drive/MyDrive/LSTM_Model.h5')

# ... (import và định nghĩa model, dữ liệu và callback như trong đoạn code của bạn)

# Huấn luyện mô hình và lưu thông tin lịch sử

# Biểu đồ độ mất mát
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Accuracy')
plt.plot(history.history['accuracy'], label='Validation Accuracy')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
