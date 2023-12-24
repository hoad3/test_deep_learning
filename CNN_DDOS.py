!pip install tensorflow

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn import preprocessing

import os
dataset_path = []

for dirname, _, filenames in os.walk('/content/drive/MyDrive/archive(1)'):
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

y_train_one_hot = to_categorical(y_train, num_classes=18)
y_test_one_hot = to_categorical(y_test, num_classes=18)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten


# ... (Các bước khác như import và xử lý dữ liệu)

# Chuyển đổi nhãn thành one-hot encoding nếu sử dụng categorical_crossentropy

features = samples.drop(' Label', axis=1)
labels = samples[' Label']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN với softmax activation và categorical_crossentropy loss
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(18, activation='softmax'))  # num_classes là số lượng lớp của mô hình

# Biên soạn mô hình với categorical_crossentropy loss
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
with tf.device('/GPU:1'):
    history = model.fit(X_train_scaled, y_train_one_hot, epochs=10, batch_size=128, validation_split=0.2)

# Đánh giá mô hình trên tập kiểm tra
accuracy = model.evaluate(X_test_scaled, y_test_one_hot)
print(f"Accuracy on test set: {accuracy[1]*100:.2f}%")



from keras.models import load_model

# Đường dẫn đến file model
model.save('/content/drive/MyDrive/CNN_Model.h5')


import matplotlib.pyplot as plt

# ... (Các bước import và xử lý dữ liệu)


# Lấy giá trị loss và accuracy từ lịch sử huấn luyện
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Vẽ sơ đồ Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Vẽ sơ đồ Accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Dự đoán xác suất cho từng lớp trên tập kiểm tra
y_prob = model.predict(X_test_scaled)

# Chọn lớp có xác suất cao nhất cho mỗi mẫu
y_pred = y_prob.argmax(axis=1)

# Tính toán các metric
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
