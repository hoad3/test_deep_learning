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


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

k = 5  # Số láng giềng
knn_model = KNeighborsClassifier(n_neighbors=k)

with tf.device('/GPU:1'):
    history = knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
