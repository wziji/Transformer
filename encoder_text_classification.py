# 文本分类实验

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from Encoder import *
import pandas as pd



# 1. 数据信息
max_features = 20000
maxlen = 64
batch_size = 32


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", \
                                                      num_words=max_features)
y_train, y_test = pd.get_dummies(y_train), pd.get_dummies(y_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)



# 2. 构造模型，及训练模型

inputs = Input(shape=(64,), dtype='int32')
embeddings = Embedding(max_features, 128)(inputs)

print("\n"*2)
print("embeddings:")
print(embeddings)

mask_inputs = padding_mask(inputs)

out_seq = Encoder(2, 128, 4, 256, maxlen)(embeddings, mask_inputs, False)

print("\n"*2)
print("out_seq:")
print(out_seq)

out_seq = GlobalAveragePooling1D()(out_seq)

print("\n"*2)
print("out_seq:")
print(out_seq)

out_seq = Dropout(0.3)(out_seq)
outputs = Dense(64, activation='relu')(out_seq)

out_seq = Dropout(0.3)(out_seq)
outputs = Dense(16, activation='relu')(out_seq)

out_seq = Dropout(0.3)(out_seq)
outputs = Dense(2, activation='softmax')(out_seq)

model = Model(inputs=inputs, outputs=outputs)
print(model.summary())


opt = Adam(lr=0.0002, decay=0.00001)
loss = 'categorical_crossentropy'
model.compile(loss=loss,
             optimizer=opt,
             metrics=['accuracy'])


print('Train...')
history = model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=10,
         validation_data=(x_test, y_test))
