from tensorflow import keras
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout, Convolution2D, \
    MaxPooling2D, Reshape
from keras.layers import Flatten, Dense, Activation, BatchNormalization, GRU, LSTM, Lambda
from keras import Model, Sequential
from keras.regularizers import l2, l1_l2
from keras.optimizers import Adam, SGD
from keras.layers import Bidirectional
import keras.backend as K
import numpy as np
from sklearn import metrics
import tensorflow as tf
from keras.layers import Lambda

# gridsearch
def base_feature_1(ed=100, ps=5, lr=3e-4, fd_2=32):
    fd = 128
    dp = 0.5
    l2value = 0.001

    def slice1(x, index):
        return x[:, :index]

    def slice2(x, index):
        return x[:, index:]

    indata = Input(shape=(106,))

    main_input = Lambda(slice1, output_shape=(99,), arguments={'index': 99})(indata)
    fea_input = Lambda(slice2, output_shape=(7,), arguments={'index': 99})(indata)

    x = Embedding(output_dim=ed, input_dim=5, input_length=99, name='embedding')(main_input)

    a = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), name='conv1')(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool1')(a)

    b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value), name='conv2')(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool2')(b)

    c = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value), name='conv3')(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool3')(c)

    merge = Concatenate(axis=-1, name='con')([apool, bpool, cpool])

    x = Dropout(dp, name='dropout')(merge)
    x = Flatten(name='flatten')(x)
    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)
    x = Dense(32)(x)

    x_normal = BatchNormalization()(x)
    fea_cnn3 = Dense(32, activation='relu', kernel_regularizer=l2(l2value))(fea_input)
    fea_cnn3_normal = BatchNormalization()(fea_cnn3)
    x = Concatenate(axis=-1, name='lastLayer')([x_normal, fea_cnn3_normal])

    output = Dense(1, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)

    model = Model(inputs=indata, outputs=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def base_feature():
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 3e-4
    l2value = 0.001

    main_input = Input(shape=(99,), dtype='int64', name='main_input')

    fea_input = Input(shape=(7,), name='fea_input')

    x = Embedding(output_dim=ed, input_dim=5, input_length=99, name='embedding')(main_input)

    a = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), name='conv1')(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool1')(a)

    b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value), name='conv2')(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool2')(b)

    c = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value), name='conv3')(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool3')(c)

    merge = Concatenate(axis=-1, name='con')([apool, bpool, cpool])

    x = Dropout(dp, name='dropout')(merge)

    x = Flatten(name='flatten')(x)

    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)

    x = Dense(32)(x)

    x_normal = BatchNormalization()(x)

    fea_cnn3 = Dense(32, activation='relu', kernel_regularizer=l2(l2value))(fea_input)

    fea_cnn3_normal = BatchNormalization()(fea_cnn3)

    x = Concatenate(axis=-1, name='lastLayer')([x_normal, fea_cnn3_normal])

    output = Dense(1, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input, fea_input], outputs=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def cnn_fea_block(length, length_a):
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 3e-4
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    fea_input = Input(shape=(length_a,), name='fea_input')

    x = Embedding(output_dim=ed, input_dim=5, input_length=length, name='embedding')(main_input)

    a = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), name='conv1')(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool1')(a)

    b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value), name='conv2')(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool2')(b)

    c = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value), name='conv3')(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same', name='maxpool3')(c)

    merge = Concatenate(axis=-1, name='con')([apool, bpool, cpool])

    x = Dropout(dp, name='dropout')(merge)

    x = Flatten(name='flatten')(x)

    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)
    x = Dense(32)(x)
    x_normal = BatchNormalization()(x)

    fea_cnn3 = Dense(32, activation='relu', kernel_regularizer=l2(l2value))(fea_input)
    fea_cnn3_normal = BatchNormalization()(fea_cnn3)

    x = Concatenate(axis=-1, name='lastLayer')([x_normal, fea_cnn3_normal])

    output = Dense(32, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input, fea_input], outputs=output, name='cnn_fea')

    return model


def clf_block(name, length):
    l2value = 0.001

    input = Input(shape=(length,))
    output = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2value))(input)

    model = Model(inputs=input, outputs=output, name=name)
    return model


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * margin_square + (1 - y_true) * square_pred)


def Transfer_fea_MT(in_length, f_shape, weight=0.2):
    input_t = Input(shape=(in_length,))
    input_t_f = Input(shape=(f_shape,))

    input_s = Input(shape=(in_length,))
    input_s_f = Input(shape=(f_shape,))

    cf_block = cnn_fea_block(in_length, f_shape)
    clfs = clf_block('clf_s', 32)
    clft = clf_block('clf_t', 32)

    x_s = cf_block([input_s, input_s_f])
    x_t = cf_block([input_t, input_t_f])

    s_pred = clfs(x_s)
    t_pred = clft(x_t)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='distance')([x_t, x_s])

    model = Model(inputs=[input_t, input_t_f, input_s, input_s_f], outputs=[t_pred, s_pred])
    adam = Adam(3e-4)

    model.compile(loss={'clf_t': 'binary_crossentropy', 'clf_s': 'binary_crossentropy'},
                  optimizer=adam,
                  loss_weights={'clf_t': 1 - weight, 'clf_s': weight}, metrics=['accuracy'])
    return model
