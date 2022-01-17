import os
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout
from keras.layers import Flatten, Dense,BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

def base(length, out_length):


    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 3e-4
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    
    x = Embedding(output_dim=ed, input_dim=5, input_length=length, name='embedding')(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value), name='conv1')(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same', name='maxpool1')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value), name='conv2')(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same', name='maxpool2')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value), name='conv3')(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same', name='maxpool3')(c)

    merge = Concatenate(axis=-1, name='con')([apool, bpool, cpool])

    x = Dropout(dp, name='dropout')(merge)

  
    x = Flatten(name='flatten')(x)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def base_feature(length, length_a, out_length):

    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 3e-4
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    fea_input = Input(shape=(length_a,), name='fea_input')

    x = Embedding(output_dim=ed, input_dim=5, input_length=length, name='embedding')(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value), name='conv1')(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same', name='maxpool1')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value), name='conv2')(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same', name='maxpool2')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value), name='conv3')(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same', name='maxpool3')(c)

    merge = Concatenate(axis=-1, name='con')([apool, bpool, cpool])

    x = Dropout(dp, name='dropout')(merge)

    x = Flatten(name='flatten')(x)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    x = Dense(32)(x)

    x_normal = BatchNormalization()(x)

    fea_cnn3 = Dense(32, activation='relu', W_regularizer=l2(l2value))(fea_input)

    fea_cnn3_normal = BatchNormalization()(fea_cnn3)

    x = Concatenate(axis=-1, name='lastLayer')([x_normal, fea_cnn3_normal])

    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=[main_input, fea_input], output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model

