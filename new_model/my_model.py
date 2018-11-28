import numpy as np
from keras.layers import Layer
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, concatenate, BatchNormalization, ZeroPadding2D, Dropout
from keras.models import Model

# activation functions
activation = 'elu'
last_activation = 'linear'

def get_eye_model(img_cols, img_rows, img_ch):

    eye_img_input = Input(shape=(img_cols, img_rows, img_ch))
    h = Conv2D(96, (3, 3), activation=activation)(eye_img_input)
    h = BatchNormalization()(h)
    h = MaxPool2D(pool_size=(2,2))(h)
    h = Conv2D(256, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(384, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(64, (3, 3), activation=activation)(h)
    out = BatchNormalization()(h)

    model = Model(inputs=eye_img_input, outputs=out)

    return model

def get_face_model(img_cols, img_rows, img_ch):

    face_img_input = Input(shape=(img_cols, img_rows, img_ch))
    h = ZeroPadding2D(padding=(1, 1))(face_img_input)
    h = Conv2D(64, (3,3), activation=activation)(h)
    h = ZeroPadding2D(padding=(1,1))(h)
    h = BatchNormalization()(h)
    h = Conv2D(64, (3,3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(h)

    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(128, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(128, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(h)


    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(256, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(256, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(256, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(h)

    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(512, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(512, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(512, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(h)

    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(512, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(512, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(512, (3, 3), activation=activation)(h)
    h = BatchNormalization()(h)
    out = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(h)

    model = Model(inputs=face_img_input, outputs=out)

    return model

# final model
def get_eye_tracker_model(img_cols, img_rows, img_ch):
    # get partial models
    eye_net = get_eye_model(img_cols, img_rows,img_ch)
    face_net_part = get_face_model(img_cols, img_rows,img_ch)

    # right eye model
    right_eye_input = Input(shape=(img_cols, img_rows,img_ch))
    right_eye_net = eye_net(right_eye_input)

    # left eye model
    left_eye_input = Input(shape=(img_cols, img_rows,img_ch))
    left_eye_net = eye_net(left_eye_input)

    # face model
    face_input = Input(shape=(img_cols, img_rows,img_ch))
    face_net = face_net_part(face_input)

    # face grid
    face_grid = Input(shape=(1, 25, 25))

    # dense layers for eyes
    e = concatenate([left_eye_net, right_eye_net])
    e = Flatten()(e)
    fc_e1 = Dense(128, activation=activation)(e)
    fc_e1 = BatchNormalization()(fc_e1)

    # dense layers for face
    f = Flatten()(face_net)
    fc_f1 = Dense(256, activation=activation)(f)
    fc_f1 = BatchNormalization()(fc_f1)
    fc_f2 = Dense(128, activation=activation)(fc_f1)
    fc_f2 = BatchNormalization()(fc_f2)
    fc_f3 = Dense(64, activation=activation)(fc_f2)
    fc_f3 = BatchNormalization()(fc_f3)

    # dense layers for face grid
    fg = Flatten()(face_grid)
    fc_fg1 = Dense(256, activation=activation)(fg)
    fc_fg1 = BatchNormalization()(fc_fg1)
    fc_fg2 = Dense(128, activation=activation)(fc_fg1)
    fc_fg2 = BatchNormalization()(fc_fg2)


    # final dense layers
    h = concatenate([fc_e1, fc_f3, fc_fg2])
    fc1 = Dense(128, activation=activation)(h)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.3)(fc1)
    fc2 = Dense(2, activation=last_activation)(fc1)

    # final model
    final_model = Model(
        inputs=[right_eye_input, left_eye_input, face_input, face_grid],
        outputs=[fc2])

    return final_model
