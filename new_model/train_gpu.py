import os
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
from load_data import load_data_from_npz, load_batch, load_data_names, load_batch_from_names_random
from my_model import get_eye_tracker_model
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.models import save_model, load_model
from Euc_dist_loss import euclidean_distance_loss

# generator for data loaded from the npz file
def generator_npz(data, batch_size, img_ch, img_cols, img_rows):

    while True:
        for it in list(range(0, data[0].shape[0], batch_size)):
            x, y = load_batch([l[it:it + batch_size] for l in data], img_ch, img_cols, img_rows)
            yield x, y


# generator with random batch load (train)
def generator_train_data(names, path, batch_size, img_cols, img_rows, img_ch):

    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_cols, img_rows, img_ch)
        yield x, y


# generator with random batch load (validation)
def generator_val_data(names, path, batch_size, img_cols, img_rows, img_ch):

    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_cols, img_rows, img_ch)
        yield x, y


def train(args):

    #getting gpu parameters
    G = args.gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    #todo: manage parameters in main
    if args.data == "big":
        dataset_path = '../data'  #../data
        # dataset_path = r'D:\gazecapture_small'
    if args.data == "big":
        train_path = '../dataset/train'
        val_path = '../dataset/validation'
        # test_path = r"C:\Users\Aliab\PycharmProjects\data_small\test"
        # train_path = r"C:\Users\Aliab\PycharmProjects\data_small\train"
        # val_path = r"C:\Users\Aliab\PycharmProjects\data_small\validation"

    print("{} dataset: {}".format(args.data, dataset_path))
    # train parameters
    n_epoch = args.max_epoch
    batch_size = args.batch_size
    patience = args.patience

    # image parameter
    img_cols = 224
    img_rows = 224
    img_ch = 3

    # using multiGPU for training
    if G <= 1:
        print('training with 1 GPU')
        model = get_eye_tracker_model(img_cols, img_rows, img_ch)
    else:
        print('[INFO] training with {} GPUs...'.format(G))
        with tf.device('/cpu:0'):
            model = get_eye_tracker_model(img_cols, img_rows, img_ch)

        # make the multigpu model
        parallel_model = multi_gpu_model(model, gpus=G)


    # model summary

    model.summary()
    # weights
    # print("Loading weights...",  end='')
    # weights_path = "weights/weights.003-4.05525.hdf5"
    # model.load_weights(weights_path)
    # print("Done.")

    # optimizer
    # lr = 1e-3
    sgd = SGD(lr=1e-3, decay=1e-4, momentum=9e-1, nesterov=True)
    adam = Adam(1e-3, decay=1e-4)

    # Loading Previous model check
    model_file = "model.hdf5"
    if os.path.isfile(model_file):
        # load model weights
        print('model loaded successfully....')
        model.load_weights(model_file)

    # compile model
    parallel_model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

    # making variable lr for validation loss decrement
    var_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, verbose=1, mode='auto', min_lr=0)

    # Modelcheckpoint to save the weights
    Mdl_chk_pnt = ModelCheckpoint('model.hdf5', save_best_only=True, verbose=1, monitor='val_loss', mode='auto')

    # data
    # todo: parameters not hardocoded
    if args.data == "big":
        # train data
        train_names = load_data_names(train_path)
        # validation data
        val_names = load_data_names(val_path)

    # debug
    # x, y = load_batch([l[0:batch_size] for l in train_data], img_ch, img_cols, img_rows)
    # x, y = load_batch_from_names(train_names[0:batch_size], dataset_path, img_ch, img_cols, img_rows)
    # record the history

    history = History()

    # fitting the model
    i = 1
    # a = 0
    import numpy as np
    min_val_loss = np.inf
    while i <= 100:

        history = parallel_model.fit_generator(
            generator=generator_train_data(train_names, dataset_path, batch_size, img_cols, img_rows, img_ch),
            steps_per_epoch=(len(train_names)) / batch_size,
            epochs=1,
            verbose=1,
            shuffle=True,
            validation_data=generator_val_data(val_names, dataset_path, batch_size, img_cols, img_rows, img_ch),
            validation_steps=(len(val_names)) / batch_size,
            callbacks=[EarlyStopping(patience=patience), Mdl_chk_pnt, var_lr]
            )

        print(history.history['val_loss'])

        # # check if the loss reduced then save the model
        if history.history['val_loss'][0] < min_val_loss:
            min_val_loss = history.history['val_loss'][0]
            print("saving weights with val_loss {val_loss}".format(val_loss=history.history['val_loss'][0],
                                                                   val_acc=history.history['val_acc'][0]))
            save_model(model, "model.hdf5")
        # else:
        #     a = a + 1
        #     print('a value changed')
        # if history.history['val_loss'][0] >= min_val_loss and a == 2:
        #     lr = lr * 0.6
        #     print('learning rate decreased to {}'.format(lr))
        i = i + 1