import os
from load_data import load_batch, load_data_names, load_batch_from_names, load_batch_from_names_random
from my_model import get_eye_tracker_model
import numpy as np
from keras.models import load_model
from keras.optimizers import SGD, adam

def generator(data, batch_size, img_cols, img_rows, img_ch):

    while True:
        for it in list(range(0, data[0].shape[0], batch_size)):
            x, y = load_batch([l[it:it + batch_size] for l in data], img_cols, img_rows, img_ch)
            yield x, y


def test_big(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
    names_path = r"C:\Users\Aliab\PycharmProjects\data\test"
    print("Names to test: {}".format(names_path))

    dataset_path = r"D:\GazeCapture"
    print("Dataset: {}".format(names_path))

    weights_path = "weight_vgg.hdf5"
    print("Weights: {}".format(weights_path))

    # image parameter
    img_cols = 128
    img_rows = 128
    img_ch = 3

    # test parameter
    batch_size = 64
    chunk_size = 500

    # model
    model = get_eye_tracker_model(img_cols, img_rows, img_ch)

    # model summary
    model.summary()

    # weights
    print("Loading weights...")
    model = load_model(weights_path)

    model.load_weights(weights_path)
    # data
    test_names = load_data_names(names_path)

    # limit amount of testing data
    # test_names = test_names[:1000]

    # results
    err_x = []
    err_y = []

    print("Loading testing data...")
    for it in list(range(0, len(test_names), chunk_size)):

        x, y = load_batch_from_names_random(test_names[it:it + chunk_size],  dataset_path, batch_size, img_cols, img_rows, img_ch)
        # x, y = load_batch_from_names(test_names[it:it + chunk_size], dataset_path, img_ch, img_cols, img_rows)
        predictions = model.predict(x=x, batch_size=batch_size, verbose=1)

        # print and analyze predictions
        for i, prediction in enumerate(predictions):
            print("PR: {} {}".format(prediction[0], prediction[1]))
            print("GT: {} {} \n".format(y[i][0], y[i][1]))

            err_x.append(abs(prediction[0] - y[i][0]))
            err_y.append(abs(prediction[1] - y[i][1]))

    # mean absolute error
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)

    # standard deviation
    std_x = np.std(err_x)
    std_y = np.std(err_y)

    # final results
    print("MAE: {} {} ( samples)".format(mae_x, mae_y))
    print("STD: {} {} ( samples)".format(std_x, std_y))


if __name__ == '__main__':
    test_big()
