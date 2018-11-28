import argparse
from train_gpu import train   # needs to be changed before sending to server
from test_small import test_small
from test_big import test_big

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true', help='train flag')
    parser.add_argument('-test', action='store_true', help='train flag')
    parser.add_argument('-data', type=str, default='big', help='which dataset, small or big')
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=100, help='max number of iterations (default 100)')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size (default 50)')
    parser.add_argument('-patience', type=int, default=10, help='early stopping patience (default 10)')
    parser.add_argument('-dev', type=str, default="1,2,3,4", help='what cpu or gpu (recommended) use to train the model')
    parser.add_argument("-gpus", "--gpus", type=int, default=4, help="# of GPUs to use for training")
    args = parser.parse_args()
    # train
    if args.train:
        train(args)


    # test on small dataset
    if args.test and args.data == "small":
        test_small(args)

    # test on original dataset
    if args.test and args.data == "big":
        test_big(args)
