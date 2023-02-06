from model import train_util
from model import test_util

def main():
    train = True
    test = False

    if train:
        train_util.train_start()
    if test:
        test_util.test_start()

if __name__ == '__main__':
    main()
