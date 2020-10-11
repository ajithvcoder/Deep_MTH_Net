import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--train_epoch", type=int, default=70)
    parser.add_argument('--momentum', type=float, default=0.9)

    return parser