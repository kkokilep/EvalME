import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Implementation of ReSA")

    parser.add_argument("--config_path", type=str, default="./configs/test.yaml", help="path to yaml config file")


    args = parser.parse_args()
    return args