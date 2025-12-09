import argparse
import torch
import dataset


def main(device, args):
    print(args.e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", default=5)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/brandon/Documents/dev/PodPure/model/data"
    metadata = data_dir + "/metadata.csv"
    data_fp = data_dir + "/podcast_files/"

    data = dataset.podDataset(metadata, data_fp)

    main(device, args)
