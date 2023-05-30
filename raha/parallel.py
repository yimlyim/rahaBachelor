from raha import *
import os
import sys

from distributed import Client, LocalCluster

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=8)
    client = Client(cluster)


    dataset_name = "flights"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }

    data = raha.dataset.Dataset(dataset_dictionary)
    print(data.dataframe)

    print("Hello World!")