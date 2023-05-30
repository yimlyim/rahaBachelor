import raha.dataset
import os
import sys

from distributed import Client, LocalCluster

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=8)
    client = Client(cluster)


    dataset_dict = {
        "name": "movies_1",
        "path": "datasets/movies_1/dirty.csv",
        "clean_path": "datasets/movies_1/clean.csv"
    }
    dset = raha.dataset.Dataset(dataset_dict)
    dframe = dset.dataframe
    print(dframe)

    cluster = LocalCluster(n_workers=4, threads_per_worker=8)
    client = Client(cluster)
    dframe = client.persist(dframe)
    
    print(dframe.compute())


    print("Hello World!")