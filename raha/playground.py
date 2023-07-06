import raha
import raha.dataset as dset
import raha.detection as dt
import time
import itertools
import os
import sys
import dask
from multiprocessing import shared_memory as sm
from distributed import Client, LocalCluster
import dask.dataframe as dd
from dask.distributed import get_client
import pickle
import pandas
import numpy
from dask.distributed import get_worker
from constants import *
import tempfile
import detection_parallel
import constants
import dataset_parallel as dp

deserialized = None
def some_local_func(i):
    return i+1

def parallel(i):
    print("This is the type: "+ str(type(deserialized)))
    return some_local_func(i)


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=8, threads_per_worker=1, processes=True, memory_limit='2GB')
    client = Client(cluster)

    dataset_dictionary = {
    "name": "movies_1",
    "path": "./../datasets/movies_1/dirty.csv",
    "clean_path": "./../datasets/movies_1/clean.csv",
    "results-folder": "./../datasets/movies_1/raha-baran-results-movies_1"
    }   

    dataset_par = dp.DatasetParallel(dataset_dictionary)
    dataset_par.results_folder = dataset_dictionary["results-folder"]
    dataset_par.initialize_dataset()

    dataframe_loaded = dp.DatasetParallel.load_shared_dataframe(dataset_par.dirty_mem_ref)
    """
    for column in dataframe_loaded.columns.tolist():
        print("Loaded column data:" + column)
        print(dp.DatasetParallel.load_shared_dataframe(column))
        print("")
    """

    det = detection_parallel.DetectionParallel()
    #print(dataframe_loaded)
    print("x_frame:{} , y_frame {} || x_shared:{} , y_shared:{}".format(dataframe_loaded.shape[0], dataframe_loaded.shape[1], dataset_par.dataframe_num_rows, dataset_par.dataframe_num_cols))
    strategies = det.run_strategies(dataset_par)
    det.generate_features(dataset_par, strategies)
    det.build_clusters(dataset_par, [])

    client.shutdown()
    client.close()

    dataset = dset.Dataset(dataset_dictionary)
    dataset.results_folder = dataset_dictionary["results-folder"]
    det_single = dt.Detection()
    det_single.run_strategies(dataset)
    det_single.generate_features(dataset)
    det_single.build_clusters(dataset)

    dataset_par.cleanup_dataset()
