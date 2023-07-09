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
    cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1, processes=True, memory_limit='2GB')
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
    clusters = det.build_clusters(dataset_par, [])
    start_time = time.time()
    while len(dataset_par.labeled_tuples) < det.LABELING_BUDGET:
        det.sample_tuple(dataset_par, clusters)
        if dataset_par.has_ground_truth:
            det.label_with_ground_truth(dataset_par)
    end_time = time.time()
    det.TIME_TOTAL += end_time-start_time
    print("Sampling tuples and labeling with ground truth(parallel): {}".format(end_time-start_time))
    det.propagate_labels(dataset_par, clusters)

    det.predict_labels(dataset_par, clusters)

    client.shutdown()
    client.close()
    print("")
    dataset = dset.Dataset(dataset_dictionary)
    dataset.results_folder = dataset_dictionary["results-folder"]
    det_single = dt.Detection()
    det_single.run_strategies(dataset)
    det_single.generate_features(dataset)
    det_single.build_clusters(dataset)

    start_time = time.time()
    while len(dataset.labeled_tuples) < det_single.LABELING_BUDGET:
        det_single.sample_tuple(dataset)
        if dataset.has_ground_truth:
            det_single.label_with_ground_truth(dataset)
    end_time = time.time()
    det_single.TIME_TOTAL += end_time-start_time
    print("Sampling tuples and labeling with ground truth(non parallel): {}".format(end_time-start_time))
    det_single.propagate_labels(dataset)
    det_single.predict_labels(dataset)
    print("Raha parallel(total):{}\nRaha non parallel(total):{}\nPerformance: {}".format(det.TIME_TOTAL, det_single.TIME_TOTAL, det.TIME_TOTAL/det_single.TIME_TOTAL))

    dataset_par.cleanup_dataset()
