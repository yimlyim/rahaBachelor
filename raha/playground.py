import raha
import raha.dataset
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
    """
    dataset_dictionary = {
    "name": "tax",
    "path": "./../datasets/tax/dirty.csv",
    "clean_path": "./../datasets/tax/clean.csv",
    "results-folder": "./../datasets/tax/strategy-profiling-toy"
    }   

    dataset_par = dp.DatasetParallel(dataset_dictionary)
    dataset_par.results_folder = dataset_dictionary["results-folder"]
    dp.DatasetParallel.create_shared_dataset(dataset_par)
    dataset_par = dp.DatasetParallel.load_shared_dataset(dataset_par.own_mem_ref)

    current = time.time()
    area = dp.DatasetParallel.create_shared_dataframe(dataset_par.dirty_path, dataset_par.dirty_mem_ref)
    dataframe_loaded = dp.DatasetParallel.load_shared_dataframe(dataset_par.dirty_mem_ref)
    after = time.time()
    print("Serializing and deserializing took: " + str(after - current))
    #print(dataframe_loaded)

    dp.DatasetParallel.create_shared_split_dataframe(dataset_par.dirty_mem_ref)
    for column in dataframe_loaded.columns.tolist():
        print("Loaded column data:" + column)
        print(type(dp.DatasetParallel.load_shared_dataframe(column)))
        print("")

    dataset = raha.dataset.Dataset(dataset_dictionary)
    dataset.dataframe = dataframe_loaded
    dataset.dirty_mem_ref = dataset_dictionary["name"]
    dataset.results_folder = "./../datasets/tax/strategy-profiling-tax"

    det_non_par = dt.Detection()
    det_non_par.run_strategies(dataset)

   
    det = detection_parallel.DetectionParallel()
    det.run_strategies(dataset_par)

    col_test_name = "l_name"
    res1start = time.time()
    res1 = str(dataframe_loaded.columns.get_loc(col_test_name))
    res1end = time.time()
    res2start = time.time()
    res2 = str(dp.DatasetParallel.get_column_names("./../datasets/tax/dirty.csv").index(col_test_name))
    res2end = time.time()
    print("Index of column " + col_test_name + "with get_loc: " + res1 + " in time: " + str(res1end-res1start))
    print("Index of column "+ col_test_name + "with columns.tolist(): " + res2 + " in time: " + str(res2end-res2start))
    
    # Test one instance of PVD config, roughly 150KB per dict returned(maybe store on disk?)
    test_config = [['tax', 'PVD', ['f_name', 'F']], ['tax', 'PVD', ['f_name', 'V']], ['tax', 'PVD', ['f_name', 'E']], ['tax', 'PVD', ['f_name', 'G']]]
    #print("Test run_pattern:")
    #print(det.run_pattern_strategy(test_config[1][2], dataset_par.own_mem_ref))
    print("Dirtyref:{} \nDatasetref:{}".format(dataset_par.dirty_mem_ref, dataset_par.own_mem_ref))
    """
    ####Test Toy on run###
    dataset_dictionary = {
    "name": "tax",
    "path": "./../datasets/flights/dirty.csv",
    "clean_path": "./../datasets/flights/clean.csv",
    "results-folder": "./../datasets/flights/raha-baran-results-flights"
    }   

    dataset_par = dp.DatasetParallel(dataset_dictionary)
    dataset_par.results_folder = dataset_dictionary["results-folder"]
    dp.DatasetParallel.create_shared_dataset(dataset_par)
    dataset_par = dp.DatasetParallel.load_shared_dataset(dataset_par.own_mem_ref)

    area = dp.DatasetParallel.create_shared_dataframe(dataset_par.dirty_path, dataset_par.dirty_mem_ref)
    dataframe_loaded = dp.DatasetParallel.load_shared_dataframe(dataset_par.dirty_mem_ref)

    dp.DatasetParallel.create_shared_split_dataframe(dataset_par.dirty_mem_ref)
    for column in dataframe_loaded.columns.tolist():
        print("Loaded column data:" + column)
        print(dp.DatasetParallel.load_shared_dataframe(column))
        print("")

    det = detection_parallel.DetectionParallel()
    det.run_strategies(dataset_par)

    df = pandas.read_csv("./../datasets/tax/dirty.csv", sep=",", header="infer", encoding="utf-8", dtype=str,
                                    keep_default_na=False, low_memory=False).applymap(dp.DatasetParallel.value_normalizer)
    print(df)
    print(sys.getsizeof(df)/1000000)


    main_frame_area = sm.SharedMemory(name=dataset_par.dirty_mem_ref, create=False)
    main_frame_area.close()
    main_frame_area.unlink()
    del main_frame_area

    main_frame_area = sm.SharedMemory(name=dataset_par.own_mem_ref, create=False)
    main_frame_area.close()
    main_frame_area.unlink()
    del main_frame_area

    for col_name in dp.DatasetParallel.get_column_names(dataset_dictionary["path"]):
        col_frame_area = sm.SharedMemory(name=col_name, create=False)
        col_frame_area.close()
        col_frame_area.unlink()
        del col_frame_area

    client.shutdown()
    client.close()