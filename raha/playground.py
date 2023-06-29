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

deserialized = None
def some_local_func(i):
    return i+1

def parallel(i):
    print("This is the type: "+ str(type(deserialized)))
    return some_local_func(i)


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=10, threads_per_worker=2, processes=True)
    client = Client(cluster)
    """
    print("Hello World!")
    configuration_list = [
    list(a) for a in
    list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                           ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
    list(itertools.product(["gaussian"],
                           ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]
    #print(configuration_list)
    dataset = raha.dataset.Dataset.read_csv_dataset("./../datasets/toy/dirty.csv")
    dataset = dataset.compute()
    datadict = {}
    for i, j, k in dataset.values.tolist():
        datadict[(i, j, k)] = ""
    print(datadict)
    print(OUTLIER_DETECTION, PATTERN_VIOLATION_DETECTION, HOMOGENEITY)

    tasks = []
    numbers = [i for i in range(1, 3)]
    tasks.append(client.map(parallel, numbers))
    print(tasks)
    print(client.gather(futures=tasks, direct=True))
    print(os.path.join(tempfile.gettempdir(), "tax" + "-" + "1234567.csv"))

    configuration_list = [
                    list(a) for a in
                    list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                                            ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
                    list(itertools.product(["gaussian"],
                                            ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]
    print(configuration_list)

    kwargs = {'sep': ',', 'header':'infer', 'encoding':'utf-8', 'dtype': str, 'keep_default_na': False, 'low_memory': False}
    dset = dd.read_csv(urlpath='./../datasets/tax/dirty.csv', blocksize="16MB", **kwargs)
    dset = client.compute(dset).result()
    dset_size = sys.getsizeof(dset)

    print(dset)
    sharedMemArea = sm.SharedMemory(name='Testarea', create=True, size=dset_size)
    dsetBytes = pickle.dumps(dset)
   
    for i in range(0, len(dsetBytes)):
        sharedMemArea.buf[i] = dsetBytes[i]



    print(sys.getsizeof(sharedMemArea))
    numbers = [i for i in range(1, 3)]
    tasks.append(client.map(parallel, numbers))
    print(tasks)
    print(client.gather(futures=tasks, direct=True))
    
    det = detection_parallel.DetectionParallel()
    ret = det.setup_outlier_metadata("test")
    starttime = time.time()
    obj = det.load_shared_dataframe("Testarea")
    endtime = time.time()
    print(obj)
    print("Time needed for loading object: " + str(endtime - starttime))
    word = "".join(obj["l_name"].tolist())
    #print(word)
    print("Columns: " + str(pandas.read_csv("./../datasets/tax/dirty.csv", nrows=0).columns.tolist()))
    print("We are at: "  + __file__)

    character_dict = {ch: 1 for ch in "".join(obj["l_name"].tolist())}
    character_dict_list = [["l_name", character] for character in character_dict] 
    print(character_dict_list)

    dictlist = []
    for ch in character_dict:
        dictlist.append(["l_name", ch])
    print(dictlist)

    results_list = det.setup_pattern_violation_metadata("Testarea", "./../datasets/tax/dirty.csv")
    print(results_list)

    dataset_dictionary = {
    "name": "tax",
    "path": "./../datasets/tax/dirty.csv",
    "clean_path": "./../datasets/tax/clean.csv"
    }   
    dataset = raha.dataset.Dataset(dataset_dictionary)
    dataset.dataframe = obj
    dataset.results_folder = "./../datasets/tax/strategy-profiling-tax"
    det.run_strategies(dataset)
    det_non_par = dt.Detection()
    det_non_par.run_strategies(dataset)
    """
    #al = obj.columns.tolist()
    #print("\n\nTesting RVD itertools.product in run_strats")
    #print(det.setup_rule_violation_metadata("Testarea", "./../datasets/movies_1/dirty.csv"))
    current = time.time()
    area = detection_parallel.DetectionParallel.create_shared_dataframe("./../datasets/movies_1/dirty.csv", "movies")
    dataframe_loaded = detection_parallel.DetectionParallel.load_shared_dataframe("movies")
    after = time.time()
    print("Serializing and deserializing took: " + str(after - current))
    #print(dataframe_loaded)

    detection_parallel.DetectionParallel.create_shared_split_dataset("movies")
    for column in dataframe_loaded.columns.tolist():
        print("Loaded column data:" + column)
        print(detection_parallel.DetectionParallel.load_shared_dataframe(column).values)
        print("")

    current = time.time()
    area = detection_parallel.DetectionParallel.create_shared_dataframe("./../datasets/tax/dirty.csv", "tax")
    dataframe_loaded = detection_parallel.DetectionParallel.load_shared_dataframe("tax")
    after = time.time()
    print("Serializing and deserializing took: " + str(after - current))
    #print(dataframe_loaded)

    detection_parallel.DetectionParallel.create_shared_split_dataset("tax")
    for column in dataframe_loaded.columns.tolist():
        print("Loaded column data:" + column)
        print(detection_parallel.DetectionParallel.load_shared_dataframe(column).values)
        print("")

    dataset_dictionary = {
    "name": "tax",
    "path": "./../datasets/tax/dirty.csv",
    "clean_path": "./../datasets/tax/clean.csv"
    }   
    dataset = raha.dataset.Dataset(dataset_dictionary)
    dataset.dataframe = dataframe_loaded
    dataset.dirty_ref = dataset_dictionary["name"]
    dataset.results_folder = "./../datasets/tax/strategy-profiling-tax"
    det = detection_parallel.DetectionParallel()
    det.run_strategies(dataset)

    det_non_par = dt.Detection()
    det_non_par.run_strategies(dataset)

    time.sleep(100)
    client.close()
    sharedMemArea.close()
    sharedMemArea.unlink()