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

    dataset_dictionary = {
    "name": "tax",
    "path": "./../datasets/tax/dirty.csv",
    "clean_path": "./../datasets/tax/clean.csv"
    }   

    dataset_par = dp.DatasetParallel(dataset_dictionary)
    dataset_par.results_folder = "./../datasets/tax/strategy-profiling-tax"
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