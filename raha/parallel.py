import raha
import time
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

def test(i):
    time.sleep(5)
    return i

def heavytest(i):
    kwargs = {'sep': ',', 'header':'infer', 'encoding':'utf-8', 'dtype': str, 'keep_default_na': False, 'low_memory': False}

    dset = dd.read_csv(urlpath='datasets/movies_1/dirty.csv', blocksize="16MB", **kwargs)
    dset = get_client().compute(dset)

    time.sleep(5)
    return dset

def testpickle(i):
    print("Worker {} attaches to shared mem...".format(get_worker().id))
    sma = sm.SharedMemory(name='Testarea', create=False)
    print("Worker {} attached to shared mem.".format(get_worker().id))
    shared_obj = (sharedMemArea.buf.tobytes())[:len(dsetBytes)]
    deserialized = pickle.loads(shared_obj)


    if i == 1:
        deserialized = deserialized.applymap(lambda x: x+"WORKER 111111")
    else:
        deserialized = deserialized.applymap(lambda x: x+"WORKER 222222")

    print("Loaded Object:\n{}".format(deserialized))
    sma.close()
    return 1

def testnonpickle(i):
    print("Worker {} attaches to shared mem...".format(get_worker().id))
    print("Worker {} attached to shared mem.".format(get_worker().id))
    kwargs = {'sep': ',', 'header':'infer', 'encoding':'utf-8', 'dtype': str, 'keep_default_na': False, 'low_memory': False}
    dset = dd.read_csv(urlpath='datasets/movies_1/dirty.csv', blocksize="16MB", **kwargs)
    dset = get_client().compute(dset).result()
    print("Loaded Object:\n{}".format(dset))
    return 1

if __name__ == "__main__":
    """
    #Test futures as batch tasks
    cluster = LocalCluster(n_workers=4, threads_per_worker=8)
    client = Client(cluster)
  
    tasks = []
    for i in range(1, 15):
        tasks.append(client.submit(test, i))
    print(tasks)
    print(client.gather(futures=tasks, direct=True))

    heavytasks = []
    for i in range(1, 15):
        heavytasks.append(client.submit(heavytest, i))
    print(heavytasks)
    finalizedtasks = client.gather(futures=heavytasks, direct=True)
    print(finalizedtasks)
    print(client.gather(futures=finalizedtasks, direct=True))
    """
    
    #Test shared memory
    cluster = LocalCluster(n_workers=4, threads_per_worker=8)
    client = Client(cluster)
    
    kwargs = {'sep': ',', 'header':'infer', 'encoding':'utf-8', 'dtype': str, 'keep_default_na': False, 'low_memory': False}
    dset = dd.read_csv(urlpath='./../datasets/movies_1/dirty.csv', blocksize="16MB", **kwargs)
    dset = client.compute(dset).result()
    dset_size = sys.getsizeof(dset)

    print(dset)
    sharedMemArea = sm.SharedMemory(name='Testarea', create=True, size=dset_size)
    dsetBytes = pickle.dumps(dset)
   
    for i in range(0, len(dsetBytes)):
        sharedMemArea.buf[i] = dsetBytes[i]


    tasks = []
    for i in range(1, 3):
        tasks.append(client.submit(testpickle, i))
    print(tasks)
    print(client.gather(futures=tasks, direct=True))
    time.sleep(5)

    shared_obj = (sharedMemArea.buf.tobytes())[:len(dsetBytes)]
    deserialized = pickle.loads(shared_obj)
    print(deserialized)
    print(type(deserialized))

    print(sys.getsizeof(sharedMemArea))
    sharedMemArea.close()
    sharedMemArea.unlink()

    