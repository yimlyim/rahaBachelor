import os
import re
import sys
import math
import time
import json
import random
import pickle
import hashlib
import tempfile
import itertools
import multiprocessing

import numpy
import pandas
import scipy.stats
import scipy.spatial
import scipy.cluster
import sklearn.svm
import sklearn.tree
import sklearn.cluster
import sklearn.ensemble
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.kernel_ridge
import sklearn.neural_network
import sklearn.feature_extraction
from constants import *
import constants
from dask.distributed import get_client
from multiprocessing import shared_memory as sm
import dask
from dask.distributed import get_worker
import dask.dataframe
import dataset as dset

########################################
class DetectionParallel:
    def __init__(self):
        self.LABELING_BUDGET = 20
        self.USER_LABELING_ACCURACY = 1.0
        self.VERBOSE = False
        self.SAVE_RESULTS = True
        self.CLUSTERING_BASED_SAMPLING = True
        self.STRATEGY_FILTERING = False
        self.CLASSIFICATION_MODEL = "GBC"  # ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
        self.LABEL_PROPAGATION_METHOD = HOMOGENEITY   # ["homogeneity", "majority"]
        self.ERROR_DETECTION_ALGORITHMS = [OUTLIER_DETECTION, PATTERN_VIOLATION_DETECTION, 
                                           RULE_VIOLATION_DETECTION, KNOWLEDGE_BASE_VIOLATION_DETECTION]
        self.HISTORICAL_DATASETS = []
    
    @staticmethod
    def create_shared_dataframe(dataframe_filepath, mem_area_name):
        """
        Creates a shared memory area and stores the dataframe in serialized form byte-wise in there.
        """   
        MB_1 = 1e6
        num_partitions = 10
        client = get_client()
        filesize = os.path.getsize(dataframe_filepath)     
        
        #Aim for 10 partitions
        blocksize = filesize / num_partitions if filesize >= num_partitions else MB_1 
        print("Blocksize of " + dataframe_filepath + " is:" + str(blocksize)) 

        #Read DataFrame in parallel
        kwargs = {'sep': ',', 'header':'infer', 'encoding':'utf-8', 'dtype': str, 'keep_default_na': False, 'low_memory': False}
        dataset = dask.dataframe.read_csv(urlpath=dataframe_filepath, blocksize=int(blocksize), **kwargs).applymap(dset.Dataset.value_normalizer)
        dataset = client.compute(dataset).result()

        pickled_dataset = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
        pickled_dataset_size = len(pickled_dataset)
        print("Size of pickled dataframe " + str(pickled_dataset_size))

        shared_mem_area = sm.SharedMemory(name=mem_area_name, create=True, size=pickled_dataset_size)
        shared_mem_area.buf[:pickled_dataset_size] = pickled_dataset

        shared_mem_area.close()
        del shared_mem_area
        return mem_area_name

    @staticmethod
    def create_shared_split_dataset(dataframe_ref):
        dataframe = DetectionParallel.load_shared_dataframe(dataframe_ref)

        for column in dataframe.columns.tolist():
            pickled_dataset = pickle.dumps(dataframe[column], protocol=pickle.HIGHEST_PROTOCOL)
            pickled_dataset_size = len(pickled_dataset)
            shared_mem_area = sm.SharedMemory(name=column, create=True, size=pickled_dataset_size)
            shared_mem_area.buf[:pickled_dataset_size] = pickled_dataset
            shared_mem_area.close()
            del shared_mem_area
        return



    @staticmethod
    def get_column_names(dataframe_filepath):
        return pandas.read_csv(dataframe_filepath, nrows=0).columns.tolist()

    @staticmethod
    def load_shared_dataframe(dataframe_ref):

        shared_mem_area = sm.SharedMemory(name=dataframe_ref, create=False)
        deserialized_frame = pickle.loads(shared_mem_area.buf)

        shared_mem_area.close()
        del shared_mem_area
        return deserialized_frame

    #Todo
    def run_outlier_strategy(self):
        return {}
        
    #Todo
    def run_pattern_strategy(self, configuration, dataframe_ref, dataframe_filepath):
        """
        """
        outputted_cells = {}
        column_name, character = configuration
        dataframe = DetectionParallel.load_shared_dataframe(column_name)
        j = DetectionParallel.get_column_names(dataframe_filepath).index(column_name)

        for i, value in dataframe.items():
            try:
                if len(re.findall("[" + character + "]", value, re.UNICODE)) > 0:
                    outputted_cells[(i, j)] = ""
            except:
                print("Error occured in run_pattern_strategy in worker  " + str(get_worker().id))
                continue

        return outputted_cells

    #Todo
    def run_rule_strategy(self):
        return {}

    #Todo        
    def run_knowledge_strategy(self):
        return {}

    def parallel_strat_runner_process(self, args):
        """
        Runs all error detection strategies in a seperate worker process.
        """
        start_time = time.time()
        outputted_cells = {}
        dataframe_ref, algorithm, configuration = args
        strategy_name = json.dumps([algorithm, configuration])
        strategy_name_hashed = str( int( hashlib.sha1( strategy_name.encode("utf-8")).hexdigest(), base=16))

        match algorithm:
            case constants.OUTLIER_DETECTION:
                #Run outlier detection strategy
                run_outlier_strategy()
            case constants.PATTERN_VIOLATION_DETECTION:
                #Run pattern violation detection strategy
                #Needs only column dataframe
                run_pattern_strategy(configuration, dataframe_ref)
            case constants.RULE_VIOLATION_DETECTION:
                #Run rule violation detection strategy
                #NEEDS WHOLE DATAFRAME
                run_rule_strategy()
            case constants.KNOWLEDGE_BASE_VIOLATION_DETECTION:
                #Run knowledge base violation strategy
                run_knowledge_strategy()
            case _:
                raise ValueError("Algorithm " + str(algorithm) + " is not supported!")
        
        end_time = time.time()
        strategy_results = {
            "name": strategy_name,
            "output": list(outputted_cells.keys()),
            "runtime": end_time - start_time
        }
        
        return strategy_results
    
    @staticmethod
    def setup_outlier_metadata(dataframe_ref):
        """
        Worker-Process in a parallel manner. Creates Gaussian configuration and Histogram configuration and return them.
        """
        configurations = []
        
        #Create Cartesian Product
        cartesian_config = [
                        list(a) for a in
                        list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                                                ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
                        list(itertools.product(["gaussian"],
                                                ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]

        configurations.extend([dataframe_ref, OUTLIER_DETECTION, conf] for conf in cartesian_config)
        return configurations

    @staticmethod
    def pattern_violation_worker(dataframe_ref, column_name, dataframe_filepath):
        """
        Worker-Process in a parallel manner. Extracts all characters of one specific column and returns them.
        """
        configurations = []
        client = get_client()

        #Load Shared DataFrame column by accessing shared memory area, named by column_name
        dataset_column = DetectionParallel.load_shared_dataframe(column_name)

        #Concatenate all content of a column into a long string
        column_data = "".join(dataset_column.tolist())
        #Notice which character appeared in this column
        character_dict = {character: 1 for character in column_data} 
        character_dict_list = [[column_name, character] for character in character_dict]  

        configurations.extend([dataframe_ref, PATTERN_VIOLATION_DETECTION, conf] for conf in character_dict_list)
        del dataset_column, character_dict, column_data
        return configurations

    @staticmethod
    def setup_pattern_violation_metadata(dataframe_ref, dataframe_filepath):
        """
        Calculates Meta-Data for pattern-violation application later on
        """
        futures = []
        configurations = []
        client = get_client()
        column_names = DetectionParallel.get_column_names(dataframe_filepath)
        
        #Call a worker for each column name
        arguments1 = [dataframe_ref] * len(column_names)
        arguments2 = [column_name  for column_name in column_names]
        arguments3 = [dataframe_filepath] * len(column_names)
        futures.append(client.map(DetectionParallel.pattern_violation_worker, arguments1, arguments2, arguments3))

        #Gather results and merge "list of lists" into one big list
        results_unmerged = client.gather(futures=futures, direct=True)[0]
        results = list(itertools.chain.from_iterable(results_unmerged))

        return results
    @staticmethod
    def setup_rule_violation_metadata(dataframe_ref, dataframe_filepath):
        """
        Calculates Meta-Data for rule-violation application later on.
        This method just creates pairs of column names as potential FDs
        """
        configurations = []
        column_names = DetectionParallel.get_column_names(dataframe_filepath)
        column_pairs = [[col1, col2] for (col1,col2) in itertools.product(column_names, column_names) if col1 != col2]
        configurations.extend([[dataframe_ref, RULE_VIOLATION_DETECTION, column_pair] for column_pair in column_pairs])

        return configurations
    @staticmethod
    def setup_knowledge_violation_metadata(dataframe_ref, dataframe_filepath):
        configurations = []
        paths = [os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base", path) for path in os.listdir(os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base"))]
        configurations.extend([[dataframe_ref, KNOWLEDGE_BASE_VIOLATION_DETECTION, path] for path in paths])

        return configurations

    def run_strategies(self, dataset):
        """
        Creates strategies metadata and executes each strategy for a seperate worker process
        """
        #TODO - Implement strategy filtering if-else clause + preloading results
        starttime = time.time()
        strategy_profile_path = os.path.join(dataset.results_folder, "strategy-profiling")
        client = get_client()
        futures = []

        for algorithm_name in self.ERROR_DETECTION_ALGORITHMS:
            match algorithm_name:
                case constants.OUTLIER_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_outlier_metadata, dataset.dirty_ref))
                case constants.PATTERN_VIOLATION_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_pattern_violation_metadata, dataset.dirty_ref, dataset.path))
                case constants.RULE_VIOLATION_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_rule_violation_metadata, dataset.dirty_ref, dataset.path))
                case constants.KNOWLEDGE_BASE_VIOLATION_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_knowledge_violation_metadata, dataset.dirty_ref, dataset.path))
                case _:
                    raise ValueError("Algorithm " + str(algorithm) + " is not supported!")
        
        
        results = list(itertools.chain.from_iterable(client.gather(futures=futures, direct=True)))
        endtime = time.time()
        print(results)
        print("Raha strategy metadata generation(parallel): "+  str(endtime - starttime))
        #Start Detecting Errors in parallel
        #futures = client.map(self.parallel_strat_runner_process, results)
        #strategy_profiles = client.gather(futures=futures, direct=True)


        return

########################################
