########################################
# Dataset
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# October 2017
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import re
import os
import sys
import html
import shutil

import pandas
import dask
import dask.dataframe as dd
import constants
from distributed import Client, LocalCluster
import dataset as dset
import pickle
from multiprocessing import shared_memory as sm
from dask.distributed import get_worker
from dask.distributed import get_client
import hashlib
########################################


########################################
class DatasetParallel:
    """
    The dataset class.
    """

    def __init__(self, dataset_dictionary):
        """
        The constructor creates a dataset.
        """
        self.name = dataset_dictionary["name"]
        self.own_mem_ref = self.hash_with_salt(constants.DATASET_MEMORY_REF)
        self.dirty_mem_ref = self.hash_with_salt(dataset_dictionary["name"])
        self.clean_mem_ref = self.hash_with_salt(dataset_dictionary["name"] + '-clean')
        self.differences_dict_mem_ref = self.dirty_mem_ref + "-differences-dict"
        self.dirty_path = dataset_dictionary["path"]
        self.dictionary = dataset_dictionary
        self.dataframe_num_rows = 0
        self.dataframe_num_cols = 0
        self.labeled_tuples = {}
        self.labeled_cells = {}
        self.labels_per_cluster = {}
        self.detected_cells = {}
        self.results_folder = os.path.join(os.path.dirname(dataset_dictionary["path"]), "raha-baran-results-" + dataset_dictionary["name"])

        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

        if "clean_path" in dataset_dictionary:
            self.has_ground_truth = True
            self.clean_path = dataset_dictionary["clean_path"]
        if "repaired_path" in dataset_dictionary:
            self.has_been_repaired = True
            self.repaired_path = dataset_dictionary["repaired_path"]

    def initialize_dataset(self):
        """
        Creates Shared-Memory areas and loads the corresponding dataframe into it.
        For each column one area is created, also one for the whole dataframe with all columns
        Stores its own object into shared memory
        """
        self.create_shared_dataframe(self.dirty_path, self.dirty_mem_ref, dataset=self)
        self.create_shared_dataframe(self.clean_path, self.clean_mem_ref)
        self.create_shared_split_dataframe(self.dirty_mem_ref)
        self.create_shared_dataset(self)
        differences_dict = self.get_dataframes_difference(self.load_shared_dataframe(self.dirty_mem_ref), self.load_shared_dataframe(self.clean_mem_ref))
        self.create_shared_object(differences_dict, self.differences_dict_mem_ref)

    def cleanup_dataset(self):
        """
        Cleans up shared memory areas, which were created for computation.
        """
        #Clean-Up whole dirty Dataframe.
        try:
            main_frame_area = sm.SharedMemory(name=self.dirty_mem_ref, create=False)
            main_frame_area.close()
            main_frame_area.unlink()
            del main_frame_area
        except Exception as e: print(e)

        #Clean-Up whole clean Dataframe
        try:
            clean_frame_area = sm.SharedMemory(name=self.clean_mem_ref, create=False)
            clean_frame_area.close()
            clean_frame_area.unlink()
            del clean_frame_area
        except Exception as e: print(e)

        #Clean-Up differences-dict
        try:
            diff_dict_area = sm.SharedMemory(name=self.differences_dict_mem_ref, create=False)
            diff_dict_area.close()
            diff_dict_area.unlink()
        except Exception as e: print(e)

        #Clean-Up feature vectors
        try:
            for j in range(self.dataframe_num_cols):
                feature_frame_area = sm.SharedMemory(name= self.dirty_mem_ref + "-feature-result-" + str(j), create=False)
                feature_frame_area.close()
                feature_frame_area.unlink()
                del feature_frame_area
        except Exception as e: print(e)

        #Clean-Up Seperate Column-Dataframes.
        try:
            for col_name in DatasetParallel.get_column_names(self.dirty_path):
                col_frame_area = sm.SharedMemory(name=col_name, create=False)
                col_frame_area.close()
                col_frame_area.unlink()
                del col_frame_area
        except Exception as e: print(e)      

        #Clean-Up strategy profiles
        try:
            profiles_frame_area = sm.SharedMemory(name=self.dirty_mem_ref + "-strategy_profiles", create=False)
            profiles_frame_area.close()
            profiles_frame_area.unlink()
            del profiles_frame_area
        except Exception as e: print(e)

        #Clean-Up values that were shared for prediction phase
        try:
            predict_area = sm.SharedMemory(name=self.own_mem_ref + "-predictvariables", create=False)
            predict_area.close()
            predict_area.unlink()
        except Exception as e: print(e)

        #Clean-Up Own Dataset in Shared-Mem, does not destroy this object that is already loaded.
        try:
            dataset_frame_area = sm.SharedMemory(name=self.own_mem_ref, create=False)
            dataset_frame_area.close()
            dataset_frame_area.unlink()
            del dataset_frame_area
        except Exception as e: print(e)

    @staticmethod
    def create_shared_object(obj, name):
        pickled_obj= pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        pickled_obj_size = len(pickled_obj)
        shared_mem_area = sm.SharedMemory(name=name, create=True, size=pickled_obj_size)
        shared_mem_area.buf[:pickled_obj_size] = pickled_obj

        shared_mem_area.close()
        del shared_mem_area
        return

    @staticmethod
    def create_shared_dataset(dataset):
        """
        Creates a shared dataset object. The given dataset will be serialized and its output written into a shared memory area.
        Other Processes can obtain this area by referencing the shared memory area by its name.
        """
        pickled_dataset = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
        pickled_dataset_size = len(pickled_dataset)
        shared_mem_area = sm.SharedMemory(name=dataset.own_mem_ref, create=True, size=pickled_dataset_size)
        shared_mem_area.buf[:pickled_dataset_size] = pickled_dataset

        shared_mem_area.close()
        del shared_mem_area
        return


    @staticmethod
    def create_shared_dataframe(dataframe_filepath, mem_area_name, dataset=None):
        """
        Creates a shared dataframe object. The given dataframe will be serialized and its output written into a shared memory area.
        Other Processes can obtain this area by referencing the shared memory area by its name. 
        """
        MB_1 = 1e6
        num_partitions = 10
        client = get_client()
        filesize = os.path.getsize(dataframe_filepath)     
        
        #Aim for 10 partitions
        blocksize = filesize / num_partitions if filesize >= num_partitions else MB_1 
        #print("Blocksize of " + dataframe_filepath + " is:" + str(blocksize)) 

        #Read DataFrame in parallel
        kwargs = {'sep': ',', 'header':'infer', 'encoding':'utf-8', 'dtype': str, 'keep_default_na': False, 'low_memory': False}
        dataframe = dask.dataframe.read_csv(urlpath=dataframe_filepath, blocksize=int(blocksize), **kwargs).applymap(DatasetParallel.value_normalizer)
        dataframe = client.compute(dataframe).result()
        dataframe.reset_index(inplace=True, drop=True)

        if dataset is not None:
            dataset.dataframe_num_rows = dataframe.shape[0]
            dataset.dataframe_num_cols = dataframe.shape[1]


        pickled_dataframe = pickle.dumps(dataframe, protocol=pickle.HIGHEST_PROTOCOL)
        pickled_dataframe_size = len(pickled_dataframe)
        #print("Size of pickled dataframe " + str(pickled_dataframe_size))

        shared_mem_area = sm.SharedMemory(name=mem_area_name, create=True, size=pickled_dataframe_size)
        shared_mem_area.buf[:pickled_dataframe_size] = pickled_dataframe

        shared_mem_area.close()
        del shared_mem_area
        return mem_area_name


    @staticmethod
    def create_shared_split_dataframe(dataframe_ref):
        """
        Creates several shared memory areas. Each area contains a single column of a given dataframe as pandas.Series objects.
        The given dataframes will be serialized and its output written into the corresponding shared memory area.
        Other Processes can obtain these areas by referencing the shared memory area by its name.
        """
        dataframe = DatasetParallel.load_shared_dataframe(dataframe_ref)

        for column in dataframe.columns.tolist():
            pickled_dataframe = pickle.dumps(dataframe[column], protocol=pickle.HIGHEST_PROTOCOL)
            pickled_dataframe_size = len(pickled_dataframe)
            try:
                shared_mem_area = sm.SharedMemory(name=column, create=True, size=pickled_dataframe_size)
                shared_mem_area.buf[:pickled_dataframe_size] = pickled_dataframe
                shared_mem_area.close()
            except FileExistsError:
                shared_mem_area = sm.SharedMemory(name=column, create=False, size=pickled_dataframe_size)
                shared_mem_area.buf[:pickled_dataframe_size] = pickled_dataframe
                shared_mem_area.close()
            except Exception as shm_err:
                print('Failed to create or attach to a split dataframe: {}'.format(shm_err))
                raise 

            del shared_mem_area
        return

    @staticmethod
    def load_shared_object(mem_area_name):
        shared_mem_area = sm.SharedMemory(name=mem_area_name, create=False)
        return pickle.loads(shared_mem_area.buf)

    @staticmethod
    def load_shared_num_rows(dataset_ref):
        """
        Loads number of rows from shared memory.
        """
        dataset = DatasetParallel.load_shared_dataset(dataset_ref)
        shared_mem_area = sm.SharedMemory(name=dataset.num_rows_ref, create=False)
        deserialized_num_rows = pickle.loads(shared_mem_area.buf)

        del shared_mem_area
        return deserialized_num_rows

    @staticmethod
    def load_shared_num_cols(dataset_ref):
        """
        Loads number of columns from shared memory.
        """
        dataset = DatasetParallel.load_shared_dataset(dataset_ref)
        shared_mem_area = sm.SharedMemory(name=dataset.num_cols_ref, create=False)
        deserialized_num_cols = pickle.loads(shared_mem_area.buf)

        del shared_mem_area
        return deserialized_num_cols

    @staticmethod
    def load_shared_dataset(dataset_ref):
        """
        Loads a shared memory dataset, which is stored and serialized in a shared_memory area(dataset_ref).
        The loaded dataset will be deserialized and returned.
        """
        shared_mem_area = sm.SharedMemory(name=dataset_ref, create=False)
        deserialized_dataset = pickle.loads(shared_mem_area.buf)

        shared_mem_area.close()
        del shared_mem_area
        return deserialized_dataset

    @staticmethod
    def load_shared_dataframe(dataframe_ref):
        """
        Loads a shared memory dataframe, which is stored and serialized in a shared_memory area(dataframe_ref).
        The loaded dataframe will be deserialized and returned.
        """
        shared_mem_area = sm.SharedMemory(name=dataframe_ref, create=False)
        deserialized_frame = pickle.loads(shared_mem_area.buf)

        shared_mem_area.close()
        del shared_mem_area
        return deserialized_frame

    @staticmethod
    def get_column_names(dataframe_filepath):
        """
        Returns a List of the column names of a given dataframe csv file.
        """
        return pandas.read_csv(dataframe_filepath, nrows=0).columns.tolist()

    @staticmethod
    def read_csv_dataframe(dataframe_path):
        """
        This method reads a dataset from a csv file path.
        """
        #Params to get passed to pandas read_csv function
        dataframe = pandas.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                    keep_default_na=False, low_memory=False).applymap(self.value_normalizer)
        return dataframe

    @staticmethod
    def write_csv(destination_path, dataframe_ref=None, dataframe=None, copy=False, source_path=None, pickle=False):
        """
        Writes Dataframe as csv file to given path.
        """
        if dataframe_ref is not None:
            DatasetParallel.load_shared_dataframe(dataframe_ref).to_csv(destination_path, sep=",", header=True, index=False, encoding="utf-8")
        elif dataframe is not None:
            dataframe.to_csv(destination_path, sep=",", header=True, index=False, encoding="utf-8")    
        else:
            if copy and source_path is not None:
                #print("Copying file from: " + source_path + " to: " + destination_path)
                try:
                    source_path = source_path
                    destination_path = destination_path
                    shutil.copyfile(source_path, destination_path)
                except:
                    raise ValueError("Copying csv to dest failed in write_csv()!")
            else:
                raise ValueError("Not enough values passed in write_csv()!")

    @staticmethod
    def get_dataframes_difference(dataframe_1, dataframe_2):
        """
        This method compares two dataframes and returns the different cells.
        """
        if dataframe_1.shape != dataframe_2.shape:
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        difference_dictionary = {}
        difference_dataframe = dataframe_1.where(dataframe_1.values != dataframe_2.values).notna()
        for j in range(dataframe_1.shape[1]):
            for i in difference_dataframe.index[difference_dataframe.iloc[:, j]].tolist():
                difference_dictionary[(i, j)] = dataframe_2.iloc[i, j]
        return difference_dictionary

    def get_actual_errors_dictionary(self):
        return DatasetParallel.load_shared_object(self.differences_dict_mem_ref)

    @staticmethod
    def value_normalizer(value):
        """
        This method takes a value and minimally normalizes it.
        """
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
        return value

    def hash_with_salt(self, word):
        salt = os.urandom(16)
        word_encoded = word.encode()
        salted_word_hash = hashlib.sha1(word_encoded + salt).hexdigest()
        return salted_word_hash


########################################
