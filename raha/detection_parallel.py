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
import dataset_parallel as dp
import raha

########################################
class DetectionParallel:
    def __init__(self):
        self.LABELING_BUDGET = 20
        self.USER_LABELING_ACCURACY = 1.0
        self.BENCHMARK = True
        self.VERBOSE = False
        self.SAVE_RESULTS = False
        self.CLUSTERING_BASED_SAMPLING = True
        self.STRATEGY_FILTERING = False
        self.CLASSIFICATION_MODEL = "GBC"  # ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
        self.LABEL_PROPAGATION_METHOD = HOMOGENEITY   # ["homogeneity", "majority"]
        self.ERROR_DETECTION_ALGORITHMS = [OUTLIER_DETECTION, PATTERN_VIOLATION_DETECTION, 
                                           RULE_VIOLATION_DETECTION, KNOWLEDGE_BASE_VIOLATION_DETECTION]
        self.HISTORICAL_DATASETS = []
        self.TFID_ENABLED = False
    

    def run_outlier_strategy(self, configuration, dataset_ref, strategy_name_hash):
        """
        Detects cells which don't match given detection strategy - Outlier Detection.
        Returns dict, which contains coordinate of potentially defect cells.
        """
        outputted_cells = {}
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        dataframe_ref = dataset.dirty_mem_ref
        folder_path = os.path.join(tempfile.gettempdir(), dataset.name +"/")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        dataset_path = os.path.join(tempfile.gettempdir(),dataset.name + "/" + dataset.name + "-" + strategy_name_hash + ".csv")
        #Save Memory by copying prestripped file, read write_csv function
        dataset.write_csv(destination_path=dataset_path, dataframe_ref=dataframe_ref)

        parameters = ["-F", ",", "--statistical", "0.5"] + ["--" + configuration[0]] + configuration[1:] + [dataset_path]
        #print("Worker: " + str(get_worker().id) + " started dboost.run")
        raha.tools.dBoost.dboost.imported_dboost.run(parameters)

        dboost_result_path = dataset_path + "-dboost_output.csv"
        if os.path.exists(dboost_result_path) and os.path.getsize(dboost_result_path) > 0:
            dboost_dataframe = pandas.read_csv(dboost_result_path, sep=",", header=None, encoding="utf-8",
                                               dtype=str, keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())

            for i, j in dboost_dataframe.values.tolist():
                if int(i) > 0:
                    outputted_cells[(int(i)-1, int(j))] = ""
            os.remove(dboost_result_path)
               
        os.remove(dataset_path)
        return outputted_cells
        

    def run_pattern_strategy(self, configuration, dataset_ref):
        """
        Detects cells which don't match given detection strategy - Pattern Violation Detection.
        Returns dict, which contains coordinate of potentially defect cells.
        """
        outputted_cells = {}
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        column_name, character = configuration
        dataframe = dp.DatasetParallel.load_shared_dataframe(column_name)
        j = dp.DatasetParallel.get_column_names(dataset.dirty_path).index(column_name)
        #print("Worker: " + str(get_worker().id) + " running core run_pattern part")
        for i, value in dataframe.items():
            try:
                if len(re.findall("[" + character + "]", value, re.UNICODE)) > 0:
                    outputted_cells[(i, j)] = ""
            except:
                #print("Error occured in run_pattern_strategy in worker  " + str(get_worker().id))
                continue

        return outputted_cells

    def run_rule_strategyITERTUPLES(self, configuration, dataset_ref):
        """
        Detects cells which don't match given detection strategy - Rule Violation Detection.
        Returns dict, which contains coordinate of potentially defect cells.
        """
        value_dict = {}
        outputted_cells = {}
        left_attribute, right_attribute = configuration
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        dataframe = dp.DatasetParallel.load_shared_dataframe(dataset.dirty_mem_ref)

        left_attribute_j = dataframe.columns.get_loc(left_attribute)
        right_attribute_j = dataframe.columns.get_loc(right_attribute)
     
        for row_tuple in dataframe.itertuples():
            if row_tuple[left_attribute_j]:
                if row_tuple[left_attribute_j] not in value_dict:
                    value_dict[row_tuple[left_attribute_j]] = {}
                if row_tuple[right_attribute_j]:
                    value_dict[row_tuple[left_attribute_j]][row_tuple[right_attribute_j]] = 1
        for row_tuple in dataframe.itertuples():
            i = row_tuple.Index
            if row_tuple[left_attribute_j] in value_dict and len(value_dict[row_tuple[left_attribute_j]]) > 1:
                outputted_cells[(i, left_attribute_j)] = ""
                outputted_cells[(i, right_attribute_j)] = ""

        return outputted_cells

    #Runs like run_rule_strategyNPARR but with items() mixed, testing performance
    def run_rule_strategyITEMS(self, configuration, dataset_ref):
        value_dict = {}
        outputted_cells = {}
        left_attribute, right_attribute = configuration

        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        dataframe_left_column = dp.DatasetParallel.load_shared_dataframe(left_attribute)
        dataframe_right_column = dp.DatasetParallel.load_shared_dataframe(right_attribute)

        left_attribute_j = dp.DatasetParallel.get_column_names(dataset.dirty_path).index(left_attribute)
        right_attribute_j = dp.DatasetParallel.get_column_names(dataset.dirty_path).index(right_attribute)

        num_elements = len(dataframe_left_column)

        for i, value in dataframe_left_column.items():
            #print("{} Left attr:{}, {} Right attr:{}, index:{}".format(left_attribute,dataframe_left_column[i], right_attribute, dataframe_right_column[i], i))

            if value:
                if value not in value_dict:
                    value_dict[value] = {}
                if dataframe_right_column[i]:
                    value_dict[value][dataframe_right_column[i]] = 1
        for i in numpy.arange(0, num_elements):
            if dataframe_left_column[i] in value_dict and len(value_dict[dataframe_left_column[i]]) > 1:
                outputted_cells[(i, left_attribute_j)] = ""
                outputted_cells[(i, right_attribute_j)] = ""
        
        return outputted_cells

    def run_rule_strategyITERROWS(self,configuration, dataset_ref):
        value_dictionary = {}
        outputted_cells = {}
        l_attribute, r_attribute = configuration
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        d = dp.DatasetParallel.load_shared_dataframe(dataset.dirty_mem_ref)

        l_j = d.columns.get_loc(l_attribute)
        r_j = d.columns.get_loc(r_attribute)

        for i, row in d.iterrows():
            if row[l_attribute]:
                if row[l_attribute] not in value_dictionary:
                    value_dictionary[row[l_attribute]] = {}
                if row[r_attribute]:
                    value_dictionary[row[l_attribute]][row[r_attribute]] = 1
        for i, row in d.iterrows():
            if row[l_attribute] in value_dictionary and len(value_dictionary[row[l_attribute]]) > 1:
                outputted_cells[(i, l_j)] = ""
                outputted_cells[(i, r_j)] = ""
        return outputted_cells
    
    def run_rule_strategy(self, configuration, dataset_ref):
        """
        Detects cells which don't match given detection strategy - Rule Violation Detection.
        Returns dict, which contains coordinate of potentially defect cells.
        """
        value_dict = {}
        outputted_cells = {}
        left_attribute, right_attribute = configuration

        #Read Columns as seperate Series Objects - a more memory efficient approach
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        dataframe_left_column = dp.DatasetParallel.load_shared_dataframe(left_attribute)
        dataframe_right_column = dp.DatasetParallel.load_shared_dataframe(right_attribute)

        left_attribute_j = dp.DatasetParallel.get_column_names(dataset.dirty_path).index(left_attribute)
        right_attribute_j = dp.DatasetParallel.get_column_names(dataset.dirty_path).index(right_attribute)

        num_elements = len(dataframe_left_column)
        
        #Process through both columns and use the index to synchronize correct positional access
        for i in numpy.arange(0, num_elements):   
            left_value = dataframe_left_column[i]
            right_value = dataframe_right_column[i]
            if left_value:
                if left_value not in value_dict:
                    value_dict[left_value] = {}
                if right_value:
                    value_dict[left_value][right_value] = 1

        #Update the defect cells dictionary of a cell, if left value references more than 1 right value
        for i in numpy.arange(0, num_elements):
            left_value = dataframe_left_column[i]
            if left_value in value_dict and len(value_dict[left_value]) > 1:
                outputted_cells[(i, left_attribute_j)] = ""
                outputted_cells[(i, right_attribute_j)] = ""
        
        return outputted_cells

    def run_knowledge_strategy(self, configuration, dataset_ref):
        """
        Detects cells which don't match given detection strategy - Knowledge Base Violation Detection.
        Returns dict, which contains coordinate of potentially defect cells.
        """
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        outputted_cells = raha.tools.KATARA.katara.run(dataset, configuration)
        return outputted_cells

    def parallel_strat_runner_process(self, args):
        """
        Runs all error detection strategies in a seperate worker process.
        """
        start_time = time.time()
        outputted_cells = {}
        dataset_ref, algorithm, configuration = args
        strategy_name = json.dumps([algorithm, configuration])
        strategy_name_hashed = str( int( hashlib.sha1( strategy_name.encode("utf-8")).hexdigest(), base=16))

        match algorithm:
            case constants.OUTLIER_DETECTION:
                #Run outlier detection strategy
                outputted_cells = self.run_outlier_strategy(configuration, dataset_ref, strategy_name_hashed)
            case constants.PATTERN_VIOLATION_DETECTION:
                #Run pattern violation detection strategy
                outputted_cells = self.run_pattern_strategy(configuration, dataset_ref)
            case constants.RULE_VIOLATION_DETECTION:
                #Run rule violation detection strategy
                outputted_cells = self.run_rule_strategy(configuration, dataset_ref)
            case constants.KNOWLEDGE_BASE_VIOLATION_DETECTION:
                #Run knowledge base violation strategy
                outputted_cells = self.run_knowledge_strategy(configuration, dataset_ref)
            case _:
                raise ValueError("Algorithm " + str(algorithm) + " is not supported!")
        
        end_time = time.time()
        strategy_results = {
            "name": strategy_name,
            "output": list(outputted_cells.keys()),
            "runtime": end_time - start_time
        }

        if self.SAVE_RESULTS:
            dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
            pickle.dump(strategy_results, open(os.path.join(dataset.results_folder, "strategy-profiling", strategy_name_hashed + ".dictionary"), "wb"))
        if self.VERBOSE:
            print("{} cells are detected by {}".format(len(outputted_cells), strategy_name))

        return strategy_results
    
    @staticmethod
    def setup_outlier_metadata(dataset_ref):
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

        configurations.extend([dataset_ref, OUTLIER_DETECTION, conf] for conf in cartesian_config)
        return configurations

    @staticmethod
    def pattern_violation_worker(dataset_ref, column_name):
        """
        Worker-Process in a parallel manner. Extracts all characters of one specific column and returns them.
        """
        configurations = []
        client = get_client()

        #Load Shared DataFrame column by accessing shared memory area, named by column_name
        dataset_column = dp.DatasetParallel.load_shared_dataframe(column_name)

        #Concatenate all content of a column into a long string
        column_data = "".join(dataset_column.tolist())
        #Notice which character appeared in this column
        character_dict = {character: 1 for character in column_data} 
        character_dict_list = [[column_name, character] for character in character_dict]  

        configurations.extend([dataset_ref, PATTERN_VIOLATION_DETECTION, conf] for conf in character_dict_list)
        del dataset_column, character_dict, column_data
        return configurations

    @staticmethod
    def setup_pattern_violation_metadata(dataset_ref):
        """
        Calculates Meta-Data for pattern-violation application later on
        """
        futures = []
        configurations = []
        client = get_client()
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        column_names = dp.DatasetParallel.get_column_names(dataset.dirty_path)
        
        #Call a worker for each column name
        arguments1 = [dataset_ref] * len(column_names)
        arguments2 = [column_name  for column_name in column_names]
        futures.append(client.map(DetectionParallel.pattern_violation_worker, arguments1, arguments2))

        #Gather results and merge "list of lists" into one big list
        results_unmerged = client.gather(futures=futures, direct=True)[0]
        results = list(itertools.chain.from_iterable(results_unmerged))

        return results
    @staticmethod
    def setup_rule_violation_metadata(dataset_ref):
        """
        Calculates Meta-Data for rule-violation application later on.
        This method just creates pairs of column names as potential FDs
        Generates exactly n*(n-1) FD pairs
        """
        configurations = []
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        column_names = dp.DatasetParallel.get_column_names(dataset.dirty_path)
        column_pairs = [[col1, col2] for (col1,col2) in itertools.product(column_names, column_names) if col1 != col2]
        
        configurations.extend([[dataset_ref, RULE_VIOLATION_DETECTION, column_pair] for column_pair in column_pairs])

        return configurations
    @staticmethod
    def setup_knowledge_violation_metadata(dataset_ref):
        configurations = []
        paths = [os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base", path) for path in os.listdir(os.path.join(os.path.dirname(__file__), "tools", "KATARA", "knowledge-base"))]
        configurations.extend([[dataset_ref, KNOWLEDGE_BASE_VIOLATION_DETECTION, path] for path in paths])

        return configurations

    def run_strategies(self, dataset):
        """
        Creates strategies metadata and executes each strategy for a seperate worker process
        """
        #TODO - Implement preloading results
        starttime = time.time()
        strategy_profile_path = os.path.join(dataset.results_folder, "strategy-profiling")
        client = get_client()
        futures = []

        if self.STRATEGY_FILTERING:
            for data_dictionary in self.HISTORICAL_DATASETS + [dataset.dictionary]:
                raha.utilities.dataset_profiler(data_dictionary)
                raha.utilities.evaluation_profiler(data_dictionary)
            return raha.utilities.get_selected_strategies_via_historical_data(dataset.dictionary, self.HISTORICAL_DATASETS)

        #TODO implement preloading here

        for algorithm_name in self.ERROR_DETECTION_ALGORITHMS:
            match algorithm_name:
                case constants.OUTLIER_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_outlier_metadata, dataset.own_mem_ref))
                case constants.PATTERN_VIOLATION_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_pattern_violation_metadata, dataset.own_mem_ref))
                case constants.RULE_VIOLATION_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_rule_violation_metadata, dataset.own_mem_ref))
                case constants.KNOWLEDGE_BASE_VIOLATION_DETECTION:
                    futures.append(client.submit(DetectionParallel.setup_knowledge_violation_metadata, dataset.own_mem_ref))
                case _:
                    raise ValueError("Algorithm " + str(algorithm) + " is not supported!")
        
        #Gather Results of all workers, metadata configuration
        results = list(itertools.chain.from_iterable(client.gather(futures=futures, direct=True)))
        endtime = time.time()
        print("Raha strategy metadata generation(parallel): "+  str(endtime - starttime))

        #Start Detecting Errors in parallel
        futures = client.map(self.parallel_strat_runner_process, results)
        #Gather Results of all workers, detected cells as dicts
        strategy_profiles = client.gather(futures=futures, direct=True)
        endtime = time.time()
        print("Raha running all strategies total time(parallel): "+  str(endtime - starttime))
        #print(strategy_profiles)
        return strategy_profiles


    def generate_features_one_col(self, dataset_ref, column_index):
        """
        Worker-Process. Calculates a feature-matrix for one column. 
        A row represents 1 feature-vector of a cell, the row_index is the x-coordinate of the cell the column_index the y-coordinate.
        A column represents the results of 1 specific strategy on *all* cells.
        Does not return the feature-matrix but rather a reference to it in a shared memory area. 
        """
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        strategy_profiles_area = sm.SharedMemory(name=dataset.dirty_mem_ref + "-strategy_profiles", create=False)
        strategy_profiles = pickle.loads(strategy_profiles_area.buf)
        feature_vectors = numpy.zeros((dataset.dataframe_num_rows, len(strategy_profiles)))

        for strategy_index, strategy_profile in enumerate(strategy_profiles):
            strategy_name = json.loads(strategy_profile["name"])[0]

            if strategy_name in self.ERROR_DETECTION_ALGORITHMS:
                for cell in strategy_profile["output"]:
                    if cell[1] == column_index:
                        feature_vectors[cell[0], strategy_index] = 1.0
        if self.TFID_ENABLED:
            vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
            column_name = dp.DatasetParallel.get_column_names(dataset.dirty_path)[column_index]
            corpus = dp.DatasetParallel.load_shared_dataframe(column_name)
            try:
                tfid_features = vectorizer.fit_transform(corpus)
                feature_vectors = numpy.column_stack((feature_vectors, numpy.array(tfidf_features.todense())))
            except:
                pass

        promising_strategies = numpy.any(feature_vectors != feature_vectors[0, :], axis=0)
        feature_vectors = feature_vectors[:, promising_strategies] 

        #Store feature vectors in a shared memory area
        dp.DatasetParallel.create_shared_object(feature_vectors, dataset.dirty_mem_ref + "-feature-result-" + str(column_index))

        strategy_profiles_area.close()
        return dataset.dirty_mem_ref + "-feature-result-" + str(column_index)

    def generate_features(self, dataset, strategy_profiles):
        """
        Generates feature vector for each column. A seperate matrix is being built for each column.
        Strategies, which mark all cells as either detected or undetected are being discarded.
        """
        start_time = time.time()
        client = get_client()
        dp.DatasetParallel.create_shared_object(strategy_profiles, dataset.dirty_mem_ref + "-strategy_profiles")
        futures = []

        #Start workers and append their future-references to the futures list. From each passed parameter-list one entry is passed to the worker.
        futures.append(client.map(self.generate_features_one_col, [dataset.own_mem_ref] * dataset.dataframe_num_cols, numpy.arange(dataset.dataframe_num_cols)))

        results = client.gather(futures=futures, direct=True)
        end_time = time.time()
 
        print("Generate Features(parallel): " + str(end_time - start_time))

        return results

    def build_clusters_single_column(self, dataset_ref, column_index):
        """
        Worker-Process. Calculates all clusters and the respective cells of *one* column.
        """
        dataset = dp.DatasetParallel.load_shared_dataset(dataset_ref)
        column_features = dp.DatasetParallel.load_shared_object(dataset.dirty_mem_ref + "-feature-result-" + str(column_index))

        clusters_k_c_ce = {k : {} for k in range(2, self.LABELING_BUDGET + 2)}
        cells_clusters_k_ce = {k : {} for k in range(2, self.LABELING_BUDGET + 2)}

        try:
            clustering_model = scipy.cluster.hierarchy.linkage(column_features, method="average", metric="cosine")
            #The bigger our labeling budget is, the more clusters will be generated per column
            for k in clusters_k_c_ce:
                model_labels = [l-1 for l in scipy.cluster.hierarchy.fcluster(clustering_model, k, criterion="maxclust")]
                # DEBUG print("\nWorker for column " + str(column_index) + " k:" + str(k) + " is " + str(model_labels))
                #Model label contains a 1D-Array, where each index represents the row of a cell and column_index represents the column of the cell
                #c is the number of the cluster this cell belongs to
                for index, c in enumerate(model_labels):
                        if c not in clusters_k_c_ce[k]:
                            #Create a dict containing all cells which belong to cluster number c
                            #Depends on labeling budget k, which represents the total number of clusters available per column
                            clusters_k_c_ce[k][c] = {}
                        
                                #index = row, column_index = column -> coordinates of a specific cell
                        cell = (index, column_index)
                        clusters_k_c_ce[k][c][cell] = 1
                        cells_clusters_k_ce[k][cell] = c
        except:
            pass
        
        if self.VERBOSE:
            print("A hierarchical clustering model is built for column {}".format(column_index))
        clusters_k_j_c_ce = {k : clusters_k_c_ce[k] for k in range(2, self.LABELING_BUDGET+2)}
        cells_clusters_k_j_ce =  {k : cells_clusters_k_ce[k] for k in range(2, self.LABELING_BUDGET+2)}

        #print(clusters_k_c_ce if column_index == 0 else "")
        #TODO Think about if you want to return these lists or rather save them in shared mem again.
        #print("\nI'm worker: {}, my task is column {}\nMy result is: {}".format(get_worker().id, column_index, [column_index, clusters_k_c_ce,"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" , cells_clusters_k_ce]))
        return [column_index, clusters_k_j_c_ce, cells_clusters_k_j_ce]

    def build_clusters(self, dataset, features_refs):
        """
        Calculates clusters for all columns.
        """
        start_time = time.time()
        clustering_results = []
        client = get_client()
        futures = []

        futures.append(client.map(self.build_clusters_single_column, [dataset.own_mem_ref]*dataset.dataframe_num_cols, numpy.arange(dataset.dataframe_num_cols)))
        results = client.gather(futures=futures, direct=True)[0]
        results.sort(key= lambda x: x[0], reverse=False)

        end_time = time.time()
        print("Build clusters (parallel): " + str(end_time-start_time))

        return results
    
    def sample_tuple(self, dataset, clustering_results):
        """
        Calculates a sample-tuple which will later be labeled by the user or by the ground-truth
        """
        k = len(dataset.labeled_tuples)+2
        for j in numpy.arange(dataset.dataframe_num_cols):
            for c in clustering_results[j][1][k]:
                dataset.labels_per_cluster[(j, c)] = {cell : dataset.labeled_cells[cell][0] for cell in clustering_results[j][1][k][c]
                                                             if cell[0] in dataset.labeled_tuples}

        if self.CLUSTERING_BASED_SAMPLING:
            tuple_score = numpy.zeros(dataset.dataframe_num_rows)
            for i in numpy.arange(dataset.dataframe_num_rows):
                if i not in dataset.labeled_tuples:
                    score = 0.0
                    for j in numpy.arange(dataset.dataframe_num_cols):
                        if clustering_results[j][1][k]:
                            cell = (i, j)
                            c = clustering_results[j][2][k][cell]
                            score += math.exp(-len(dataset.labels_per_cluster[(j, c)]))
                    tuple_score[i] = math.exp(score)
        else:
            tuple_score = numpy.ones(dataset.dataframe_num_rows)
        sum_tuple_score = sum(tuple_score)
        p_tuple_score = tuple_score / sum_tuple_score
        dataset.sampled_tuple = numpy.random.choice(numpy.arange(dataset.dataframe_num_rows), 1, p=p_tuple_score)[0]
        end_time = time.time()
        if self.VERBOSE:
            print("Tuple {} is sampled".format(dataset.sampled_tuple))

        return dataset.sampled_tuple  
    
    def label_with_ground_truth(self, dataset):
        k = len(dataset.labeled_tuples) + 2
        dataset.labeled_tuples[dataset.sampled_tuple] = 1
        actual_errors_dictionary = dataset.get_actual_errors_dictionary()
        clean_dataframe = dp.DatasetParallel.load_shared_dataframe(dataset.clean_mem_ref)

        for j in numpy.arange(dataset.dataframe_num_cols):
            cell = (dataset.sampled_tuple, j)
            user_label = int(cell in actual_errors_dictionary)
            flip_result_chance = random.random()

            if flip_result_chance > self.USER_LABELING_ACCURACY:
                user_label = 1 - user_label
            dataset.labeled_cells[cell] = [user_label, clean_dataframe.iloc[cell]]
        if self.VERBOSE:
            print("Tuple {} is labeled.".format(dataset.sampled_tuple))
        return    

#2: {0: {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (4, 0): 0, (5, 0): 0}, 1: {(0, 1): 0, (1, 1): 0, (2, 1): 0, (3, 1): 0, (4, 1): 0, (5, 1): 1}, 2: {(0, 2): 1, (1, 2): 1, (2, 2): 0, (3, 2): 0, (4, 2): 0, (5, 2): 1}}



########################################
