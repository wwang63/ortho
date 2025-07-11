import numpy as np
import nupack as nu
import multiprocessing as mp
from orthogonal import *
from toeholds import *
from tqdm import tqdm
from functools import partial
from scipy.stats import rankdata
from seqwalk import design
from math import comb
import itertools
import random

def get_substrings_set(seq, k):
    string_set = []
    for i in range(len(seq)-k+1):
        string_set.append(seq[i:i+k])
    return string_set

def get_substrings_set_with_reverse_complement(seq, k):
    string_set = []
    for i in range(len(seq)-k+1):
        string_set.append(seq[i:i+k])
        string_set.append(nu.reverse_complement(seq[i:i+k]))
    return string_set

def toeholds_get_substrings_set_with_rc(toeholds, k):
    string_set = []
    for i in range(len(toeholds)):
        string_set = string_set + get_substrings_set_with_reverse_complement(toeholds[i], k)
    return string_set

def generate_putative_barcodes(toeholds, q, k, len_barcode, num_sets):
    num_barcodes = len(toeholds)
    string_set = toeholds_get_substrings_set_with_rc(toeholds, q)
    barcodes = design.max_size(len_barcode, k, alphabet="ACGT", RCfree=1,prevented_patterns=string_set)
    print(f"generated {len(barcodes)} barcodes to sample from")
    print(f"each sampled set contains {num_barcodes} barcodes")
    if len(barcodes) < num_barcodes:
        raise ValueError(f"q,k choice too stringent, can only generate {len(barcodes)} possible barcodes")
    
    barcode_sets = []
    
    max_possible_barcodes = comb(len(barcodes),num_barcodes)
    if max_possible_barcodes < num_sets:
        print(f"can only generate {max_possible_barcodes} barcodes")
        for combination in itertools.combinations(barcodes, num_barcodes):
            barcode_sets.append(combination)
    else:
        while len(barcode_sets) < num_sets:
            # Generate a random subset
            random_subset = random.sample(barcodes, num_barcodes)
            
            # Check if we've already generated this subset
            if random_subset not in barcode_sets:
                barcode_sets.append(random_subset)
                
    return barcode_sets

def bar_and_toe_evaluation_thermo(data, model, conc):
    library, id, num_bar = data
    
    # generate evaluation metric data
    nu_mat, on_t = nupack_matrix_no_mp_barcodes(library, model, conc, num_bar)
    # only calcuate ensemble data for the barcode domains
    off_target_ensemble, on_target_ensemble = ensemble_toe_matrix_no_mp(library[:num_bar], model)
    
    min_ontarget_probability = np.min(on_t)
    average_ontarget_probability = np.mean(on_t)
    max_offtarget_probability = np.max(nu_mat)
    average_offtarget_probability = np.mean(nu_mat)
    max_on_target_defect = np.max(on_target_ensemble)
    average_on_target_defect = np.mean(on_target_ensemble)
    min_off_target_defect = np.min(off_target_ensemble)
    average_off_target_defect = np.mean(off_target_ensemble)
    
    return id, (min_ontarget_probability, average_ontarget_probability, 
                max_offtarget_probability, average_offtarget_probability, 
                max_on_target_defect, average_on_target_defect, 
                min_off_target_defect, average_off_target_defect)
    
def evaluate_barcodes_thermo(barcode_sets, toeholds, model, ncores, conc):
    evaluation_func = partial(bar_and_toe_evaluation_thermo, 
                            model=model, 
                            conc=conc)
    bars_and_toes = barcode_sets.copy()
    for i in range(len(bars_and_toes)):
        bars_and_toes[i] = bars_and_toes[i] + toeholds
    assert(len(bars_and_toes[0]) != len(barcode_sets[0]))
    results = np.empty(len(barcode_sets), dtype=object)
    tasks = [(bars_and_toes[i], i, len(barcode_sets[i])) for i in range(len(bars_and_toes))]
    assert(len(bars_and_toes) == len(barcode_sets))
    with Pool(ncores) as pool:
        for i, result in tqdm(pool.imap(evaluation_func, tasks), total=len(bars_and_toes)):
            results[i] = result
    return results
    
def bar_and_toe_evaluation_string(data):
    library, id, num_bar = data
    
    # generate evaluation metric data
    # only calcuate alignment data for the barcode domains
    a_matrix = alignment_matrix(library[:num_bar], duplex=1)
    np.fill_diagonal(a_matrix, np.inf)

    min_edit_distance = np.min(a_matrix)
    average_edit_distance = average_no_diagonal(a_matrix)
    return id, (min_edit_distance, average_edit_distance)

def evaluate_barcodes_string(barcode_sets, toeholds, ncores):
    evaluation_func = partial(bar_and_toe_evaluation_string)
    bars_and_toes = barcode_sets.copy()
    for i in range(len(bars_and_toes)):
        bars_and_toes[i] = bars_and_toes[i] + toeholds
    assert(len(bars_and_toes[0]) != len(barcode_sets[0]))
    results = np.empty(len(barcode_sets), dtype=object)
    tasks = [(bars_and_toes[i], i, len(barcode_sets[i])) for i in range(len(bars_and_toes))]
    assert(len(bars_and_toes) == len(barcode_sets))
    with Pool(ncores) as pool:
        for i, result in tqdm(pool.imap(evaluation_func, tasks), total=len(bars_and_toes)):
            results[i] = result
    return results

def np_bar_row_worker(args):
    library, model, conc, i, num_barcodes = args
    size = len(library)
    row = np.zeros(size)
    on_p = np_ontarget(library[i], model, conc, duplex=1) / conc
    
    for j in range(i, size):
        if i != j and j < num_barcodes and i < num_barcodes:
            res = np_complements_crosstalk(library[i], library[j], model, conc)
            p = np.max(res) / conc
        elif i != j: 
            res = np_crosstalk(library[i], library[j], model, conc, duplex=1)
            p = np.max(res) / conc
        else:
            # diagonal does not matter because we only care about off targets between 
            # sense and antisense toehold pairs
            p = 0
        row[j] = p  # only upper triangle [i, j]
    return i, row[i:], on_p  # return upper triangle 

def nupack_matrix_no_mp_barcodes(library, model, conc, num_barcodes):
    size = len(library)
    np_probs = np.zeros((size, size)) # low is good so 0 is best; ignore toehold self interactions
    on_probs = np.ones(size) # high is good so 1 is best; ignore toehold self interactions

    for i in range(num_barcodes):
        i, row_slice, on_t = np_bar_row_worker((library, model, conc, i, num_barcodes))
        np_probs[i, i:] = row_slice
        np_probs[i+1:, i] = row_slice[1:]  # fill lower triangle by symmetry
        on_probs[i] = on_t
    return np_probs, on_probs


"""
General purpose function to rank evaluation results. 
Weights are a dictionary of metric names to a tuple of their weights and whether 
higher is better (1) or lower is better (-1).     
"""
def rank_combinations_weighted(results, weights, return_top = 1):
    if len(results) == 0:
        return None, {}
    
    # Extract metrics from results
    valid_results = [(i, metrics) for i, metrics in enumerate(results)]
    indices, metrics_list = zip(*valid_results)
    indices = list(indices)
    metrics_array = np.array(metrics_list)
    
    # Get metric names in order
    metric_names = list(weights.keys())
    num_metrics = len(metric_names)
    
    # Verify we have the right number of metrics
    if len(metrics_array[0]) != num_metrics:
        raise ValueError(f"Expected {num_metrics} metrics, got {metrics_array.shape[1]}")
    
    # Calculate ranks for each metric
    metric_ranks = {}
    for i, metric_name in enumerate(metric_names):
        weight, direction = weights[metric_name]
        metric_values = metrics_array[:, i]
        
        # Apply direction: negate if higher is better (direction=1)
        # this is because rankdata gives rank 1 to smallest value
        if direction == 1:
            metric_ranks[metric_name] = rankdata(-metric_values, method='min')
        else:  # direction == -1, lower is better
            metric_ranks[metric_name] = rankdata(metric_values, method='min')
    
    # Calculate weighted average rank
    weighted_ranks = np.zeros(len(indices))
    total_weight = 0
    
    for metric_name in metric_names:
        weight, direction = weights[metric_name]
        weighted_ranks += metric_ranks[metric_name] * weight
        total_weight += weight
    
    weighted_ranks /= total_weight
    
    # Find best combination (lowest weighted rank)
    sorted_comb_ids = np.argsort(weighted_ranks)
    best_combination_id = np.argmin(weighted_ranks)
    
    # Print results
    print(f"Best combination ID: {best_combination_id}")
    print("\nIndividual ranks:")
    for metric_name in metric_names:
        rank = metric_ranks[metric_name][best_combination_id]
        print(f"{metric_name}: {rank}")
    
    print(f"\nRaw data for best combination:")
    print(results[best_combination_id])
    if return_top == 1:
        return best_combination_id
    else:
        return sorted_comb_ids[:return_top]
    

