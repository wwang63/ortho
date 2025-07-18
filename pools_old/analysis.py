import nupack as nu 
import numpy as np 
from tqdm import tqdm
from orthogonal import *
from barcodes import *
from toeholds import *
from finalize_seqs import *
    
def nupack_toe_matrix_mp(library, model, conc, ncores):
    size = len(library)
    np_probs = np.zeros((size, size))

    tasks = [(library, model, conc, i) for i in range(size)]
    with Pool(ncores) as pool:
        for i, result in tqdm(pool.imap(np_toe_row_worker, tasks), total=size):
            np_probs[i, i:] = result
            np_probs[i+1:, i] = result[1:]  # fill lower triangle by symmetry
    return np_probs

def ensemble_toe_matrix_mp(library, model, ncores):
    size = len(library)
    off_target_ensemble = np.zeros((size, size))
    on_target_ensemble = np.zeros(size)     

    tasks = [(library, model, i) for i in range(size)]
    with Pool(ncores) as pool:
        for i, off_p, on_p in tqdm(pool.imap(ensemble_toe_row_worker, tasks), total=size):
            off_target_ensemble[i, i:] = off_p
            off_target_ensemble[i+1:, i] = off_p[1:]  #fill lower triangle by symmetry
            on_target_ensemble[i] = on_p

    return off_target_ensemble, on_target_ensemble 

def weighted_edit_distance_worker(task):
    i,j,seq1,seq2 = task
    a=edit_distance_with_weighted_score(seq1, nu.reverse_complement(seq2), exponential_weight)
    b=edit_distance_with_weighted_score(seq2, nu.reverse_complement(seq1), exponential_weight)
    return i,j,np.min([a,b])

def weighted_edit_distance_mp(library, ncores):
    size = len(library)
    weighted_edit_distance = np.zeros((size,size))

    tasks = [(i,j,library[i], library[j]) for i in range(size) for j in range(i+1,size)]
    with Pool(ncores) as pool:
        for i,j, result in tqdm(pool.imap(weighted_edit_distance_worker, tasks), total=size):
            weighted_edit_distance[i,j] = result
            weighted_edit_distance[j,i] = result
    np.fill_diagonal(weighted_edit_distance, np.inf)
    return weighted_edit_distance
    
def evaluate_combinations(combinations, model, conc, ncores):
    # weighted edit distance 
    weighted_edit_distance = weighted_edit_distance_mp(combinations, ncores)
    
    # probabilities 
    np_probs = nupack_toe_matrix_mp(combinations, model, conc, ncores)
    
    # ensemble defects 
    off_target_ensemble, on_target_ensemble = ensemble_toe_matrix_mp(combinations, model, ncores)
    
    # combine results 
    return(np.min(weighted_edit_distance), average_no_diagonal(weighted_edit_distance), 
           np.max(np_probs), np.mean(np_probs), 
           np.max(on_target_ensemble), np.mean(on_target_ensemble), 
           np.min(off_target_ensemble), np.mean(off_target_ensemble))

def nupack_matrix_mp_barcodes(library, model, conc, num_barcodes, ncores):
    size = len(library)
    np_probs = np.zeros((size, size)) # low is good so 0 is best; ignore toehold self interactions
    on_probs = np.ones(size) # high is good so 1 is best; ignore toehold self interactions
    
    tasks = [(library, model, conc, i, num_barcodes) for i in range(num_barcodes)]
    with Pool(ncores) as pool:
        for i, row_slice, on_t in tqdm(pool.imap(np_bar_row_worker, tasks), total=num_barcodes):
            np_probs[i, i:] = row_slice
            np_probs[i+1:, i] = row_slice[1:]  # fill lower triangle by symmetry
            on_probs[i] = on_t
            
    return np_probs, on_probs

def evaluate_barcodes(barcodes, toeholds, model, conc, ncores):
    string_results = evaluate_barcodes_string([barcodes], toeholds, ncores)[0]
    library = barcodes + toeholds
    off_target_ensemble, on_target_ensemble = ensemble_toe_matrix_mp(barcodes, model, ncores)
    np_probs, on_probs = nupack_matrix_mp_barcodes(library, model, conc, len(barcodes), ncores)
    
    return(string_results + (
           np.min(on_probs), np.mean(on_probs),
           np.max(np_probs), np.mean(np_probs), 
           np.max(on_target_ensemble), np.mean(on_target_ensemble), 
           np.min(off_target_ensemble), np.mean(off_target_ensemble)))

def evaluate_pairings_worker(data):
    i, seq, unbound_structure, my_model = data 
    defect = nu.defect(strands=[seq],structure=unbound_structure, model=my_model)
    return i, defect
    
def evaluate_pairings(pairing_set, my_model, ncores):
    longest_common_substring = find_longest_common_substring(pairing_set)
    unbound_structure = ss_structure(len(pairing_set[0]))
    tasks = [(i, pairing_set[i], unbound_structure, my_model) for i in range(len(pairing_set))]
    results = np.empty(len(pairing_set), dtype = object)
    with Pool(ncores) as pool:
        for i, defect in tqdm(pool.imap(evaluate_pairings_worker, tasks), total=len(pairing_set)):
            results[i] = defect
    return(longest_common_substring, np.max(results), np.mean(results))
    