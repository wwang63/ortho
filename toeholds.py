import numpy as np
import csv
import nupack as nu
import multiprocessing as mp
from orthogonal import *
import itertools
import random
from tqdm import tqdm
from functools import partial
from scipy.stats import rankdata

def average_no_diagonal(matrix):
    matrix = np.array(matrix)
    
    # Create mask for non-diagonal elements
    mask = ~np.eye(matrix.shape[0], matrix.shape[1], dtype=bool)
    
    # Get non-diagonal elements and calculate average
    non_diagonal_elements = matrix[mask]
    return np.mean(non_diagonal_elements)

def edit_distance_with_weighted_score(seq1, seq2, weight_func):    
    # Step 1: Build standard Levenshtein distance matrix
    E = np.zeros((len(seq1) + 1, len(seq2) + 1))
    
    # Initialize first row and column
    for i in range(len(seq1) + 1):
        E[i, 0] = i
    for j in range(len(seq2) + 1):
        E[0, j] = j
    
    # Fill the matrix
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1
            
            E[i, j] = min(
                E[i-1, j] + 1,      # deletion
                E[i, j-1] + 1,      # insertion
                E[i-1, j-1] + cost  # substitution/match
            )
    
    standard_distance = E[len(seq1), len(seq2)]
    
    # Step 2: Backtrack to find the optimal alignment
    alignment = []
    i, j = len(seq1), len(seq2)
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            # Check if we came from diagonal (substitution/match)
            if seq1[i-1] == seq2[j-1]:
                diagonal_cost = E[i-1, j-1]
            else:
                diagonal_cost = E[i-1, j-1] + 1
            
            if E[i, j] == diagonal_cost:
                # Match or substitution
                if seq1[i-1] == seq2[j-1]:
                    alignment.append(('match', i-1, j-1, seq1[i-1]))
                else:
                    alignment.append(('substitute', i-1, j-1, seq1[i-1], seq2[j-1]))
                i -= 1
                j -= 1
                continue
        
        if i > 0 and E[i, j] == E[i-1, j] + 1:
            # Deletion from seq1
            alignment.append(('delete', i-1, j, seq1[i-1]))
            i -= 1
        elif j > 0 and E[i, j] == E[i, j-1] + 1:
            # Insertion to seq1
            alignment.append(('insert', i, j-1, seq2[j-1]))
            j -= 1
        else:
            # This shouldn't happen with correct implementation
            break

    alignment.reverse()  # We built it backwards
    
    # Step 3: Calculate weighted score for this alignment
    weighted_score = 0.0
    
    for operation in alignment:
        if operation[0] == 'match':
            # No cost for matches
            continue
        elif operation[0] == 'substitute':
            pos_in_seq1 = operation[1]
            weight = weight_func(pos_in_seq1, len(seq1))
            weighted_score += weight
        elif operation[0] == 'delete':
            pos_in_seq1 = operation[1]
            weight = weight_func(pos_in_seq1, len(seq1))
            weighted_score += weight
        elif operation[0] == 'insert':
            pos_in_seq1 = operation[1]  # This is where it would be inserted
            weight = weight_func(pos_in_seq1, len(seq1))
            weighted_score += weight
    
    return weighted_score

# weighted edit distance
def exponential_weight(pos, seq_len):
    if seq_len == 1:
        return 1.0
    normalized_pos = pos / (seq_len - 1)  # 0 to 1
    return (np.exp(1) ** -normalized_pos) + 1

def alignment_toe_sense_antisense_matrix(library):
    similar = np.zeros((len(library),len(library)))
    #for i in tqdm(range(len(similar))):
    for i in range(len(similar)):
        for j in range(i,len(similar)):
            vals = []
            vals.append(edit_distance_with_weighted_score(library[i], library[j], exponential_weight))
            vals.append(edit_distance_with_weighted_score(library[j], library[i], exponential_weight))
            val = np.min(vals)
            similar[i][j] = val
            similar[j][i] = val
    
    return similar

def np_complements_crosstalk(seq1, seq2, model, conc):
    A = nu.Strand(seq1, name='A')
    B = nu.Strand(seq2, name='B')
    # off target testing
    o2 = nu.Complex([A, ~B], name="o2")
    o2_tube = nu.Tube(strands={A: conc, ~B: conc},
                name='o2_tube', complexes=nu.SetSpec(max_size=2, include = [o2]))
    o3 = nu.Complex([~A, B], name="o3")
    o3_tube = nu.Tube(strands={~A: conc, B: conc},
                name='o3_tube', complexes=nu.SetSpec(max_size=2, include = [o3]))
    
    # compute tubes
    tube_results = nu.tube_analysis(tubes=[o2_tube,o3_tube], 
                                    model=model)
    
    # get concentrations 
    o2_conc = tube_results.tubes[o2_tube].complex_concentrations[o2]
    o3_conc = tube_results.tubes[o3_tube].complex_concentrations[o3]

    return [o2_conc, o3_conc]

def np_toe_row_worker(args):
    library, model, conc, i = args
    size = len(library)
    row = np.zeros(size)
    
    for j in range(i, size):
        if i != j:
            res = np_complements_crosstalk(library[i], library[j], model, conc)
            p = np.max(res) / conc
        else:
            # diagonal does not matter because we only care about off targets between 
            # sense and antisense toehold pairs
            p = 0
        row[j] = p  # only upper triangle [i, j]
    return i, row[i:]  # return upper triangle 

def nupack_toe_matrix_no_mp(library, model, conc):
    size = len(library)
    np_probs = np.zeros((size, size))

    for i in range(size):
        i, row_slice = np_toe_row_worker((library, model, conc, i))
        np_probs[i, i:] = row_slice
        np_probs[i+1:, i] = row_slice[1:]  # fill lower triangle by symmetry
    return np_probs

def np_complements_cross_ensemble(seq1, seq2, model):
    o2 = ensemble_defect_pair(seq1, nu.reverse_complement(seq2), model, 
                                duplex_structure(len(seq1)))
    o3 = ensemble_defect_pair(nu.reverse_complement(seq1), seq2, model, 
                                duplex_structure(len(seq1)))
    return [o2, o3]

def ensemble_toe_row_worker(args):
    library, model, i = args
    size = len(library)
    row = np.zeros(size)
    on_p = np.max(np_ontarget_ensemble(library[i], model, duplex=1))
    
    for j in range(i, size):
        if i != j:
            res = np_complements_cross_ensemble(library[i], library[j], model)
            p = np.min(res)
        else:
            # diagonal does not matter 
            p = 1
        
        row[j] = p  # only upper triangle [i, j]

    return i, row[i:], on_p  # return upper triangle 

def ensemble_toe_matrix_no_mp(library, model):
    size = len(library)
    off_target_ensemble = np.zeros((size, size))
    on_target_ensemble = np.zeros(size)

    for i in range(size):
        i, row_slice, on_t = ensemble_toe_row_worker((library, model, i))
        off_target_ensemble[i, i:] = row_slice
        off_target_ensemble[i+1:, i] = row_slice[1:]  #fill lower triangle by symmetry
        on_target_ensemble[i] = on_t

    return off_target_ensemble, on_target_ensemble

def generate_putative_toeholds(seq, args):
    oligo_length, barcode_length, fragment_length, max_final_c_domain_length, toehold_length, toehold_search_range, identity_threshold = args
    putative_toeholds = []
    position = fragment_length - toehold_length 
    while position < len(seq):
        toehold_subcandidates = []
        # for each possible toehold position, add the toehold to the list
        for i in range(toehold_search_range):
            toehold_subcandidates.append((seq[position:position+toehold_length].upper(),position))
            position += 1 
        putative_toeholds.append(toehold_subcandidates)
        # if the remaining sequence is less than the max final c domain length, break
        if len(seq) - position <= max_final_c_domain_length:
            break
        position = position - toehold_search_range + fragment_length - toehold_length
    return putative_toeholds

def toehold_combinations(putative_toeholds, args, how_many):
    oligo_length, barcode_length, fragment_length, max_final_c_domain_length, toehold_length, toehold_search_range, identity_threshold = args
    
    # if how_many = 0, we evaluate all toehold combinations 
    # else we evaluate how_many random toehold combinations
    if how_many == 0:        
        toehold_combinations = list(itertools.product(*putative_toeholds))
    else:
        toehold_combinations = np.empty(how_many, dtype=object)
        
        for i in range(how_many):
            # Pick one random tuple from each sublist
            random_combination = tuple(random.choice(sublist) for sublist in putative_toeholds)
            toehold_combinations[i] = random_combination
        
    return toehold_combinations

def evaluation_combination_string(data):
    library, id = data
    duplex = 1 # since toeholds inherently form duplexes
    
    # generate evaluation metric data
    a_matrix = alignment_toe_sense_antisense_matrix(library)
    np.fill_diagonal(a_matrix, np.inf)
    min_edit_distance = np.min(a_matrix)
    average_edit_distance = average_no_diagonal(a_matrix)

    return id, (min_edit_distance, average_edit_distance)

def evaluate_combinations_string(combinations, ncores):
    evaluation_func = partial(evaluation_combination_string)
    results = np.empty(len(combinations), dtype=object)
    tasks = [(list(zip(*combinations[i]))[0], i) for i in range(len(combinations))]
    with Pool(ncores) as pool:
        for i, result in tqdm(pool.imap(evaluation_func, tasks), total=len(combinations)):
            results[i] = result
    return results

def evaluation_combination_thermo(data, model, conc):
    library, id = data
    duplex = 1 # since toeholds inherently form duplexes
    
    # generate evaluation metric data
    nu_mat = nupack_toe_matrix_no_mp(library, model, conc)
    off_target_ensemble, on_target_ensemble = ensemble_toe_matrix_no_mp(library, model)
    
    max_offtarget_probability = np.max(nu_mat)
    average_offtarget_probability = np.mean(nu_mat)
    max_on_target_defect = np.max(on_target_ensemble)
    average_on_target_defect = np.mean(on_target_ensemble)
    min_off_target_defect = np.min(off_target_ensemble)
    average_off_target_defect = np.mean(off_target_ensemble)

    return id, (max_offtarget_probability, average_offtarget_probability, 
           max_on_target_defect, average_on_target_defect, min_off_target_defect, average_off_target_defect)
    


def evaluate_combinations_thermo(combinations, model, ncores, conc):
    evaluation_func = partial(evaluation_combination_thermo, 
                            model=model, 
                            conc=conc)
    results = np.empty(len(combinations), dtype=object)
    tasks = [(list(zip(*combinations[i]))[0], i) for i in range(len(combinations))]
    with Pool(ncores) as pool:
        for i, result in tqdm(pool.imap(evaluation_func, tasks), total=len(combinations)):
            results[i] = result
    return results
                
    