import difflib
import math
import itertools
import numpy as np
import random 
import nupack as nu
from orthogonal import *
from scipy.stats import rankdata
from tqdm import tqdm

def get_domains(toeholds, full_seq, toehold_length):
    num_domains = len(toeholds) + 1
    domains = np.empty(num_domains, dtype=object)
    domains[0] = full_seq[:toeholds[0][1]]
    for i in range(0, len(toeholds)-1):
        domains[i+1] = full_seq[(toeholds[i][1]+toehold_length):toeholds[i+1][1]]
    domains[num_domains - 1] = full_seq[(toeholds[len(toeholds)-1][1]+toehold_length):]
    
    return domains
"""
Find the longest substring that appears in any 2 strings in strings, a list of strings.
"""
def find_longest_common_substring(strings):
    if len(strings) < 2:
        return "", 0, None
    
    best_substring = ""
    
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            matcher = difflib.SequenceMatcher(None, strings[i], strings[j])
            match = matcher.find_longest_match(0, len(strings[i]), 0, len(strings[j]))
            
            if match.size > len(best_substring):
                best_substring = strings[i][match.a:match.a + match.size]
    
    return len(best_substring)

"""
generate num_sets pairings of toeholds and barcodes
"""
def generate_pairings(toeholds, barcodes, num_sets):
    # since len(toeholds) = len(barcodes)
    # there are len(toeholds)! ways to pair them
    max_sets = math.factorial(len(toeholds))
    if num_sets > max_sets:
        print("cannot generate more sets than the number of possible pairings. will generate all possible pairings")
    
    if num_sets >= max_sets or num_sets == 0:
        # generate all pairings between toeholds and barcodes
        all_pairings = list(itertools.permutations(range(len(toeholds))))
        result = []
        for perm in all_pairings:
            pairing_list = []
            for i, barcode_idx in enumerate(perm):
                pairing_list.append((toeholds[i], barcodes[barcode_idx]))
            result.append(pairing_list)
        return result
    
    else:
        # generate num_sets random pairings between toeholds and barcodes
        result = []
        used_pairings = set()
        
        while len(result) < num_sets:
            # Create a random permutation by shuffling barcode indices
            barcode_indices = list(range(len(barcodes)))
            random.shuffle(barcode_indices)
            
            # Convert to tuple for hashing (to check for duplicates)
            perm_tuple = tuple(barcode_indices)
            
            # Skip if we've already generated this pairing
            if perm_tuple in used_pairings:
                continue
                
            used_pairings.add(perm_tuple)
            
            # Create the pairing
            pairing = []
            for i, barcode_idx in enumerate(barcode_indices):
                pairing.append((toeholds[i], barcodes[barcode_idx]))
            result.append(pairing)
        
        return result

def evaluate_pairings_string_worker(data):
    i, pairing = data 
    longest_common_substring = find_longest_common_substring(pairing)
    return i, [longest_common_substring]

def evaluate_pairings_string(pairings, ncores):
    common_substrings = np.empty(len(pairings), dtype = object)
    tasks = [(i, pairings[i]) for i in range(len(pairings))]
    with Pool(ncores) as pool:
        for i, result in tqdm(pool.imap(evaluate_pairings_string_worker, tasks), total=len(pairings)):
            common_substrings[i] = result
    return common_substrings

def evaluate_pairings_thermo_worker(data):
    i, pairings, unbound_structure, my_model = data
    max_defect = 0
    avg_defect = 0
    for seq in pairings:
        defect = nu.defect(strands=[seq],structure=unbound_structure, model=my_model)
        if defect > max_defect:
            max_defect = defect
        avg_defect += defect
    avg_defect /= len(pairings)
    return i, (max_defect, avg_defect)

def evaluate_pairings_thermo(pairings, my_model, ncores):
    results = np.empty(len(pairings), dtype = object)
    unbound_structure = ss_structure(len(pairings[0][0]))
    tasks = [(i, pairings[i], unbound_structure, my_model) for i in range(len(pairings))]
    with Pool(ncores) as pool:
        for i, result in tqdm(pool.imap(evaluate_pairings_thermo_worker, tasks), total=len(pairings)):
            results[i] = result
    return results
    
def generate_sidewinder_strands(pairing, domains):
    top_strands = []
    bottom_strands = []
    
    for i in range(len(pairing)):
        if i == 0:
            top_strand = domains[i] + pairing[i][0] + pairing[i][1]
            bottom_strand = nu.reverse_complement(domains[i])
        else:
            top_strand = nu.reverse_complement(pairing[i-1][1]) + domains[i] + pairing[i][0] + pairing[i][1]
            bottom_strand = nu.reverse_complement(domains[i]) + nu.reverse_complement(pairing[i-1][0])
        top_strands.append(top_strand)
        bottom_strands.append(bottom_strand)
    
    # last strand 
    top_strand = nu.reverse_complement(pairing[-1][1]) + domains[-1]
    bottom_strand = nu.reverse_complement(domains[-1]) + nu.reverse_complement(pairing[-1][0])
    top_strands.append(top_strand)
    bottom_strands.append(bottom_strand)
    
    return top_strands, bottom_strands
    