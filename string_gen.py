import nupack as nu 
import numpy as np 
from seqwalk import design
import math
import sys
import random 
import csv
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from tqdm import tqdm

# import custom modules
if 'barcodes' in sys.modules:
    del sys.modules['barcodes']
if 'toeholds' in sys.modules:
    del sys.modules['toeholds']
if 'finalize_seqs' in sys.modules:
    del sys.modules['finalize_seqs']
if 'analysis' in sys.modules:
    del sys.modules['analysis']

from orthogonal import *
from barcodes import *
from toeholds import *
from finalize_seqs import *
from analysis import *

"""
Save the _pool data to a CSV file.

Args:
    pool: List of tuples (top_strands, bottom_strands) for each CDS
    filename: Output CSV filename
"""
def save_to_csv(pool, filename="sidewinder_strands.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['CDS_Index', 'Strand_Type', 'Strand_Index', 'Sequence'])
        
        # Write data
        for cds_idx, (top_strands, bottom_strands) in enumerate(pool):
            # Write top strands
            for strand_idx, strand in enumerate(top_strands):
                writer.writerow([cds_idx, 'top', strand_idx, strand])
            
            # Write bottom strands
            for strand_idx, strand in enumerate(bottom_strands):
                writer.writerow([cds_idx, 'bottom', strand_idx, strand])
                
# Settings 
#oligo_length = 96 
oligo_length = 150
barcode_length = 22 
fragment_length = oligo_length - 2 * barcode_length
max_final_c_domain_length = oligo_length - barcode_length
toehold_length = 10
toehold_search_range = 15
identity_threshold = 0.6
args = oligo_length, barcode_length, fragment_length, max_final_c_domain_length, toehold_length, toehold_search_range, identity_threshold

model =nu.Model(material="dna", celsius=50, sodium=0.154, magnesium=0.01)
ncores = mp.cpu_count()
conc = 1e-8

q = 7
k = 9

count = 1000

# check for correct call and filenmae
if len(sys.argv) < 3:
    print("Usage: python thermo_gen.py <filename to read> <filename to save>")
    sys.exit(1) # Exit with an error code

filename = sys.argv[1]
save_filename = sys.argv[2]


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    # import CDS sequences
    CDS_sequences = []
    #filename = "test_native_CDS_first_24_no_Hip205.csv"
    with open(filename, "r",encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0:
                CDS_sequences.append(row[0])
                

    # string generation 
    string_pool = []
    toehold_idx_seps = [0]
    all_putative_toeholds = []

    # generate the set of all toeholds for all CDSs
    print("Generating toeholds")
    for cds_idx, cds in enumerate(CDS_sequences):    
        putative_toeholds = generate_putative_toeholds(cds, args)
        all_putative_toeholds += putative_toeholds
        toehold_idx_seps.append(len(all_putative_toeholds))

    combinations = toehold_combinations(all_putative_toeholds, args, count)
    data = evaluate_combinations_string(combinations, ncores)
    weights = {
            "min_edit_distance": (1.0, 1),
            "average_edit_distance": (0.5, 1)
            }
    bestid = rank_combinations_weighted(data,weights)
    toeholds = combinations[bestid]
    toeholds_seqs = [toeholds[i][0] for i in range(len(toeholds))]

    # generate barcodes
    print("Generating barcodes")
    barcode_sets = generate_putative_barcodes(toeholds_seqs, q,k,barcode_length, count)
    barcode_data = evaluate_barcodes_string(barcode_sets, toeholds_seqs, ncores)
    weights = {
            'min_edit_distance': (1.0,1),
            'average_edit_distance': (0.5,1)
        }
    best_barcodeid = rank_combinations_weighted(barcode_data, weights)
    barcodes=barcode_sets[best_barcodeid]
        
    # generate pairs
    print("Generating pairs")
    pairings = generate_pairings(toeholds_seqs,barcodes,num_sets=count)
    pairings_string = [[pairs[0] + pairs[1] for pairs in pairing] for pairing in pairings]
    evaluated_pairings = evaluate_pairings_string(pairings_string, ncores)
    weights = {
            'longest_common_substring': (1.0,0)
        }
    best_pair_id =rank_combinations_weighted(evaluated_pairings,weights)
    best_pair = pairings[best_pair_id]

    # generate final sidewinder strands
    for i in range(len(toehold_idx_seps)-1):
        my_toeholds_with_idx = toeholds[toehold_idx_seps[i]:toehold_idx_seps[i+1]]
        my_pairs = best_pair[toehold_idx_seps[i]:toehold_idx_seps[i+1]]
        domains = get_domains(my_toeholds_with_idx, CDS_sequences[i], toehold_length)
        top_strands, bottom_strands = generate_sidewinder_strands(my_pairs, domains)
        string_pool.append((top_strands, bottom_strands))

    save_to_csv(string_pool, save_filename)
