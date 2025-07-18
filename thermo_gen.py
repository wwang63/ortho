import nupack as nu 
import csv
from tqdm import tqdm
import sys
import multiprocessing as mp
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
barcode_length = 18 
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
     
    # import CDS sequences from file
    CDS_sequences = []
    with open(filename, "r",encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0:
                CDS_sequences.append(row[0])
                    
    # thermodynamic generation
    thermo_pool = []

    toehold_idx_seps = [0]
    all_putative_toeholds = []
    # generate the set of all toeholds for all CDSs
    print("Generating toeholds")
    for cds_idx, cds in enumerate(CDS_sequences):    
        putative_toeholds = generate_putative_toeholds(cds, args)
        all_putative_toeholds += putative_toeholds
        toehold_idx_seps.append(len(all_putative_toeholds))

    # first judge the the toeholds based on string metrics
    toe_combinations = toehold_combinations(all_putative_toeholds, args, count)
    toe_string_data = evaluate_combinations_string(toe_combinations, ncores)
    # For top 20% 
    n_top = int(np.ceil(len(toe_string_data) * 0.2))
    top_20_percent_indices = np.argsort(toe_string_data)[:n_top]
    weights = {
            "min_edit_distance": (1.0, 1),
            "average_edit_distance": (0.5, 1)
            }
    top_20_percent_indices = rank_combinations_weighted(toe_string_data,weights,n_top)
    top_toehold_combinations = toe_combinations[top_20_percent_indices]

    # then judge based on thermo metrics 
    toe_thermo_data = evaluate_combinations_thermo(top_toehold_combinations, model, ncores, conc)
    weights = {
        'max_offtarget_probability': (2.0,0),
        'average_offtarget_probability': (1.0,0),
        'max_on_target_defect': (1.0,0),
        'average_on_target_defect': (0.5,0),
        'min_off_target_defect': (1.0,1),
        'average_off_target_defect': (0.5,1)
    }
    best_toe_id = rank_combinations_weighted(toe_thermo_data, weights)
    toeholds = top_toehold_combinations[best_toe_id]
    print(toe_thermo_data[best_toe_id])
    print(toe_string_data[top_20_percent_indices[best_toe_id]])
    toeholds_seqs = [toeholds[i][0] for i in range(len(toeholds))]
    
    # generate barcodes
    print("Generating barcodes")
    barcode_sets = generate_putative_barcodes(toeholds_seqs, q,k,barcode_length,count)
    barcode_string_data = evaluate_barcodes_string(barcode_sets, toeholds_seqs, ncores)

    n_top = int(np.ceil(len(barcode_string_data) * 0.2))
    weights = {
            "min_edit_distance": (1.0, 1),
            "average_edit_distance": (0.5, 1)
            }
    top_20_percent_indices = rank_combinations_weighted(barcode_string_data,weights,n_top)

    top_barcode_combinations = [barcode_sets[i] for i in top_20_percent_indices]
    barcode_data = evaluate_barcodes_thermo(top_barcode_combinations, toeholds_seqs, model, ncores, conc)
    weights = {
                'min_ontarget_probability': (0.5,1), 
                'average_ontarget_probability': (0.25,1),
                'max_offtarget_probability': (2.0,0),
                'average_offtarget_probability': (1.0,0),
                'max_on_target_defect': (2.0,0),
                'average_on_target_defect': (1.0,0),
                'min_off_target_defect': (0.5,1),
                'average_off_target_defect': (0.25,1)
            }
    best_barcodeid = rank_combinations_weighted(barcode_data, weights)
    print(barcode_string_data[top_20_percent_indices[best_barcodeid]])
    barcodes=top_barcode_combinations[best_barcodeid]
    
    # generate pairs
    print("Generating pairs")
    pairings_tuple = generate_pairings(toeholds_seqs,barcodes,num_sets=count)
    pairings = [[pairing[0]+pairing[1] for pairing in pairings_set] for pairings_set in pairings_tuple]
    assert(len(pairings) == len(pairings_tuple))
    evaluated_string_pairings = evaluate_pairings_string(pairings, ncores)
    evaluated_string_pairings = [sublist[0] for sublist in evaluated_string_pairings]
    n_top = int(np.ceil(len(evaluated_string_pairings) * 0.2))
    top_20_percent_indices = np.argsort(evaluated_string_pairings)[:n_top]
    top_pairings_tuple = [pairings_tuple[i] for i in top_20_percent_indices]
    top_pairings = [pairings[i] for i in top_20_percent_indices]

    pairing_data = evaluate_pairings_thermo(top_pairings, model, ncores)
    weights = {
        'structure_defect': (1.0, 0),
        'average_structure_defect': (0.5, 0),
    }
    best_pair_id = rank_combinations_weighted(pairing_data, weights)
    best_pair = top_pairings_tuple[best_pair_id]
    print(evaluated_string_pairings[best_pair_id])
    print(pairing_data[best_pair_id])
    
    # generate final sidewinder strands
    print("Generating final sidewinder strands")
    thermo_pool = []
    for i in range(len(toehold_idx_seps)-1):
        my_toeholds_with_idx = toeholds[toehold_idx_seps[i]:toehold_idx_seps[i+1]]
        my_pairs = best_pair[toehold_idx_seps[i]:toehold_idx_seps[i+1]]
        domains = get_domains(my_toeholds_with_idx, CDS_sequences[i], toehold_length)
        top_strands, bottom_strands = generate_sidewinder_strands(my_pairs, domains)
        thermo_pool.append((top_strands, bottom_strands))
    
    # save thermo_pool
    save_to_csv(thermo_pool, save_filename)
