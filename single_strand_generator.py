# Template duplex pool generator code
import nupack as nu
from seqwalk import design
import multiprocessing
from orthogonal import *
from timing_utils import TimingRecorder, configure_start_method

# denote single stranded setting
duplex = 0

#<MODIFIABLE> 0: Library save file
save_file = "single_library.csv"

#<MODIFIABLE> 0: minimal intermediate reporting, 1: full reporting
reporting = 1 

#<MODIFIABLE> Number of cores for parallel processing
ncores = multiprocessing.cpu_count() 

#<MODIFIABLE> Working concentration of strands in molar
conc = 1e-6 

#<MODIFIABLE> Length of sequences
l = 16

#<MODIFIABLE> SSM K parameter 
k = 6

#<MODIFIABLE> character similarity threshold
threshold_SIM = 12 

#<MODIFIABLE> Standard operating temperature of DNA reaction in celsius
rxn_temp = 37 

#<MODIFIABLE> Nupack model, can change salt conditions here
my_model = nu.Model(material='dna', celsius=rxn_temp) 

#<MODIFIABLE> On-target probability threshold
threshold_ON = 0.7

#<MODIFIABLE> Off-target probability threshold
threshold_OFF = 0.1

if __name__ == "__main__":
    configure_start_method(verbose=True)
    timer = TimingRecorder(baseline_path="_single_timings.json", autosave=True)

    with timer.time_block("STEP 1/6: Generating SSM Hamiltonian Set"):
        library = design.max_size(l, k, alphabet="ACGT", RCfree=duplex)
        print_library_size(library)

    with timer.time_block("STEP 2/6: Similarity Optimization"):
        library = sim_optimization(library, threshold_SIM, reporting, duplex)
        print_library_size(library)

    with timer.time_block("STEP 3/6: Generating Thermodynamic Complex Probabilities"):
        nu_mat, on_t = nupack_matrix_mp(library, my_model, conc, ncores, duplex)

    with timer.time_block("STEP 4/6: ON-Target Optimization"):
        library, on_t, nu_mat = on_target_optimization(on_t, library, threshold_ON,
                                                       reporting, nu_mat)
        print_library_size(library)

    with timer.time_block("STEP 5/6: OFF-Target Optimization"):
        library, on_t, nu_mat = off_target_optimization(nu_mat, library, threshold_OFF,
                                                        reporting, on_t)
        print_library_size(library)

    with timer.time_block("STEP 6/6: Saving Library"):
        save_lib(library, save_file)

    timer.report()
