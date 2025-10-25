# Template duplex pool generator code
import nupack as nu
from seqwalk import design
import multiprocessing
from orthogonal import *
from timing_utils import TimingRecorder, configure_start_method

# denote duplex setting
duplex = 1

#<MODIFIABLE> 0: Library save file
save_file = "duplex_library.csv"

#<MODIFIABLE> 0: Minimal intermediate reporting, 1: full reporting
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
threshold_ON = 0.9

#<MODIFIABLE> Off-target probability threshold
threshold_OFF = 0.1

#<MODIFIABLE> Melting temperature search lower bound
low = 25

#<MODIFIABLE> Melting temperature search upper bound
high = 85

#<MODIFIABLE> Melting temperature search grain (ie the step value)
grain = 1

#<MODIFIABLE> Minimum difference between highest melting temperatures among off targets 
#             and lowest melting temperature among on targets.
delta = 5 

#<MODIFIABLE> Largest range of melting temperatures allowed for on-target sequences
#             Note that 0 is the default value, which means no range restriction.
my_range = 0

if __name__ == "__main__":
    configure_start_method(verbose=True)
    timer = TimingRecorder(baseline_path="_duplex_timings.json", autosave=True)

    with timer.time_block("STEP 1/9: Generating SSM Hamiltonian Set"):
        library = design.max_size(l, k, alphabet="ACGT", RCfree=duplex)
        print_library_size(library)

    with timer.time_block("STEP 2/9: Similarity Optimization"):
        library = sim_optimization(library, threshold_SIM, reporting, duplex)
        print_library_size(library)

    with timer.time_block("STEP 3/9: Generating Thermodynamic Complex Probabilities"):
        nu_mat, on_t = nupack_matrix_mp(library, my_model, conc, ncores, duplex)

    with timer.time_block("STEP 4/9: ON-Target Optimization"):
        library, on_t, nu_mat = on_target_optimization(on_t, library, threshold_ON,
                                                       reporting, nu_mat)
        print_library_size(library)

    with timer.time_block("STEP 5/9: OFF-Target Optimization"):
        library, on_t, nu_mat = off_target_optimization(nu_mat, library, threshold_OFF,
                                                        reporting, on_t)
        print_library_size(library)

    with timer.time_block("STEP 6/9: Generating Melting Temperatures"):
        library_with_complements = []
        for seq in library:
            library_with_complements.append(seq)
            library_with_complements.append(nu.reverse_complement(seq))
        tm_mat = tm_mp(library_with_complements, low, high, grain, conc, ncores)
        np.save("_tmmat.npy", tm_mat) #to_remove in final
        save_lib(library, "intermediate.csv") #to_remove in final

    with timer.time_block("STEP 7/9: Melting Temperature Filtering"):
        library = tm_optimization(library, tm_mat, delta, reporting)
        print_library_size(library)

    with timer.time_block("STEP 8/9: Melting Temperature Range Optimization"):
        library, best_range = tm_bounds_optimization(library, tm_mat, my_range, reporting)
        print_library_size(library)

    with timer.time_block("STEP 9/9: Saving Library"):
        save_lib(library, save_file)

    timer.report()
