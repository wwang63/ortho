from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from multiprocessing import Manager
import multiprocessing
import numpy as np
import sys
from tqdm import tqdm
import pickle
import nupack as nu
from seqwalk import design
from orthogonal import *
from timing_utils import TimingRecorder, configure_start_method

save_message = "Do you wish to presently save your library to a user-accessible file? (0:no, 1:yes)\n"

def usage():
    print("Command must be one of: <init>, <optimize>, <filterTm>, <tmRange>, <quit>\n")

def user_save(library, exit):
    # Library File Name
    library_file = input("<Library File>\n")
    print("Writing to file: " + library_file +"\n")
    resultFyle = open(library_file,'w')
    for r in library:
        print(r)
        resultFyle.write(r + "\n")
    resultFyle.close()
    if exit:
        sys.exit(0)

if __name__ == "__main__":
    configure_start_method(verbose=True)

    print("\nYou may notice .npy and .pkl files being generated in your working" +
          " directory. These are used to reload information" +
          " from the last previous instance of the program so you don't have to reinput" +
          " commands. They are required for the optimize and filterTm comamnds.\n")
    
    print("Available commands: <init>, <optimize>, <filterTm>, <tmRange>, <quit>\n")
    print("<init> generates an SSM optimized library and character similarity optimized"+ 
          " library. It will also generate thermodynamic probability matrices.")
    print("<optimize> optimizes the library for on-target and off-target binding" + 
          " probabilities.")
    print("<filterTm> optimizes the library for melting temperatures.")
    print("<tmRange> optimizes the library for melting temperature ranges.")
    print("<quit> exits the program.\n")
    
    print("Note that optimize takes the input probability matrices from the last init call.\n")
    
    
    duplex = int(input("<Single stranded (0) or duplex library (1)>\n"))
    if duplex:
        print("\nWorking to generate a duplex library \n")
    else:
        print("\nWorking to generate a single stranded library \n")
    
    reporting = int(input("<Reporting Status (0:no, 1:yes)>\n"))
    if reporting:
        print("Verbose reporting toggled ON.\n")
    else:
        print("Verbose reporting toggled OFF.\n")
    while True:        
        command = input("\nEnter Command\n")
        
        if command not in ["init", "optimize", "filterTm", "quit", "tmRange"]:
            usage()
            
        elif command == "quit":
            sys.exit(0)
            
        elif (not duplex) and (command == "filterTm" or command == "tmRange"):
            print("Melting temperature filtering is not available on single stranded libraries.")
        
        elif command == "init":
            ncores = int(input("<Number of Cores> (0 to use all available cores)\n"))
            if ncores == 0:
                ncores = multiprocessing.cpu_count()
            nu.config.threads = ncores
            
            conc = float(input("<Working Concentration (M)> "))
            
            # SSM parameters
            l = int(input("<l> "))
            k = int(input("<k> "))
            
            # Symmetry optimization parameters
            threshold_SIM = int(input("<Similarity Threshold> "))
            
            # Standard desired operating temperature of DNA reaction. Used for probability matrix.
            rxn_temp = float(input("<Reaction Temperature (°C)> "))    
            
            my_model = nu.Model(material='dna', celsius=rxn_temp) 
            # ^^ MODIFY W/ FURTHER SALT SPECIFICATIONS IF DESIRED (see nupack documentation)
            
            # Run SSM Hamiltonian Set generation via Seqwalk
            timer = TimingRecorder(baseline_path="_interactive_init_timings.json", autosave=True)

            with timer.time_block("STEP 1/3: Generating SSM Hamiltonian Set"):
                library = design.max_size(l, k, alphabet="ACGT", RCfree=duplex)

            with timer.time_block("STEP 2/3: Similarity Optimization"):
                print(f"Similarity Threshold: {threshold_SIM}\n")
                library = sim_optimization(library, threshold_SIM, reporting, duplex)
                with open('_ssmlibrary.pkl','wb') as f:
                    pickle.dump(library, f)

            with timer.time_block("STEP 3/3: Generating Probability Matrix"):
                nu_mat, on_t = nupack_matrix_mp(library, my_model, conc, ncores, duplex)
                np.save("_ssmnumat.npy", nu_mat)
                np.save("_ssmon_t.npy", on_t)

            print(f"\nFinal library size: {len(library)}\n")
            timer.report()
            
            to_save = int(input(save_message)) 
            if to_save:
                user_save(library, 0)
            
        elif command == "optimize":
            nu_mat = np.load("_ssmnumat.npy")
            on_t = np.load("_ssmon_t.npy")
            with open('_ssmlibrary.pkl','rb') as f:
                library = pickle.load(f)
            
            # Off target optimization threshold
            threshold_OFF = float(input("<Off-Target Probability Threshold> "))
            
            # On target optimization threshold
            threshold_ON = float(input("<On-Target Probability Threshold> "))
                        
            timer = TimingRecorder(baseline_path="_interactive_opt_timings.json", autosave=True)

            with timer.time_block("STEP 1/2: On-Target Optimization"):
                print(f"On-Target Threshold: {threshold_ON}\n")
                library, on_t, nu_mat  = on_target_optimization(on_t, library, threshold_ON,
                                                       reporting, nu_mat)

            with timer.time_block("STEP 2/2: Off-Target Optimization"):
                print(f"Off-Target Threshold: {threshold_OFF}\n")
                library, on_t, nu_mat  = off_target_optimization(nu_mat, library, threshold_OFF,
                                                        reporting, on_t)

            print(f"\nFinal library size: {len(library)}\n")
            timer.report()
            
            if duplex: 
                with open('_optlibrary.pkl','wb') as f:
                    pickle.dump(library, f)
                np.save("_optnumat.npy", nu_mat) # user can recover if needed
                np.save("_opton_t.npy", on_t)
                to_save = int(input(save_message)) 
                if to_save:
                    user_save(library, 0)
            else:
                print("Single stranded library generation complete. Please save.\n")
                user_save(library,0)
                
        elif command == "filterTm":
            #nu_mat = np.load("_optnumat.npy")
            #tm_mat = np.load("_opttmmat.npy")
            with open('_optlibrary.pkl','rb') as f:
                library = pickle.load(f)
                
            ncores = int(input("<Number of Cores> (0 to use all available cores)\n"))
            if ncores == 0:
                ncores = multiprocessing.cpu_count()
            nu.config.threads = ncores
            
            conc = float(input("<Working Concentration (M)> "))
            
            low = float(input("<Tm Low Bound (°C)> "))
            high = float(input("<Tm High Bound (°C)> "))
            grain = float(input("<Tm Grain (°C)> "))
            
            # Melting temperature optimization delta
            delta = int(input("<Off and On Target Desired Tm Difference>"))
                    
            timer = TimingRecorder(baseline_path="_interactive_filter_tm_timings.json", autosave=True)

            with timer.time_block("STEP 1/1: Melting Temperature Optimization"):
                library_with_complements = []
                for seq in library:
                    library_with_complements.append(seq)
                    library_with_complements.append(nu.reverse_complement(seq))
                tm_mat = tm_mp(library_with_complements, low, high, grain, conc, ncores)
                print(tm_mat)
                library = tm_optimization(library, tm_mat, delta, reporting)

            with open('_tmfilterlibrary.pkl','wb') as f:
                pickle.dump(library, f)
            np.save("_tmmat.npy", tm_mat)
            to_save = int(input(save_message))
            if to_save:
                user_save(library, 0)

            timer.report()
        
        elif command == "tmRange":
            tm_mat = np.load("_tmmat.npy")
            with open('_tmfilterlibrary.pkl','rb') as f:
                library = pickle.load(f)
            
            my_range = float(input("<Melting Temperature Max Desired Range (°C)> "))
            
            timer = TimingRecorder(baseline_path="_interactive_tm_range_timings.json", autosave=True)

            with timer.time_block("STEP 1/1: Melting Temperature Range Optimization"):
                library, best_range = tm_bounds_optimization(library, tm_mat, my_range, reporting)

            #program complete for duplex generation
            print("Duplex library generation complete. Please save.")
            print(f"\nFinal library size: {len(library)}\n")
            user_save(library,0)
            timer.report()
