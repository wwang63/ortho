from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import nupack as nu

""" 
Returns a dot parens string representation of a duplex structure. 

Args:
    n: length of the duplex structure

Returns:
    str: dot parens string representation of the duplex structure
"""
def duplex_structure(n):
    return "("*n + "+" + ")"*n

""" 
Returns a dot parens string representation of a single stranded structure. 

Args:
    n: length of the structure's underlying DNA sequence

Returns:
    str: dot parens string representation of the single stranded structure
"""
def ss_structure(n):
    return "."*n

"""  
Compute ensemble defect of a structure between two sequences. 

Args:
    seq1: DNA sequence string
    seq2: DNA sequence string
    model: nupack model
    structure: dot parens string representation of the structure

Returns:
    float: Ensemble defect of the structure between the two sequences
"""
def ensemble_defect_pair(seq1, seq2, model, structure):
    P = nu.pairs(strands=[seq1, seq2], model=model).to_array()
    S = nu.Structure(structure).matrix()
    n = len(seq1) * 2
    return  (n - (P*S).sum())/n

""" 
Compute ensemble defect of a structure for a single sequence. 

Args:
    seq1: DNA sequence string
    model: nupack model
    structure: dot parens string representation of the structure

Returns:
    float: Ensemble defect of the structure for the sequence
"""
def ensemble_defect_sole(seq1, model, structure):
    P = nu.pairs(strands=[seq1], model=model).to_array()
    S = nu.Structure(structure).matrix()
    n = len(seq1) 
    return  (n - (P*S).sum())/n

"""
Compute ensemble defects for on target structures.  

For a duplex design, it will compute the ensemble defect for A+~A (duplex), A (single 
stranded), and ~A (single stranded). 

For a single stranded design, it will compute the ensemble defect for A (single stranded). 

Args:
    seq1: DNA sequence string
    model: nupack model
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    [float]: List of ensemble defects for the on target structures. 
"""
def np_ontarget_ensemble(seq1, model, duplex):
    if duplex:
        return [ensemble_defect_pair(seq1, nu.reverse_complement(seq1), model, 
                                     duplex_structure(len(seq1))), 
                ensemble_defect_sole(seq1, model, ss_structure(len(seq1))), 
                ensemble_defect_sole(nu.reverse_complement(seq1), model, 
                                     ss_structure(len(seq1)))]
    else:
        return [ensemble_defect_sole(seq1, model, ss_structure(len(seq1)))]

"""
Compute ensemble defects for off target structures. 

For a duplex design, it will compute the ensemble defect for A+B (duplex), A+~B (duplex), 
~A+B (duplex), and ~A+~B (duplex). 

For a single stranded design, it will compute the ensemble defect for A+B (duplex). 

Args:
    seq1: DNA sequence string
    seq2: DNA sequence string
    model: nupack model
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    [float]: List of off-target concentrations.
"""
def np_cross_ensemble(seq1, seq2, model, duplex):
    if duplex:
        o1 = ensemble_defect_pair(seq1, seq2, model, duplex_structure(len(seq1)))
        o2 = ensemble_defect_pair(seq1, nu.reverse_complement(seq2), model, 
                                  duplex_structure(len(seq1)))
        o3 = ensemble_defect_pair(nu.reverse_complement(seq1), seq2, model, 
                                  duplex_structure(len(seq1)))
        o4 = ensemble_defect_pair(nu.reverse_complement(seq1), nu.reverse_complement(seq2), 
                                  model, duplex_structure(len(seq1)))
        return [o1, o2, o3, o4]
    else:
        return [ensemble_defect_pair(seq1, seq2, model, duplex_structure(len(seq1)))]
    
"""
Compute expected off-target concentration for a given sequence on its own via NUPACK 
analysis, when considering all max size = 2 possible structures

For a duplex design, it will compute the ensemble defect for A+A (duplex) and ~A+~A 
(duplex). 

For a single stranded design, it will compute the ensemble defect for A+A (duplex). 

Args:
    seq1: DNA sequence string
    model: nupack model
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    [float]: List of off-target concentrations.
"""
def np_diag_ensemble(seq1, model, duplex):
    if duplex:
        o1 = ensemble_defect_pair(seq1, seq1, model, duplex_structure(len(seq1)))
        o2 = ensemble_defect_pair(nu.reverse_complement(seq1), nu.reverse_complement(seq1), 
                                  model, duplex_structure(len(seq1)))
        return [o1, o2]
    else:
        o1 = ensemble_defect_pair(seq1, seq1, model, duplex_structure(len(seq1)))
        return [o1]

""" 
Helper function for ensemble_matrix_mp and ensemble_matrix_no_mp.

Computes a row of the off-target ensemble matrix for a given sequence. 
The off-target ensemble matrix is a matrix where each row/column index represents a 
sequence in the library. Each entry in the off-target ensemble matrix represents the 
minimum ensemble defect of some off target complex between the sequence at the row index 
and the sequence at the column index. Only diagonal entries consider self off-targets 
for instance the complex ~A:~A. 

Also computes the maximum on-target ensemble defect between the sequence at index i with 
its reverse complement, and i and its reverse complement alone. 

Args:
    library: list of seqs
    model: nupack model
    conc: initial molar concentrations of strands 
    i: index of the sequence to compute the row for
    duplex: Whether we are designing a duplex (1) or single stranded library (0)
    
Returns:
    i: index of the sequence to compute the row for
    row[i:]: upper triangle of the np_probs row
    on_p: on-target binding probability of the sequence at index i
"""
def ensemble_row_worker(args):
    library, model, i, duplex = args
    size = len(library)
    row = np.zeros(size)
    on_p = np.max(np_ontarget_ensemble(library[i], model, duplex))
    
    for j in range(i, size):
        if i != j:
            res = np_cross_ensemble(library[i], library[j], model, duplex)
            p = np.min(res)
        else:
            res = np_diag_ensemble(library[i], model, duplex)
            p = np.min(res)
        
        row[j] = p  # only upper triangle [i, j]

    return i, row[i:], on_p  # return upper triangle 

"""
Generates the off-target probability matrix and the on-target binding probability array.
Uses multiprocessing. 

For the off-target matrix: 
    each entry [i,j] represents the maximum probability of a given off-target complex
    possible when seq i is exposed to j. (Self off-targets are only considered in the 
    diagonal entries.)
    
For the on-target binding probability array:
    each entry [i] represents the probability of on-target binding of seq i with its 
    reverse complement.

Args:
    library: list of seqs
    model: nupack conditions
    conc: molar concentrations of strands
    ncores: number of cores to use for parallel processing
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    np_probs: NxN numpy array off-target probability matrix (N is the size of the library)
    on_probs: length N numpy array on-target binding probability array
"""
def ensemble_matrix_mp(library, model, ncores, duplex):
    size = len(library)
    tasks = [(library, model, i, duplex) for i in range(size)]

    print("Generating...\n")    # generates upper triangle
    off_target_ensemble = np.zeros((size, size))
    on_target_ensemble = np.zeros(size)

    with Pool(ncores) as pool:
        for i, row_slice, on_t in tqdm(pool.imap(ensemble_row_worker, tasks), total=size):
            off_target_ensemble[i, i:] = row_slice
            off_target_ensemble[i+1:, i] = row_slice[1:]  #fill lower triangle by symmetry
            on_target_ensemble[i] = on_t

    return off_target_ensemble, on_target_ensemble

"""
Generates the off-target probability matrix and the on-target binding probability array.
Does not use multiprocessing. 

For the off-target matrix: 
    each entry [i,j] represents the maximum probability of a given off-target complex
    possible when seq i is exposed to j. (Self off-targets are only considered in the 
    diagonal entries.)
    
For the on-target binding probability array:
    each entry [i] represents the probability of on-target binding of seq i with its 
    reverse complement.

Args:
    library: list of seqs
    model: nupack conditions
    conc: molar concentrations of strands
    ncores: number of cores to use for parallel processing
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    np_probs: NxN numpy array off-target probability matrix (N is the size of the library)
    on_probs: length N numpy array on-target binding probability array
"""
def ensemble_matrix_no_mp(library, model, duplex):
    size = len(library)
    off_target_ensemble = np.zeros((size, size))
    on_target_ensemble = np.zeros(size)

    for i in range(size):
        i, row_slice, on_t = ensemble_row_worker((library, model, i, duplex))
        off_target_ensemble[i, i:] = row_slice
        off_target_ensemble[i+1:, i] = row_slice[1:]  #fill lower triangle by symmetry
        on_target_ensemble[i] = on_t

    return off_target_ensemble, on_target_ensemble

""" 
Off-target complex ensemble defect optimization. Removes sequences from library and 
ensemble matrices that have too low an off-target complex ensemble defect, below a 
specified threshold.

At a high level, we find pairs of sequences that have low off-target ensemble defects 
and remove the one with the lower total off-target score. (If equal, remove
at random). We repeat this process until no pairs of sequences have lower than threshold 
off-target ensemble defects. 

Args:
    off_target_ensemble: off-target ensemble matrix
    library: list of seqs
    threshold: threshold for off-target ensemble defect
    reporting: Bool, True for verbose output
    on_target_ensemble: on-target ensemble matrix

Returns:
    library: optimized list of seqs
    final_off_target_ensemble: optimized off-target ensemble matrix
    final_on_target_ensemble: optimized on-target ensemble matrix
"""
def ensemble_off_target_optimization(off_target_ensemble, library, threshold, reporting, on_target_ensemble):
    assert(off_target_ensemble.shape[0] == off_target_ensemble.shape[1])
    assert(off_target_ensemble.shape[0] == len(library))
    n = off_target_ensemble.shape[0]
    active = np.ones(n, dtype=bool)  # Tracks which sequences are still active
    working_matrix = np.copy(off_target_ensemble)
    
    # first filter based off diagonal of matrix
    for i in range(n):
        active[i] = (working_matrix[i,i] >= threshold)
        if reporting and not active[i]:
            print("Too small self on-target complex ensemble defect detected")
            print(f"seq {i}: " + library[i] + f", defect = {working_matrix[i,i]}")
    
    # filter based pair off-target ensemble defects
    while True:
        # mask inactive rows/columns
        working_matrix[~active, :] = np.inf
        working_matrix[:, ~active] = np.inf

        min_i, min_j = np.unravel_index(np.argmin(working_matrix), off_target_ensemble.shape)
        min_value = working_matrix[min_i, min_j]
        
        if min_value >= float(threshold):
            break # loop until no more values above threshold

        if reporting:
            print("Too small an off-target complex ensemble defect detected")
            print(f"seq {min_i}, seq {min_j}, defect = {min_value}")
            print(f"seq {min_i}: " + library[min_i])
            print(f"seq {min_j}: " + library[min_j])

        # decide which sequence to remove based on total off-target score
        score_i = np.sum(working_matrix[min_i, :])
        score_j = np.sum(working_matrix[min_j, :])
        # note that they share an entry so no need to subtract that
        
        if score_i < score_j:
            remove_idx = min_i
        elif score_i > score_j:
            remove_idx = min_j
        else:
            remove_idx = np.random.choice([min_i, min_j])
            
        active[remove_idx] = False

        if reporting:
            print(f"Removed seq {remove_idx}, Remaining Elements {np.sum(active)}")

    # final filtering
    final_library = [seq for i, seq in enumerate(library) if active[i]]
    final_off_target_ensemble = off_target_ensemble[np.ix_(active, active)]
    final_on_target_ensemble = on_target_ensemble[active]

    return final_library, final_off_target_ensemble, final_on_target_ensemble

""" 
On-target complex ensemble defect optimization. Removes sequences from library and 
ensemble matrices that have too high an on-target complex ensemble defect, above a 
specified threshold. 

Args:
    on_target_ensemble: on-target ensemble matrix
    library: list of seqs
    threshold: threshold for on-target ensemble defect
    reporting: Bool, True for verbose output
    off_target_ensemble: off-target ensemble matrix

Returns:
    library: optimized list of seqs
    final_on_target_ensemble: optimized on-target ensemble matrix
    final_off_target_ensemble: optimized off-target ensemble matrix
"""
def ensemble_on_target_optimization(on_target_ensemble, library, threshold_ON, reporting, off_target_ensemble):
    assert(len(on_target_ensemble) == len(library))
    n = len(on_target_ensemble)
    active = np.ones(n, dtype=bool)  # Tracks which sequences are still active
    for i in range(n):
        active[i] = (on_target_ensemble[i] <= threshold_ON)
        if reporting and not active[i]:
            print(f"seq {i} has too high an on-target complex ensemble defect")
            print(f"seq {i}: " + library[i] + f"    defect = {on_target_ensemble[i]}")

    # final filtering
    final_library = [seq for i, seq in enumerate(library) if active[i]]
    final_on_target_ensemble = on_target_ensemble[active]
    final_off_target_ensemble = off_target_ensemble[np.ix_(active, active)]

    return final_library, final_on_target_ensemble, final_off_target_ensemble

"""
Compute expected on-target concentration for a given sequence via NUPACK analysis, when 
considering all max size = 2 possible structures. 

For a duplex design, it will compute [A+~A] in a test tube at the specified model and at 
starting concentrations [A] = [~A] = conc. 

For a single stranded design, it will compute [A] in a test tube at the specified 
model and at starting concentration [A] = conc.

Args:
    seq1: DNA sequence string
    seq2: DNA sequence string
    model: nupack model
    conc: Initial concentration of strands in molar
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    float: Expected on-target concentration of a duplex or single stranded species given 
           the model and initial concentrations of conc. 
"""
def np_ontarget(seq1, model, conc, duplex):
    A = nu.Strand(seq1, name='A')
    if duplex:
        c1 = nu.Complex([A, ~A], name="c1")
        my_tube = nu.Tube(strands={A: conc,~A: conc},
                    name='my_tube', complexes=nu.SetSpec(max_size=2, include = [c1]))
    else:
        c1 = nu.Complex([A], name="c1")
        my_tube = nu.Tube(strands={A: conc},
                    name='my_tube', complexes=nu.SetSpec(max_size=2, include = [c1]))  
    tube_results = nu.tube_analysis(tubes=[my_tube], model=model)
    return tube_results.tubes[my_tube].complex_concentrations[c1]

"""
Compute expected off-target concentration between a given pair of sequences via NUPACK 
analysis, when considering all max size = 2 possible structures. 

For a duplex system, for strands A and B, it will compute the concentrations of:
1. [A+B] when [A] = [B] = conc
2. [A+~B] when [A] = conc and [~B] = conc
3. [~A+B] when [~A] = conc and [B] = conc
4. [~A+~B] when [~A] = [~B] = conc

For a single stranded species system, it will compute the concentrations of:
1. [A+B] when [A] = conc and [B] = conc

Concentrations are computed in a test tube at the specified model. 

Args:
    seq1: DNA sequence string
    seq2: DNA sequence string
    model: nupack model
    conc: Initial concentration of strands in molar
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    [float]: List of off-target concentrations.
"""
def np_crosstalk(seq1, seq2, model, conc, duplex):
    A = nu.Strand(seq1, name='A')
    B = nu.Strand(seq2, name='B')
    if duplex:
        # off target testing
        o1 = nu.Complex([A, B], name="o1")
        o1_tube = nu.Tube(strands={A: conc, B: conc},
                    name='o1_tube', complexes=nu.SetSpec(max_size=2, include = [o1]))
        o2 = nu.Complex([A, ~B], name="o2")
        o2_tube = nu.Tube(strands={A: conc, ~B: conc},
                    name='o2_tube', complexes=nu.SetSpec(max_size=2, include = [o2]))
        o3 = nu.Complex([~A, B], name="o3")
        o3_tube = nu.Tube(strands={~A: conc, B: conc},
                    name='o3_tube', complexes=nu.SetSpec(max_size=2, include = [o3]))
        o4 = nu.Complex([~A, ~B], name="o4")
        o4_tube = nu.Tube(strands={~A: conc, ~B: conc},
                    name='o4_tube', complexes=nu.SetSpec(max_size=2, include = [o4]))
        
        # compute tubes
        tube_results = nu.tube_analysis(tubes=[o1_tube,o2_tube,o3_tube,o4_tube], 
                                        model=model)
        
        # get concentrations 
        o1_conc = tube_results.tubes[o1_tube].complex_concentrations[o1]
        o2_conc = tube_results.tubes[o2_tube].complex_concentrations[o2]
        o3_conc = tube_results.tubes[o3_tube].complex_concentrations[o3]
        o4_conc = tube_results.tubes[o4_tube].complex_concentrations[o4]

        return [o1_conc, o2_conc, o3_conc, o4_conc]
    else:
        o1 = nu.Complex([A,B], name="o1")
        o1_tube = nu.Tube(strands={A: conc, B: conc},
                    name='o1_tube', complexes=nu.SetSpec(max_size=2, include = [o1]))
        
        # compute tubes
        tube_results = nu.tube_analysis(tubes=[o1_tube], model=model)
        
        # get concentrations
        o1_conc = tube_results.tubes[o1_tube].complex_concentrations[o1]
        
        return [o1_conc] 
    
"""
Compute expected off-target concentration for a given sequence on its own via NUPACK 
analysis, when considering all max size = 2 possible structures

For a duplex system, for strand A, it will compute the concentrations of:
1. [A+A] when [A] = 2*conc
2. [~A+~A] when [~A] = 2*conc

For a single stranded species system, it will compute the concentrations of:
1. [A+A] when [A] = 2*conc

Concentrations are computed in a test tube at the specified model. 

Args:
    seq1: DNA sequence string
    model: nupack model
    conc: Initial concentration of strands in molar
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    [float]: List of off-target concentrations.
"""
def np_crosstalk_diag(seq1, model, conc, duplex):
    A = nu.Strand(seq1, name='A')
    if duplex:
        # off target testing
        o1 = nu.Complex([A, A], name="o1")
        o1_tube = nu.Tube(strands={A: conc*2},
                    name='o1_tube', complexes=nu.SetSpec(max_size=2, include = [o1]))
        o2 = nu.Complex([~A, ~A], name="o2")
        o2_tube = nu.Tube(strands={~A: conc*2},
                    name='26_tube', complexes=nu.SetSpec(max_size=2, include = [o2]))
        
        # compute tubes
        tube_results = nu.tube_analysis(tubes=[o1_tube,o2_tube], model=model)
        
        # get concentrations 
        o1_conc = tube_results.tubes[o1_tube].complex_concentrations[o1]
        o2_conc = tube_results.tubes[o2_tube].complex_concentrations[o2]

        return [o1_conc, o2_conc]
    else:
        o1 = nu.Complex([A,A], name="o1")
        o1_tube = nu.Tube(strands={A: conc*2},
                    name='o1_tube', complexes=nu.SetSpec(max_size=2, include = [o1]))
        
        # compute tubes
        tube_results = nu.tube_analysis(tubes=[o1_tube], model=model)
        
        # get concentrations
        o1_conc = tube_results.tubes[o1_tube].complex_concentrations[o1]
        
        return [o1_conc] 
    
"""
Given sequences seq1 and seq2, compute the final expected concentration of 
1. [seq1:seq2]
2. [seq1] 
3. [seq2]
given the nupack model and initial [seq1] = conc, [seq2] = conc. When doing this, we 
ignore seq1:seq1 and seq2:seq2 as possible structures. This is used for the 
melting temperature calculation. 

Args:
    seq1: DNA sequence string
    seq2: DNA sequence string
    model: nupack model
    conc: Concentration of strands in molar

Returns:
    [float]: List of concentrations of [seq1:seq2], [seq1], and [seq2].
"""
def np_crosstalk_tm(seq1, seq2, t, conc=1e-8):
    model = nu.Model(material='dna', celsius=t) # you may want to modify this for your purposes
    A = nu.Strand(seq1, name='A')
    B = nu.Strand(seq2, name='B')
    c1 = nu.Complex([A, B], name="c1")
    c1_ignore = nu.Complex([A, A], name="c1_ignore")
    c2_ignore = nu.Complex([B, B], name="c2_ignore")
    my_tube = nu.Tube(strands={A: conc, B: conc}, 
                      name='my_tube', complexes=nu.SetSpec(max_size=2, 
                      exclude = [c1_ignore, c2_ignore], 
                      include = [c1]))
    try:
        tube_results = nu.tube_analysis(tubes=[my_tube], model=model)
        return [tube_results.tubes[my_tube].complex_concentrations[c] for c in [c1]]
    except Exception as e:
        print(f"Error in np_crosstalk_tm for {seq1} and {seq2} at {t}°C: {e}")
        return [0.0]  # Returning zero as a fallback value

""" 
Helper function for nupack_matrix_mp and nupack_matrix_no_mp.
 
Computes a row of the np_probs matrix for a given sequence. np_probs is a matrix where 
each row/column index represents a sequence in the library. Each entry in np_probs 
represents the maximum off target probability of some off target complex between the 
sequence at the row index and the sequence at the column index. Only diagonal entries 
consider self off-targets for instance the complex ~A:~A. 

Also computes the on-target binding probability of the sequence with its reverse 
complement. 

Note that we treat a probability as the expected concentration of some on or off target 
complex, divided by the initial species concentration. 

Args:
    library: list of seqs
    model: nupack model
    conc: initial molar concentrations of strands 
    i: index of the sequence to compute the row for
    duplex: Whether we are designing a duplex (1) or single stranded library (0)
    
Returns:
    i: index of the sequence to compute the row for
    row[i:]: upper triangle of the np_probs row
    on_p: on-target binding probability of the sequence at index i
"""
def row_worker(args):
    library, model, conc, i, duplex = args
    size = len(library)
    row = np.zeros(size)
    on_p = np_ontarget(library[i], model, conc, duplex) / conc
    
    for j in range(i, size):
        if i != j:
            res = np_crosstalk(library[i], library[j], model, conc, duplex)
            p = np.max(res) / conc
        else:
            res = np_crosstalk_diag(library[i], model, conc, duplex)
            p = np.max(res) / conc
        
        row[j] = p  # only upper triangle [i, j]

    return i, row[i:], on_p  # return upper triangle 

"""
Generates the off-target probability matrix and the on-target binding probability array. 
Uses multiprocessing. 

For the off-target matrix: 
    each entry [i,j] represents the maximum probability of a given off-target complex
    possible when seq i is exposed to j. (Self off-targets are only considered in the 
    diagonal entries.)
    
For the on-target binding probability array:
    each entry [i] represents the probability of on-target binding of seq i with its 
    reverse complement.

Args:
    library: list of seqs
    model: nupack conditions
    conc: molar concentrations of strands
    ncores: number of cores to use for parallel processing
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    np_probs: NxN numpy array off-target probability matrix (N is the size of the library)
    on_probs: length N numpy array on-target binding probability array
"""
def nupack_matrix_mp(library, model, conc, ncores, duplex):
    size = len(library)
    tasks = [(library, model, conc, i, duplex) for i in range(size)]

    print("Generating...\n")    # generates upper triangle
    np_probs = np.zeros((size, size))
    on_probs = np.zeros(size)

    with Pool(ncores) as pool:
        for i, row_slice, on_t in tqdm(pool.imap(row_worker, tasks), total=size):
            np_probs[i, i:] = row_slice
            np_probs[i+1:, i] = row_slice[1:]  # fill lower triangle by symmetry
            on_probs[i] = on_t

    return np_probs, on_probs

"""
Generates the off-target probability matrix and the on-target binding probability array. 
Does not use multiprocessing. 

For the off-target matrix: 
    each entry [i,j] represents the maximum probability of a given off-target complex
    possible when seq i is exposed to j. (Self off-targets are only considered in the 
    diagonal entries.)
    
For the on-target binding probability array:
    each entry [i] represents the probability of on-target binding of seq i with its 
    reverse complement.

Args:
    library: list of seqs
    model: nupack conditions
    conc: molar concentrations of strands
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    np_probs: NxN numpy array off-target probability matrix (N is the size of the library)
    on_probs: length N numpy array on-target binding probability array
"""
def nupack_matrix_no_mp(library, model, conc, duplex):
    size = len(library)
    np_probs = np.zeros((size, size))
    on_probs = np.zeros(size)

    for i in range(size):
        i, row_slice, on_t = row_worker((library, model, conc, i, duplex))
        np_probs[i, i:] = row_slice
        np_probs[i+1:, i] = row_slice[1:]  # fill lower triangle by symmetry
        on_probs[i] = on_t
    return np_probs, on_probs

""" 
Generates a character similarity matrix. 

If designing for a duplex library, each entry [i,j] is the maximum number of matching 
characters among:
1. between seq i and seq j 
2. between seq i and the reverse complement of seq j
3. between the reverse complement of seq i and seq j
4. between the reverse complement of seq i and the reverse complement of seq j

If designing for a single stranded library, each entry [i,j] is the number of matching 
characters between seq i and seq j.

Args:
    library: list of seqs
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)
    
Returns:
    similar: NxN numpy array of character similarity scores
"""
def sim_matrix(library, duplex):
    similar = np.zeros((len(library),len(library)))
    for i in tqdm(range(len(similar))):
        for j in range(i,len(similar)):
            if not duplex:
                val = sum(map(str.__eq__, library[i], library[j]))
            else: 
                # get highest similarity with all 4 possible combinations of seqs i and j
                # by considering their reverse complements also
                vals = []
                vals.append(sum(map(str.__eq__, library[i], library[j])))
                vals.append(sum(map(str.__eq__, library[i], 
                                    nu.reverse_complement(library[j]))))
                vals.append(sum(map(str.__eq__, nu.reverse_complement(library[i]), 
                                    library[j])))
                vals.append(sum(map(str.__eq__, nu.reverse_complement(library[i]), 
                                    nu.reverse_complement(library[j]))))
                val = np.max(vals)
            similar[i][j] = val
            similar[j][i] = val
    
    return similar

""" 
Similarity optimization. Removes sequences from library and probability matrix that 
are too similar above a specified threshold, threshold_SIM. 

At a high level, we find pairs of sequences that are too similar and remove the one with 
the highest similarity with some other sequence. (If equal, remove at random). We repeat 
this process until no pairs of sequences are too similar above the threshold. 

Args:
    library: list of seqs
    threshold_SIM: threshold for similarity
    reporting: Bool, True if you want to print out the process
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    library: optimized list of seqs
    
"""
def sim_optimization(library, threshold_SIM, reporting, duplex):
    working_matrix = sim_matrix(library, duplex)
    n = working_matrix.shape[0]
    
    # Zero out the diagonal to ignore self-similarity
    np.fill_diagonal(working_matrix, 0)

    active = np.ones(n, dtype=bool)  # Keep track of active (non-removed) sequences
    
    while True:
        # Mask inactive rows/cols as zero and ignore
        working_matrix[~active, :] = 0
        working_matrix[:, ~active] = 0
        
        max_val = np.max(working_matrix)
        if max_val <= threshold_SIM:
            break   # loop until similarity criterion satisfied

        i, j = np.unravel_index(np.argmax(working_matrix), working_matrix.shape)
        
        if reporting:
            print(f"{library[i]} and {library[j]} are similar")
            print(f"Similarity score: {working_matrix[i][j]}")

        # Evaluate which to remove based on max similarity to others
        a = np.copy(working_matrix[i])
        b = np.copy(working_matrix[j])
        a[j] = 0    # ignoring each other...
        b[i] = 0

        if np.max(a) > np.max(b):
            to_remove = i
        elif np.max(a) < np.max(b):
            to_remove = j
        else:
            to_remove = np.random.choice([i, j])  # Randomly choose if equal

        active[to_remove] = False

        if reporting:
            print(f"Removing {library[to_remove]}")
            print(f"Remaining Elements {np.sum(active)}")

    # Rebuild final filtered library
    final_library = [seq for idx, seq in enumerate(library) if active[idx]]
    return final_library

"""
returns 1 if a and b are different, 0 if they are the same

args:
    a: character
    b: character

returns:
    int: 1 if a and b are different, 0 if they are the same
"""
def diff(a,b):
    if a == b:
        return 0
    else:
        return 1

""" 
returns the edit distance between two sequences. Edit distance represents an alignment 
between two sequences where gaps are allowed. 

args:
    seq1: sequence 1
    seq2: sequence 2

returns:
    int: edit distance between seq1 and seq2
"""
def edit_distance(seq1, seq2):
    E = np.zeros((len(seq1) + 1, len(seq2) + 1))
    for i in range(len(seq1) + 1):
        E[i, 0] = i
    for j in range(len(seq2) + 1):
        E[0, j] = j
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            E[i,j] = min(E[i-1,j] + 1, E[i,j-1] + 1, E[i-1,j-1] + diff(seq1[i-1], seq2[j-1]))
    return E[len(seq1), len(seq2)]

""" 
Generates a character alignment matrix. 

If designing for a duplex library, each entry [i,j] is the minimum among the edit 
distances:
1. between seq i and seq j 
2. between seq i and the reverse complement of seq j
3. between the reverse complement of seq i and seq j
4. between the reverse complement of seq i and the reverse complement of seq j

If designing for a single stranded library, each entry [i,j] is the edit distance between 
seq i and seq j. 

Args:
    library: list of seqs
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)
    
Returns:
    similar: NxN numpy array of edit distances
"""
def alignment_matrix(library, duplex):
    similar = np.zeros((len(library),len(library)))
    #for i in tqdm(range(len(similar))):
    for i in range(len(similar)):
        for j in range(i,len(similar)):
            if not duplex:
                val = edit_distance(library[i], library[j])
            else: 
                # get highest similarity with all 4 possible combinations of seqs i and j
                # by considering their reverse complements also
                vals = []
                vals.append(edit_distance(library[i], library[j]))
                vals.append(edit_distance(library[i], nu.reverse_complement(library[j])))
                vals.append(edit_distance(nu.reverse_complement(library[i]), library[j]))
                vals.append(edit_distance(nu.reverse_complement(library[i]), 
                                          nu.reverse_complement(library[j])))
                val = np.min(vals)
            similar[i][j] = val
            similar[j][i] = val
    
    return similar

""" 
Alignment optimization. Removes sequences from library and probability matrix that 
are too similar: ie when they are below a specified edit distance threshold, 
threshold_ALIGN.

At a high level, we find pairs of sequences that are too similar and remove the one with 
the highest similarity with some other sequence. (If equal, remove at random). We repeat 
this process until no pairs of sequences are too similar above the threshold. 

Args:
    library: list of seqs
    threshold_ALIGN: threshold for alignment
    reporting: Bool, True if you want to print out the process
    duplex: Whether we are designing for a duplex (1) or single stranded library (0)

Returns:
    library: optimized list of seqs
"""
def align_optimization(library, threshold_EDIT, reporting, duplex):
    working_matrix = alignment_matrix(library, duplex)
    n = working_matrix.shape[0]
    
    # make the diagonal infinity to ignore self-alignment
    np.fill_diagonal(working_matrix, np.inf)

    active = np.ones(n, dtype=bool)  # Keep track of active (non-removed) sequences
    
    while True:
        # Mask inactive rows/cols as infinity and ignore
        working_matrix[~active, :] = np.inf
        working_matrix[:, ~active] = np.inf
        
        min_val = np.min(working_matrix)
        if min_val >= threshold_EDIT:
            break   # loop until edit distance criterion satisfied

        i, j = np.unravel_index(np.argmin(working_matrix), working_matrix.shape)
        
        if reporting:
            print(f"{library[i]} and {library[j]} are too similar")
            print(f"Edit distance: {working_matrix[i][j]}")

        # Evaluate which to remove based on max edit distance to others
        a = np.copy(working_matrix[i])
        b = np.copy(working_matrix[j])
        a[j] = 0    # ignoring each other...
        b[i] = 0

        if np.min(a) < np.min(b):
            to_remove = i
        elif np.min(a) > np.min(b):
            to_remove = j
        else:
            to_remove = np.random.choice([i, j])  # Randomly choose if equal

        active[to_remove] = False

        if reporting:
            print(f"Removing {library[to_remove]}")
            print(f"Remaining Elements {np.sum(active)}")

    # Rebuild final filtered library
    final_library = [seq for idx, seq in enumerate(library) if active[idx]]
    return final_library

""" 
Off-target optimization. Removes sequences from library and probability matrix that
have high off-target binding probabilities, above a specified threshold.

At a high level, we find pairs of sequences that have high off-target binding 
probabilities and remove the one with the higher total off-target score. (If equal, remove
at random). We repeat this process until no pairs of sequences have higher than threshold 
off-target binding probabilities. 

Args:
    nu_mat: probability matrix
    library: list of seqs
    threshold: threshold for off-target binding probability
    reporting: Bool, True for verbose output
    on_t: on-target binding probabilities to be updated with updated library

Returns:
    library: optimized list of seqs
    nu_mat: optimized probability matrix
    on_t: optimized on-target binding probabilities
"""
def off_target_optimization(nu_mat, library, threshold, reporting, on_t):
    assert(nu_mat.shape[0] == nu_mat.shape[1])
    assert(nu_mat.shape[0] == len(library))
    n = nu_mat.shape[0]
    active = np.ones(n, dtype=bool)  # Tracks which sequences are still active
    working_matrix = np.copy(nu_mat)
    
    # first filter based off diagonal of matrix
    for i in range(n):
        active[i] = (working_matrix[i,i] <= threshold)
        if reporting and not active[i]:
            print("Excess self off-target binding detected")
            print(f"seq {i}: " + library[i] + f", p = {working_matrix[i,i]}")
        
    while True:
        # mask inactive rows/columns
        working_matrix[~active, :] = 0
        working_matrix[:, ~active] = 0

        max_i, max_j = np.unravel_index(np.argmax(working_matrix), nu_mat.shape)
        max_value = working_matrix[max_i, max_j]
        
        if max_value <= float(threshold):
            break # loop until no more values above threshold

        if reporting:
            print("Excess off-target binding detected")
            print(f"seq {max_i}, seq {max_j}, p = {max_value}")
            print(f"seq {max_i}: " + library[max_i])
            print(f"seq {max_j}: " + library[max_j])

        # decide which sequence to remove based on total off-target score
        score_i = np.sum(working_matrix[max_i, :])
        score_j = np.sum(working_matrix[max_j, :])
        # note that they share an entry so no need to subtract that
        
        if score_i > score_j:
            remove_idx = max_i
        elif score_i < score_j:
            remove_idx = max_j
        else:
            remove_idx = np.random.choice([max_i, max_j])
            
        active[remove_idx] = False

        if reporting:
            print(f"Removed seq {remove_idx}, Remaining Elements {np.sum(active)}")

    # final filtering
    final_library = [seq for i, seq in enumerate(library) if active[i]]
    final_matrix = nu_mat[np.ix_(active, active)]
    final_on_t = on_t[active]

    return final_library, final_on_t, final_matrix

""" 
On-target optimization. Removes sequences from library and probability matrix that
have low on-target binding probabilities, below a specified threshold. 

(We simply throw out sequences with low on-target binding probabilities, below a 
specified threshold in a linear fashion.)

Args:
    on_t: on-target binding probabilities
    nu_mat: probability matrix to be updated with library
    library: list of seqs
    threshold: threshold for on-target binding probability
    reporting: Bool, True for verbose output

Returns:
    library: optimized list of seqs
    on_t:    optimized list of on-target binding probabilities
    nu_mat:  optimized probability matrix
"""
def on_target_optimization(on_t, library, threshold_ON, reporting, nu_mat):
    assert(len(on_t) == len(library))
    n = len(on_t)
    active = np.ones(n, dtype=bool)  # Tracks which sequences are still active
    for i in range(n):
        active[i] = (on_t[i] >= threshold_ON)
        if reporting and not active[i]:
            print(f"seq {i} has low on-target affinity")
            print(f"seq {i}: " + library[i] + f"    p = {on_t[i]}")

    # final filtering
    final_library = [seq for i, seq in enumerate(library) if active[i]]
    final_on_t = on_t[active]
    final_matrix = nu_mat[np.ix_(active, active)]

    return final_library, final_on_t, final_matrix

""" 
For a given row index i in a library, this function computes the melting temperature
for the complex formed by the sequences at index i and all subsequent indices in
the library (building a row in the upper triangle of a melting array matrix). 

This is done by performing a binary search over a range of temperatures
to find the temperature at which the probability of formation of the complex
is 0.5. 

Because we can only do discrete temperatures/searches, we use a binary search across 
an array of temperatures represented by the range between low and high with a given step 
interval grain. 

Args:
    i: index i
    library: list of seqs
    low: lower bound of temperature range to investigate
    high: upper bound of temperature range 
    grain: temperature grain
    conc: molar concentration of strands
    
Returns:
    i: index i
    row[i:]: upper triangle slice of the melting array matrix
"""
def row_tm_worker(args):
    i, library, low, high, grain, conc = args
    size = len(library)
    temperatures = np.arange(low, high, grain)
    row = np.zeros(size)

    for j in range(i, size):
        probs = [0] * len(temperatures)
        low_index = 0
        high_index = len(temperatures) - 1
        middle_index = 0
        top_index = 0
        while low_index <= high_index:
            middle_index = low_index + (high_index - low_index) // 2
            t_mid = temperatures[middle_index]
            if probs[middle_index] == 0:
                prob_mid = np_crosstalk_tm(library[i], library[j], t_mid, conc)[0] / conc
                probs[middle_index] = prob_mid
            else:
                prob_mid = probs[middle_index]
            if prob_mid > 0.5:
                low_index = middle_index + 1
            elif prob_mid < 0.5:
                high_index = middle_index - 1
            else:
                top_index = middle_index
                break
            if abs(prob_mid - 0.5) < abs(probs[top_index] - 0.5):
                top_index = middle_index

        if probs[top_index] > 0.001: # You may choose to neglect this if/else...
            # has to be somewhat significant! if this low (<0.1%), then we haven't found a 
            # valid melting temp!
            val = temperatures[top_index]
        else:
            val = low
        row[j] = val

    return i, row[i:]  # return only upper triangle slice

""" 
Generates a matrix of melting temperatures between all pairs of sequences in a library.
Uses multiprocessing. 

Because we can only do discrete temperatures/searches, we only evaluate temperatures
within the range between low and high with a given step interval grain. 

Args:   
    library: list of seqs
    low: lower bound of temperature range to investigate
    high: upper bound of temperature range 
    grain: temperature grain
    conc: molar concentration of strands
    ncores: number of cores to use for parallel processing

Returns:
    tm_mat: matrix of melting temperatures
"""
def tm_mp(library, low, high, grain, conc, ncores):
    size = len(library)
    tasks = [(i, library, low, high, grain, conc) for i in range(size)]

    print("Generating Melting Temperature Matrix \n")
    tm_mat = np.zeros((size, size))

    with Pool(ncores) as pool:
        for i, row_slice in tqdm(pool.imap(row_tm_worker, tasks), total=size):
            tm_mat[i, i:] = row_slice
            tm_mat[i+1:, i] = row_slice[1:]  # mirror lower triangle

    return tm_mat

""" 
Generates a matrix of melting temperatures between all pairs of sequences in a library.
Does not use multiprocessing. 

Because we can only do discrete temperatures/searches, we only evaluate temperatures
within the range between low and high with a given step interval grain. 

Args:   
    library: list of seqs
    low: lower bound of temperature range to investigate
    high: upper bound of temperature range 
    grain: temperature grain
    conc: molar concentration of strands

Returns:
    tm_mat: matrix of melting temperatures
"""
def tm_no_mp(library, low, high, grain, conc):
    size = len(library)
    tm_mat = np.zeros((size, size))
    for i in range(size):
        i, row_slice = row_tm_worker((i, library, low, high, grain, conc))
        tm_mat[i, i:] = row_slice
        tm_mat[i+1:, i] = row_slice[1:]  # mirror lower triangle

    return tm_mat

'''
Helper function for tm_optimization. 

This function removes sequences from the library that have off-target melting temperatures 
above the threshold min(tm_on) - delta. When given a pair of sequences with a higher than 
threshold melting temperature, it picks the sequence to be removed as the one with the 
higher total off-target score across the entire tm_off matrix. (If scores are equal, 
there is a 50% chance of removing either sequence).

args:
    tm_on: list of melting temperatures for on-target sequences
    tm_off: n by n matrix of melting temperatures for off-target sequences
    delta: minimum accepted difference between on-target and off-target melting 
            temperatures in optimized library

returns: 
    active_1: list of booleans indicating which sequences are still active
'''
def tm_optimization_helper(tm_on, tm_off, delta):
    threshold_1 = np.min(tm_on) - delta 
    active_1 = np.ones(len(tm_on), dtype=bool)
    working_tm_off = np.copy(tm_off)
    
    if np.min(tm_off) > threshold_1: # edge case where all off-targets are above threshold
        return np.zeros(len(tm_on), dtype=bool)
    
    # first filter based off diagonal of matrix
    for i in range(working_tm_off.shape[0]):
        active_1[i] = (working_tm_off[i,i] <= threshold_1)        
    
    while np.max(working_tm_off) > threshold_1:
        working_tm_off[~active_1, :] = 0
        working_tm_off[:, ~active_1] = 0

        max_i, max_j = np.unravel_index(np.argmax(working_tm_off), tm_off.shape)
        max_value = working_tm_off[max_i, max_j]
        
        if max_value <= float(threshold_1):
            break
        
        score_i = np.sum(working_tm_off[max_i, :])
        score_j = np.sum(working_tm_off[max_j, :])
        
        if score_i > score_j:
            remove_idx = max_i
        elif score_i < score_j:
            remove_idx = max_j
        else:
            remove_idx = np.random.choice([max_i, max_j])
                        
        active_1[remove_idx] = False

    return active_1

'''
Melting temperature optimization. Removes sequences from library that have strong 
off-target interactions with other sequences. ie such that all sequences returned 
satisfy the condition max(tm_off) - min(tm_on) <= delta.

At a high level, we iteratively check if removing the worst on-target sequence improves 
the number of sequences that satisfy the delta constraint, as compared to just removing 
the off-target sequences which fail the delta constraint. We pick the greedy strategy 
(remove off-targets or worst on-target) that lets us keep the most amount of sequences 
iteratively. 

We stop once the constraint is wholly satisfied. 

args:
    library: list of seqs
    tm_mat: matrix of melting temperatures
            (assumed to be 2n by 2n where n is the number of strands ignoring complements)
            (this means i and i+1 for i%2=0 are on-targets)
    delta: minimum accepted difference between on-target and off-target melting 
            temperatures in optimized library
    reporting: Bool, True for verbose output

returns:
    new_library: list of seqs that satisfy the delta constraint
'''
def tm_optimization(library, tm_mat, delta, reporting):
    # Extract on-target values and indices and save as an array
    on_indices = [(i, i+1) for i in range(0, len(tm_mat) - 1, 2)]
    tm_on = [tm_mat[i, j] for i, j in on_indices]
    
    # create an off target tm_mat matrix which only includes the maximum melting 
    # temperature of off targets between a strand & its complement number i and 
    # a strand & its complement number j
    tm_off = np.zeros((len(tm_mat)//2,len(tm_mat)//2))
    for i in range(len(tm_off)):
        tm_off[i][i] = np.max([tm_mat[2*i][2*i],tm_mat[2*i+1][2*i+1]])
        for j in range(i+1, len(tm_off)):
            tm_off[i][j] = np.max([tm_mat[2*i][2*j],   tm_mat[2*i+1][2*j],
                                  tm_mat[2*i][2*j+1], tm_mat[2*i+1][2*j+1]])
            tm_off[j][i] = tm_off[i][j] # symmetry!
    
    assert(len(tm_off) == len(tm_on))
    
    active_1 = tm_optimization_helper(tm_on, tm_off, delta)
    working_tm_off = np.copy(tm_off)
    working_tm_on = np.copy(tm_on)
    
    print("Melting temperature optimization in progress...\n")
    
    while True:
        # check if removing the worst on target improves the number of leftover sequences
        # and if so, remove it
        worst_on_target = np.argmin(working_tm_on)
        working_tm_on[worst_on_target] = np.max(working_tm_on) + 1 # ignore this one
        working_tm_off[worst_on_target, :] = 0
        working_tm_off[:, worst_on_target] = 0
        
        active_2 = tm_optimization_helper(working_tm_on, working_tm_off, delta)
        active_2[worst_on_target] = False
        if np.sum(active_2) <= np.sum(active_1):
            break; # it's no better! so keep the worst on-target
        else: # it's better! maybe removing more will be good
            active_1 = np.copy(active_2)
            print(np.sum(active_2))
    
    # we now have a large set of sequences that satisfy our delta constraint
    print("Melting temperature optimization complete...\n")
    
    if reporting:
        print(f"Number of sequences satisfying delta constraint: {np.sum(active_1)}\n")
        
    new_library = [seq for i, seq in enumerate(library) if active_1[i]]
    return new_library
    
'''
Melting temperature optimization which ensures that on-target structures fall within 
a specified maximum range. This function will find the best range of melting 
temperatures that maximizes number of on target structures given a specified my_range 
between a lower and upper melting temperaturebound. 

args:
    library: list of seqs
    tm_mat: matrix of melting temperatures (2n by 2n)
    my_range: desired range (NB 0 means no range restriction, returns library as is)
    reporting: Bool, True for verbose output

returns:
    best_library: list of seqs within the best range
    best_range: tuple of lower and upper melting temperature bounds
'''
def tm_bounds_optimization(library, tm_mat, my_range, reporting):
    if my_range == 0:
        return library, (0, 0)
    # extract on targets and sort
    on_indices = [(i, i+1) for i in range(0, len(tm_mat) - 1, 2)]
    tm_on = [tm_mat[i, j] for i, j in on_indices]
    sorted_tm = sorted(tm_on)
    sorted_tm = list(dict.fromkeys(sorted_tm)) # remove duplicates
    
    # search for best range 
    best_range = (0, 0)
    max_count = 0
    best_library = []
    best_indices = []
    for i in tqdm(range(len(sorted_tm))):
        min_tm = sorted_tm[i] # possible lower bounds
        max_tm = min_tm + my_range

        valid_indices = [on_indices[idx][0] // 2 for idx, tm in enumerate(tm_on) 
                         if min_tm <= tm <= max_tm]
        count = len(valid_indices)

        if count > max_count:
            max_count = count
            best_range = (min_tm, max_tm)
            best_indices = valid_indices
        
        if max_tm > sorted_tm[-1] or max_count == len(library):
            break # no point checking further

    print(f"\nOptimal Tm range: {best_range} °C")
    if reporting:
        print(f"Number of sequences within optimal range: {max_count}\n")

    best_library = [library[idx] for idx in best_indices]
    
    return best_library, best_range

# User helper functions
'''
Saves a library to a file.

args:
    library: list of seqs
    library_file: name of file to save library to
'''
def save_lib(library, library_file):
    print("Writing to file: " + library_file +"\n")
    resultFyle = open(library_file,'w')
    for r in library:
        resultFyle.write(r + "\n")
    resultFyle.close()

'''
Prints the size of the library.

args:
    library: list of seqs
'''
def print_library_size(library):
    print(f"\nCurrent library size: {len(library)}\n")