# Analysis Scripts

This directory contains analysis scripts for computing structural observables, NOE-derived distances, and REST2 exchange statistics from trajectories.

## Scripts

### `alanine_polyproline.py`
Computes **Ramachandran \(\phi/\psi\) angles** and saves them to CSV. It supports both:
- **REST2 trajectories** loaded from segmented HDF5 files
- **plain MD trajectories** loaded from segmented DCD files 

### `cln_chignolin.py`
Analyzes **CLN025 / chignolin** trajectories for the all replicas and outputs:
- **CA RMSD**
- **RMSDdist** (intramolecular distance RMSD)
- **radius of gyration**  
Results are written to a CSV file for downstream analysis. 

### `cln_noe_folded.py`
Computes **r⁻⁶-averaged NOE distances** for **CLN025** from REST2 trajectories (unscaled replica) using the **filtered NOE table**. Results are grouped by selected NOE IDs and written to CSV. 

### `cln_noe_unfolded.py`
Computes **r⁻⁶-averaged inter-residue NOE distances** for **CLN025** from REST2 trajectories using the **unfiltered NOE table**.  
All inter-residue NOEs are computed first; additional exclusions are applied later during plotting. 

### `cyclic_peptides.py`
Analyzes the cyclic peptides **cis-cyclic peptide** and **trans-cyclic peptide**. The script:
- computes **cyclic \(\phi/\psi\) dihedral angles**
- computes **NOE r⁻⁶-averaged distances**
- processes both **REST2** and **production** trajectories
- writes the resulting tables to CSV files 

### `roundtrips_exchange_acceptance.py`
Parses **REST2 log files** to evaluate replica-exchange performance. It can compute:
- **global acceptance rate**
- **acceptance rate for each replica pair**
- **roundtrip counts per walker**  
It can also optionally generate plots and export summary files in CSV or JSON format. 

### `trp_noe.py`
Computes **grouped r⁻⁶-averaged NOE distances** for **trp-cage** trajectories from REST2 HDF5 segments. It reads NOE definitions from CSV, averages distances by group, and reports restraint violations. 

### `trp_structural_analysis.py`
Performs structural analysis for **trp-cage (folded/unfolded)** trajectories. It loads either **plain MD** or **REST2** trajectories and computes **CA RMSD** relative to the reference trp-cage structure, saving the result as a NumPy array. 
