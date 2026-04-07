## Configuration Files

This directory contains example configuration files used for the systems simulated in this work.

> NOTE: These files are provided for reference only. When running simulations using the repository, configuration files are generated automatically and do not need to be created manually.

### File Structure

For each system, the following configuration files are included:

- **`nvt/config.yaml`**  
  Configuration for the NVT equilibration stage.

- **`npt/config.yaml`**  
  Configuration for the NPT equilibration stage.

- **`rest2/config.yaml`**  
  Configuration for the REST2-AMP/MM simulation (first step).

### Additional Simulations

For systems that were also simulated using cAMP/MM, an additional configuration is provided:

- **`production/config.yaml`**  
  Configuration for the cAMP/MM simulation (first step). 
