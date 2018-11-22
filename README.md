# Hydraville!

This repository contains a complete example Pywr model. The fictional city of 
Hydraville is located in a river basin within a river basin with significant 
water resource challenges. A Pywr model has been built to aid the management
and planning of those challenges. 

This model serves as a training and tutorial example for water resource planning
and decision making under uncertainty. 

## Overview


## Installation


## Obtaining the data

INSERT INSTRUCTIONS TO DOWNLOAD PREPROCESSED DATA FROM S3

## Basic usage

The basic usage of this repository requires one to first generate a Pywr
 input file. The repository allows many configurations of the same core
 models to be generated. To begin we create the most basic water only
 model derived from the "water-simple" configuration.
 
***We recommend the user create an analysis subfolder to save the 
generated models and their outputs in.***

```bash
hydraville create analysis/water-only.json --model water-simple
```

Once the input file is generated we can run the model:

```bash
hydraville run analysis/water-only.json
```



### UKCP09 data pre-processing






## Importing in to Hydra

 
 
 
## Data processing

This repository also contains code for pre-processing and sub-sampling UKCP09 weather
 data, and subsequently running that data through a series of hydrological models 
 written in Catchmod (using `pycatchmod`). 
 
### UKCP09 Pre-processing

The output of a UKCP09 Weather Generator run is assumed to be saved in the `data/` 
 directory. This output should consist of a series of ZIP archives containing CSV
 files. The following command will pre-process that data by exporting it in to a
 single HDF5 file.   
 
```bash
hydraville ukcp09 preprocess data/ data/ukcp09_full.h5
``` 

### UKCP09 Sub-sampling

The full output of a UKCP09 Weather Generator run contains many scenarios. For our 
 modelling purposes we may want to sub-sample this set to a more managable number.
 The following command will sub-sample the previously created combined archive. 
 Each sub-sample is saved in a separate file with a suffix (e.g. `data/ukcp09_sub010.h5`)
 
```bash
hydraville ukcp09 subsample data/ukcp09_full.h5 data/ukcp09.h5
```

### Running catchmod

Once the sub-sampled data has been created. The hydrological flows can be generated
 using Catchmod. The catchmod output is saved in to formats. The first is with a 
 dataframe for each of the UKCP09 files (analagous to the UKCP09 files created above).
 The second groups all the flows for each of the catchmod models into a single dataframe
 with a column for each UKCP09 scenario.
 
***Please note that running catchmod takes a long time for all scenarios.***
 
```bash
hydraville catchmod run data/ukcp09.h5 data/catchmod_flows.h5 data/catchmod_flows_by_catchment.h5
``` 
