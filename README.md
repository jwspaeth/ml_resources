# ml_resources
Some basic resources for doing modern machine learning.

**Author:** Will Spaeth

# Basic Usage
• For running a single configuration file sequentially: ./main.py -cfg_name=<config_file_name>  
• Add -p argument for running hyperparameter tests in parallel (listed at bottom of config file)  
• Add -s argument for running on a supercomputer using SLURM system  
• For new users using existing datasets, be sure to change the file path to the data in the dataset class  
  
# Configuration
• Execution is defined by two different modes, "train" and "eval". These are set at the top of each configuration file. In addition, "train" has an extra ability to perform grid search hyperparameter testing over lists of parameters.  
• "train": Trains a model in the standard sense. Loads datasets and models and fits to data. Creates results directory and saves results and logs to this directory. Parameters are defined within config file.  
• "eval": Revives an already trained model from a results directory, applies evaluation functions to this revived model, and saves to original directory. Parameters are defined within config file.  
• Grid search hyperparameter testing: This is a available within the "train" mode. At the bottom of the config file, an option dictionary should be present. Each entry in the dictionary can be a list of parameters for one key in the configuration. If values exist in the dictionary, they override the value in the respective configuration key. The system then runs one training run for each hyperparameter combination, which results in a combinatorial number of experiments (# param 1 * # param 2 * ... * # param n). The system handles saving all of these results into their respective directories. If the dictionary is empty, it is ignored.  
