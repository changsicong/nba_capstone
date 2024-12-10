# Project Repository

This repository contains all the scripts and files used to fetch, process, model, and evaluate basketball possession-level data. Each file has a specific role in the pipeline, as described below:

- **generate_dataset.py**  
  Requests raw play-by-play (PBP) data from online sources and processes it into a structured, usable format. The output includes cleaned and aggregated possession-level data suitable for further analysis.

- **split_data.py**  
  Takes the processed dataset from `generate_dataset.py` and performs additional data cleaning. It then splits the dataset into training, validation, and test subsets, ensuring a reliable pipeline for model development and evaluation.

- **first_model.py**  
  Defines and trains the first proposed model architecture. This script includes code for loading the prepared dataset, building the model, and iteratively refining its parameters through training.

- **second_model.py**  
  Defines and trains the second model architecture, which typically incorporates richer feature sets or more complex modeling techniques. Similar to `first_model.py`, it handles data loading, model definition, and training routines.

- **evaluation.py**  
  Evaluates the trained models and compares their predictions against real-world game results. It also allows for comparison with various all-in-one basketball metrics, producing performance statistics that highlight the effectiveness of each approach.

- **evaluation.py**
  Notebooks that do experiments.
