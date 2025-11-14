# CS 323 - Project 6: Privacy-Preserving Machine Learning

Author: Bach Nguyen, Son Nguyen

This contains the code and analysis for **Project 6: Privacy-Preserving Machine Learning** for the course CS 323: Data Privacy

## Project Overview

This project implements a k-anonymization pipeline on the UCI Adult dataset.
We developed functions to perform attribute generalization, enforce **$k$-anonymity** and **$l$-diversity**, and compute utility metrics using **KL-Divergence** to evaluate the trade-off between privacy and data usability.

## Data Source

The dataset used in this project is the UCI Adult dataset, also known as the “Census Income” dataset, which is publicly available from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/adult

The dataset contains demographic information extracted from the 1994 U.S. Census, including age, workclass, education, marital status, occupation, race, sex, native country, and income.

Size: 48,842 instances (with 14 attributes, including the target variable income).

## Project Structure 
The project consists of 4 files (including this README):
- __cs323_project_6.ipnyb__: The main Jupyter Notebook that contains all of the code needed to run this project.

In the __data__ folder:
- __adult.data__: Main datafile
- __adult.names__: File containing metadata about the dataset

Additional Files:
- __requirements.txt__: Text file containing names of non-native Python libraries needed to run the code

## How to run the Code

1. Dataset Setup
The dataset files (adult.data and adult.names) must be placed in the __data__ folder. Note: The main .data file does not contain column names. The .names file is parsed inside the Jupyter notebook to extract and assign column names automatically.

2. Library Installation

In addition to native Python libraries, we are using some additional Python libraries, so please make sure that you got the following libraries installed: **pandas**, **numpy**, and **pycanon**

You can check if you have them installed or not by typing the following command line into your terminal:
```python
pip install pandas numpy pycanon
```
Or you can use the __requirements.txt__ file:
```python
pip install -r requirements.txt
```
If you don't have the mentioned libraries installed, the system will install them for you.

3. Run all of the Notebook Cells
   
4. Inspect the Output
Results are printed directly in the notebook. CSV files containing generalized data are saved in the working directory.

## Major Design Decisions

- **Dynamic Generalization Functions**  
  Generalization for each quasi-identifier is implemented as a function. For numerical attributes like age, bins are computed dynamically from the dataset’s min/max values, ensuring the method adapts automatically to different datasets. For categorical attributes, hierarchies are encoded in nested levels (e.g., collapsing job categories or education levels).

- **Per-Attribute Multi-Level Control**  
  Each QID supports multiple generalization levels, with mappings defined explicitly in the code. This allows fine-grained control: one attribute can be generalized heavily (e.g., “Any” country), while another remains at a low level (e.g., detailed marital status).

- **Reverse Mapping for Utility Evaluation**  
  To evaluate KL-divergence, we built reverse maps. Each generalized value is mapped back to its possible original values, distributing probability mass uniformly. 