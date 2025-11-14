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

In addition to native Python libraries, we are using some additional Python libraries, so please make sure that you got the following libraries installed: `pandas` and `numpy`.

You can check if you have them installed or not by typing the following command line into your terminal:
```python
pip install pandas numpy
```
Or you can use the __requirements.txt__ file:
```python
pip install -r requirements.txt
```
If you don't have the mentioned libraries installed, the system will install them for you.

1. Run all of the Notebook Cells
   
2. Inspect the Output
Results are printed directly in the notebook. CSV files containing generalized data are saved in the working directory.

## Analysis

1. Generalization level (Binning of age and hours variables)

The generalization level controls how finely or coarsely numerical attributes (such as age and hours worked per week) are grouped before being passed to the DP-ID3 algorithm. Lower levels (e.g., raw values or 5-year bins) preserve more detail, while higher levels (e.g., broad groups such as $≤25$, $26–40$, $41–60$, $>60$) merge many original values together. From a privacy perspective, stronger generalization increases k-anonymity and reduces the risk of re-identification in the published model, because individual records become harder to distinguish from one another. However, the trade-off is that accuracy decreases as the bins become too coarse: the tree loses the ability to use fine-grained distinctions in the data, causing splits to become less informative and potentially shifting decision boundaries. In terms of efficiency, coarser bins significantly reduce the number of branches that the DP-ID3 tree needs to consider, which lowers the number of DP counting queries and accelerates tree construction. Fewer categories also reduce the cumulative effect of DP noise. Thus, the generalization level directly influences the privacy–utility–efficiency balance: more generalization increases privacy and speed while sacrificing predictive accuracy, whereas finer granularity has the opposite effect.

2. Privacy budget (epsilon) in `find_entropy_split`

The privacy budget $\varepsilon$ controls the amount of Laplace noise added to all label and branch counts during the split selection process. A smaller $\varepsilon$ introduces more noise, which strengthens privacy by making it harder to infer the presence or absence of any individual record from the statistics used in the decision tree. However, high noise also corrupts the computed entropy values and may cause the algorithm to select suboptimal splits, thereby reducing model accuracy. Conversely, a larger $\varepsilon$ yields more accurate counts and produces a tree closer to the non-private ID3 baseline, but at the cost of weaker privacy guarantees. From an efficiency standpoint, $\varepsilon$ does not change computational complexity directly, but indirectly affects efficiency through tree behavior: with high noise (small $\varepsilon$), split quality becomes unreliable, causing the algorithm to terminate early, produce shallower trees, or generate fewer meaningful branches. This sometimes speeds up training, but the resulting model may be less useful. Larger $\varepsilon$  improves split stability, often leading to deeper trees and more computation. Overall, $\varepsilon$ represents the central privacy–accuracy trade-off in differential privacy: lower $\varepsilon$  gives stronger privacy but poorer accuracy and more unstable splits; higher $\varepsilon$ improves utility but weakens privacy protection.

Inside the `find_entropy_split` routine, the algorithm issues several noisy counting queries, so we must account for how much privacy budget is consumed inside each call. For each attribute value ($j$), the function first releases a noisy version of the branch size ($|D_{a=j}|$), and then releases a set of noisy class-specific counts ($|D_{a=j, y=i}|$). The class-specific subsets (${D_{a=j, y=i}}*i$) form a partition of ($D*{a=j}$), so by **parallel composition**, all class counts together consume only **$\varepsilon$** (not $\varepsilon$ multiplied by the number of classes). However, the noisy total ($|D_{a=j}|$) is another query on the same records. Therefore, by **sequential composition**, the combined privacy cost of one entropy computation for a single attribute value is:

$
2\varepsilon_{\text{local}} = \varepsilon_{\text{total-count}} + \varepsilon_{\text{class-counts}}.
$

Across different attribute values ($j$), the groups (${D_{a=j}}$) form a partition of the dataset by that attribute, so those iterations compose in parallel and do **not** further accumulate privacy loss. To ensure the overall call to `find_entropy_split` remains within a node-level budget ($\varepsilon_{\text{node}}$), we therefore allocate:

$
\varepsilon_{\text{local}} = \frac{\varepsilon_{\text{node}}}{2 * |Unique Labels|},
$

so that the total consumption per split evaluation is at most $varepsilon_{\text{node}}$. This also means the cost does **not** grow with the number of class labels, preserving both correctness and tight privacy accounting for the entire DP-ID3 algorithm.