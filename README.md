# CS 323 - Project 6: Privacy-Preserving Machine Learning

Authors: Bach Nguyen, Son Nguyen

This project implements a **Differentially Private ID3 Decision Tree** algorithm and applies it to the UCI Adult Census Income dataset to predict income levels while preserving privacy through differential privacy guarantees.

## Project Overview

This project explores the application of differentially private machine learning by implementing a privacy-preserving ID3 decision tree algorithm. The algorithm uses Laplace noise mechanisms to ensure differential privacy while training a decision tree classifier. We evaluate the trade-offs between privacy protection, model accuracy, and computational efficiency.

## Data Source

The dataset used in this project is the **UCI Adult dataset**, also known as the "Census Income" dataset, which is publicly available from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/adult

The dataset contains demographic information extracted from the 1994 U.S. Census, including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country, and income (the target variable for classification).

**Dataset size**: 32,561 instances after removing missing values (original dataset has 48,842 instances)

**Target variable**: `income` (binary classification: `<=50K` or `>50K`)

## Project Structure

The project consists of the following files:

- **`cs323_project_6.ipynb`**: Main Jupyter Notebook containing data loading, preprocessing, model training, and evaluation
- **`dp_id3.py`**: Implementation of the Differentially Private ID3 Decision Tree algorithm
- **`requirements.txt`**: Python package dependencies
- **`README.md`**: This file

In the `data/` folder:

- **`adult.data`**: Main data file (CSV format without headers)
- **`adult.names`**: Metadata file containing attribute descriptions

## How to Run the Code

### Prerequisites

1. **Library Installation**

   Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:

   ```bash
   pip install pandas numpy scikit-learn
   ```

   Required libraries:

   - `pandas`: Data manipulation and analysis
   - `numpy`: Numerical computations
   - `scikit-learn`: Train-test split functionality

### Running the Code

1. **Dataset Setup**

   - Ensure the dataset files (`adult.data` and `adult.names`) are located in the `data/` folder
   - The notebook automatically loads and processes the data

2. **Execute the Notebook**

   - Open `cs323_project_6.ipynb` in Jupyter Notebook or JupyterLab
   - Run all cells sequentially (Cell 0 through Cell 15)
   - The notebook will:
     - Load and preprocess the data
     - Split into training (80%) and testing (20%) sets
     - Train the differentially private decision tree with default parameters
     - Test multiple parameter combinations and compare results
     - Generate visualizations showing accuracy vs privacy trade-offs

3. **View Results**
   - Model accuracy for the default configuration is printed in Cell 10
   - Comprehensive parameter testing results are displayed in Cells 12-14
   - Visualizations comparing different parameter settings are shown in Cell 15
   - Training warnings about numerical stability (log2) are normal and handled internally

### Running with Alternative Parameters

You can modify the following parameters in **Cell 8** of the notebook:

```python
# Default parameters
tree = DPDecisionTree(epsilon1=1.0, max_depth=6)
```

**Testing parameter combinations with experimental results:**

The notebook includes a comprehensive parameter testing section (Cells 11-15) that evaluates multiple configurations. Below are the tested combinations with their **actual accuracy results** on the Adult Census Income dataset:

1. **Very High Privacy (Strict):**

   ```python
   tree = DPDecisionTree(epsilon1=0.01, max_depth=3)
   ```

   - **Accuracy: 73.02%** - Strongest privacy protection but lowest accuracy due to high noise

2. **High Privacy:**

   ```python
   tree = DPDecisionTree(epsilon1=0.1, max_depth=4)
   ```

   - **Accuracy: 78.69%** - Good privacy with moderate accuracy degradation

3. **Moderate Privacy:**

   ```python
   tree = DPDecisionTree(epsilon1=0.5, max_depth=5)
   ```

   - **Accuracy: 81.54%** - Balanced privacy-accuracy trade-off

4. **Balanced (Default):**

   ```python
   tree = DPDecisionTree(epsilon1=1.0, max_depth=6)
   ```

   - **Accuracy: 81.46%** - Default configuration providing reasonable privacy with good accuracy

5. **Lower Privacy, Higher Accuracy:**
   ```python
   tree = DPDecisionTree(epsilon1=5.0, max_depth=8)
   ```
   - **Accuracy: 83.86%** - Weaker privacy protection but highest accuracy, approaching non-private performance

**Summary of Results:**

- Privacy budget (`epsilon1`) shows a clear accuracy trade-off: increasing from 0.01 to 5.0 improves accuracy from 73.02% to 83.86% (a 10.84 percentage point improvement)
- The relationship is not perfectly linear, with diminishing returns at higher epsilon values
- Deeper trees (combined with higher epsilon) can capture more patterns, as seen in the best-performing configuration

**Alternative datasets:**

To use a different dataset, modify **Cell 3**:

- Update the file path in `data = os.path.join(datadir, "adult.data")`
- Update column names in `df.columns = [...]`
- Ensure the target column is specified correctly in the training cell

## Parameters and Their Impact

The decision tree has two main parameters that affect privacy, accuracy, and speed:

### 1. `epsilon1` (Privacy Budget)

- **What it is**: A positive number (default: 1.0) that controls how much noise is added to protect privacy
- **Privacy**: Smaller values (like 0.1) add more noise and protect privacy better. Larger values (like 5.0) add less noise but weaken privacy protection.
- **Accuracy**: Smaller values give lower accuracy because the noise makes it harder to find good splits. Larger values give higher accuracy that's closer to non-private models.
- **Speed**: Very small values can make training slightly slower because the algorithm needs to work harder to find meaningful splits.

### 2. `max_depth` (Maximum Tree Depth)

- **What it is**: A whole number (default: 5) that controls how deep the tree can grow
- **Privacy**: Deeper trees ask more questions about the data, which can reveal more information. Shallower trees are more private.
- **Accuracy**: Deeper trees can learn more complex patterns but may overfit. However, with high noise (low epsilon), deeper trees become unreliable, so there's a sweet spot around depth 4-8.
- **Speed**: Deeper trees take longer to train because they have more nodes. Depth 3-5 trains quickly, while depth 8+ can be slow.

### Experimental Results

We tested different combinations on the Adult Census Income dataset and found clear trade-offs:

- **Very strict privacy** (epsilon=0.01, depth=3): **73.02% accuracy** - Strongest privacy but lowest accuracy
- **High privacy** (epsilon=0.1, depth=4): **78.69% accuracy** - Good privacy with moderate accuracy loss
- **Moderate privacy** (epsilon=0.5, depth=5): **81.54% accuracy** - Balanced option
- **Balanced** (epsilon=1.0, depth=6): **81.46% accuracy** - Default, good starting point
- **Lower privacy** (epsilon=5.0, depth=8): **83.86% accuracy** - Highest accuracy, near non-private performance

**Key findings**: The relationship between privacy and accuracy isn't linear. Increasing epsilon from 0.01 to 0.1 improves accuracy by 5.67 percentage points, but going from 1.0 to 5.0 only helps slightly. For sensitive data, use epsilon 0.1-0.5 (expect 3-8% accuracy drop). For less sensitive data, epsilon 1.0-5.0 keeps accuracy within 1-2% of non-private models. We recommend starting with epsilon=1.0 and max_depth=6, then adjust based on your privacy needs.
