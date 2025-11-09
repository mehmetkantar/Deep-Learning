# COMP541 Assignment 1: MLPs and Backpropagation

## Overview
This repository contains Assignment 1 for COMP541: Deep Learning Course, focusing on Multi-Layer Perceptrons (MLPs) and Backpropagation implementation. The assignment explores word embeddings, text classification, and neural network fundamentals.

## Project Structure
```
Assignment 1/
├── mehmetkantar_assignment1.ipynb    # Main assignment notebook
├── env.yml                            # Conda environment configuration
├── Assignment_1.pdf                   # Assignment instructions
├── data/                              # Dataset directory
├── imgs/                              # Images and visualizations
├── best_model_*.pt                    # Trained model checkpoints
│   ├── best_model_CoOccurrence_1layers.pt
│   ├── best_model_GloVe_1layers.pt
│   ├── best_model_GloVe_2layers.pt
│   ├── best_model_GloVe_3layers.pt
│   ├── best_model_GloVe-Twitter_1layers.pt
│   ├── best_model_GloVe-Twitter_2layers.pt
│   └── best_model_GloVe-Twitter_3layers.pt
└── README.md                          # This file
```

## Assignment Components

### 1. Word Embeddings
- Implementation and analysis of different word embedding techniques
- **Co-Occurrence Matrix**: Basic word representation based on word co-occurrence statistics
- **GloVe (Global Vectors)**: Pre-trained word embeddings on Common Crawl corpus
- **GloVe-Twitter**: Domain-specific embeddings trained on Twitter data

### 2. Neural Network Architecture
- Multi-layer perceptron implementation from scratch
- Experiments with different network depths (1, 2, and 3 layers)
- Backpropagation algorithm implementation
- Gradient descent optimization

### 3. Text Classification
- Sentiment analysis and document classification tasks
- Performance comparison across different embedding methods
- Model evaluation and hyperparameter tuning

## Environment Setup

### Prerequisites
- Python 3.7
- Anaconda or Miniconda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mehmetkantar/Deep-Learning.git
cd Deep-Learning/Assignment1
```

2. Create the conda environment using the provided `env.yml`:
```bash
conda env create -f env.yml
```

3. Activate the environment:
```bash
conda activate comp541
```

### Dependencies
The environment includes:
- **Python 3.7**: Core programming language
- **Jupyter**: Interactive notebook environment
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning utilities
- **NLTK**: Natural language processing toolkit
- **Gensim 3.8.3**: Word embedding models and topic modeling

## Usage

### Running the Notebook

1. Activate the conda environment:
```bash
conda activate comp541
```

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `mehmetkantar_assignment1.ipynb` in your browser

### Trained Models

The repository includes pre-trained models for different configurations:
- **1-layer networks**: Baseline shallow models
- **2-layer networks**: Intermediate depth models
- **3-layer networks**: Deeper architectures

Each model is trained with different embedding strategies (Co-Occurrence, GloVe, GloVe-Twitter) to compare performance.

## Key Features

### Word Embedding Analysis
- Semantic similarity computation
- Word analogy tasks (e.g., king - man + woman = queen)
- Visualization of word relationships

### MLP Implementation
- Forward propagation
- Backward propagation with gradient computation
- Weight initialization strategies
- Activation functions (ReLU, Sigmoid, etc.)
- Loss functions (Cross-entropy, MSE)

### Performance Metrics
- Training and validation accuracy
- Loss curves and convergence analysis
- Comparison across embedding methods and network depths

## Results

The trained models demonstrate:
- Impact of embedding quality on classification performance
- Trade-offs between model depth and generalization
- Effectiveness of different optimization strategies

Detailed results and visualizations are available in the notebook.

## Assignment Details
- **Course**: COMP541 - Deep Learning
- **Due Date**: November 9th, 2025 (23:59:59)
- **Student**: Mehmet Kantar

## Notes
- Ensure all dependencies are properly installed before running the notebook
- The `data/` directory should contain the required datasets (Reuters corpus via NLTK)
- Model checkpoints can be loaded for inference without retraining

## License
This is an academic assignment for educational purposes.

## Contact
For questions or issues, please contact the course instructor or refer to the assignment documentation.

---

**Last Updated**: November 2025
