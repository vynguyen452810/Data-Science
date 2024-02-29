# COVID-19 Drug Repurposing
## Problem
Amidst the COVID-19, there's pressing need to identify effective treatments. Traditionally, the drug discovery process is expensive and time-consuming, researcher look toward drug repurposing, a strategy that involves utilizing known drugs as candidates for treatment 
* Benefits of drug repurposing:
    * Have established information on safety, side effects, and optimal dosage 
    * Reduce development cost
    * Less of time-consuming
    * Allow for skipping certain stage in drug development pipeline
    * Expedite FDA regulatory approval process
## Scope of Reproducibility
Construct and apply Graph Neural Network (GNN)  on various data sources to demonstrate the multimodal and complex relationship among genes, drugs, baits, phenotypes and to identify the most promising drugs for potential COVID-19 treatment

The study replicates the experiement and results from "Hsieh, K., Wang, Y., Chen, L., Zhao, Z., Savitz, S., Jiang, X., Tang, J., & Kim, Y. (2020). Drug Repurposing for COVID-19 using Graph Neural Network with Genetic, Mechanistic, and Epidemiological Validation. Research square, rs.3.rs-114758. https://doi.org/10.21203/rs.3.rs-114758/v1"

## Generalized WorkFlow:
![Work Flow](images/workflow.png)

## Model Description and Implementation
### Knowledge Graph
* Input: graph data embedded in DRKG 
    * Node types: gene, drug, bait, phenotype
    * Edge types: gene-gene, gene-drug, bait-gene, phenotype-gene, phenotype-drug
* Model structure:
    * Encoder-GCN:
        * Input Layer (1x400) -> Graph Convolution (1x256) -> ReLu -> Concatenation of Edge Types -> Batch Normalization (Size 1280) -> Final Graph Convolution (1x256)
    * Decoder-GCN
        * Uses a linear layer to help reconstruct the original input (original graph) from the latent space representation 
    * Autoencoder - VGAE
        * Wraps GCN models, able to leverage the ability of both the GCN-encoder and GCN-decoder for graph-structured data
    * Customized GCN-Autoencoder (for comparision wiht VGAE)
        * Constructed by applying reparameterization from the latent space and using a separate decoder for reconstructing the input features graph convolution
    * Optimization 
        * Adam optimizer and learning rate = 0.01
### Ranking Models
* Use a custom neural network with Bayesian Pairwise Ranking (BPR) loss
* Input:
    * Knowledge graph generated earlier that is splitted into positive and negative train/validation/test edges 
* Model structure:
    * First fully connected layer (size=128)
    * ReLU → Dropout  → Batch Normalization (size=128)
    * Residual added
    * Output layer (size=1)
* BPR loss:
    * Subtract between positive edge and negative edges
    * Log sigmoid activation
    * BPR loss  = negative mean of this this output
### Comparing Ranking Model Classifiers
* Neural network models:
    * Long Short Term-Memory (LSTM)
    * Multilayer Perceptron (MLP)
    * Transformer
* Classification models:
    * Support Vector Machine (SVM)
    * K-Nearest Neighbor (KNN)
# Training and Reuslts
* Knowledge Graph
    * (Average) AUROC=0.651, AUPRC=0.576
* Ranking Models:
    * (Average) AUROC=0.7344, AUPRC=0.6917
* After evaluation, returned 47 candidate COVID-19 drugs
    * Excluding those are under trial or used in the model training, narrowed down to 16 candidate drugs 
    * Emodin, Choline, and Active  Hexose Correlated Compound (AHCC) ranked the highest, and these also have literature shoring drug efficacy against COVID-19 
    * Result differs from the original paper. Out of the 10 drugs

# Usage
## Installation
To use this project, you'll need to have the following libraries installed:
    - Pandas
    - Numpy
    - Scikit-learn (sklearn)
    - Pytorch
You can install these libraries using pip, Python's package manager. Here's the command to install them:
    !pip install pandas numpy scikit-learn torch

## Preparing the datasets
* Preparing the CTDbase files:
    * Download the files from their respective links, renaming the files accordingly:
        * chemicals.csv: https://ctdbase.org/detail.go?type=disease&acc=MESH%3AD000086382&view=chem 
        * genes.csv: https://ctdbase.org/detail.go?type=disease&acc=MESH%3AD000086382&view=gene 
        * pathways.csv: https://ctdbase.org/detail.go?type=disease&acc=MESH%3AD000086382&view=pathway
        * phenotype.csv: https://ctdbase.org/detail.go?type=disease&acc=MESH%3AD000086382&view=phenotype 
        * Place files in src/data/CTD/
* Preparing DRKG files:
    1. Download from https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz 
    2. Extract tar.gz file
    3. Place extracted /embed folder in src/data/DRKG/
* SAR-CoV-2 w/ Gene Interaction Study
    1. Download from: https://www.nature.com/articles/s41586-020-2286-9#Sec36, Supplementary Table 5
    2. Rename as baits-prey-mist.csv
    3. Place in src/data/biology-database/

## Script functionalities
* preprocess.ipynb - Preprocessing of the datasets
* graph-representation-ranking-Update.ipynb - Creates knowledge graph and ranking model of the preprocessed data. The file includes testing and training for each model, and well as comparison to baseline models. The models are evaluated on accuracy (AUROC and AUPRC).

## How to run code
1. Prepare all the datasets according to "Preparing the datasets"
2. Run src/preprocess.ipynb to preprocess the datasets
3. Run src/graph-representation-ranking-Update.ipynb to generate knowledge graph and rank models
4. The top 300 potent drugs will be reported out in src/top300_drugs.csv 