# COVID-19 Drug Exploratory

# Purpose
The purpose of this project is to replicate the experiement and results from - 
* Hsieh, K., Wang, Y., Chen, L., Zhao, Z., Savitz, S., Jiang, X., Tang, J., & Kim, Y. (2020). Drug Repurposing for COVID-19 using Graph Neural Network with Genetic, Mechanistic, and Epidemiological Validation. Research square, rs.3.rs-114758. https://doi.org/10.21203/rs.3.rs-114758/v1  
* Author's original research code can be found here: https://github.com/yejinjkim/drug-repurposing-graph

# Team Members
* Carol Chen
* Vy Nguyen

# Script functionalities
* preprocess.ipynb - Preprocessing of the datasets
* graph-representation-ranking-Update.ipynb - Creates knowledge graph and ranking model of the preprocessed data. The file includes testing and training for each model, and well as comparison to baseline models. The models are evaluated on accuracy (AUROC and AUPRC).

# Preparing the datasets
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

# How to run code
1. Prepare all the datasets according to "Preparing the datasets"
2. Run src/preprocess.ipynb to preprocess the datasets
3. Run src/graph-representation-ranking-Update.ipynb to generate knowledge graph and rank models
4. The top 300 potent drugs will be reported out in src/top300_drugs.csv 