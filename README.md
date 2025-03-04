# GNN-49-Assignment

## Part-1  ( 2 marks)

Use the given dataset to design GNN based model for Node Classification as per details given below:

Graph: The ogbn-products dataset is an undirected and unweighted graph, representing an Amazon product co-purchasing network [1]. Nodes represent products sold in Amazon, and edges between two products indicate that the products are purchased together. We follow [2] to process node features and target categories. Specifically, node features are generated by extracting bag-of-words features from the product descriptions followed by a Principal Component Analysis to reduce the dimension to 100.

URL: https://ogb.stanford.edu/docs/nodeprop/#ogbn-productsLinks to an external site.

You are expected to create Model using Pytorch Geometric MP-GNN based library.

Do not copy existing model given in OGB site but create your own model.


## Part-2 ( 1 Marks)

Use https://ogb.stanford.edu/docs/home/Links to an external site. to learn dataset loading and checking performance method.


## Part-3 ( 2X6 Marks)

Based  on loaded dataset also compute following point:

Diameter , number of nodes and edges , Global Clustering Coefficient of existing graph
Plot the graph with label
 Refer Relevant material from Book related to Subgraph generation and provide explanation how you are generating subgraph
Generate Node Induced Subgraph.
Generate Node embedding using 2-hop method for all nodes in subgraph using MP-GNN library in PyTorch Geometric
Plot Subgrpah and compute their Diameter.
 

# Instruction for Student:

Student is expected to use BITS Provided Labs and write Python code for model development.
Pytorch Library and Pytorch Geometric Library can be used
Python Library can be use
NetworkX Library needs to use.
This group assignment, so each member of group is expected to contribute evenly in completing the assignment.
Assignment should be submitted in PDF format which contains Codes and their outcomes along with explanation.
Link of Python code in BITS library needs to be given as we will be running the code in Lab environment.
Each group has to work independently should not copy code or outcome of other group. If Plagiarism found the that will be dealt as per BITS Policy.


----------------------------------------------------------------------------------------