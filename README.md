# AnalyzingGenderBias

## Table of contents

* [General info](#general-info)
* [Setup](#setup)



## General info

This project is the code behind my project to Utställningen Unga Forskare 2020. The code includes the scripts used to gather data, train the neural network, analyze literature and songs, create sentiment vectors, and visualize the results in Jupyter Notebooks. The produced data from personal experiments can be found under _EXPERIMENTS (literature and songs) and results/period_2 (literature over time). To extract own literary data, use get_overtime2.py. The trained neural network can be found under neural_sentiment_model.

**Note**: If you wish to train your own neural network, make sure to download the imdb dataset (for similar results) or use own classified data. It was not possible to include directly because of file size.

## Setup

To run this project, clone the github repo. The easiest way to get started with experimenting is to open up the file notebook_OverTime (for analysis over time) or notebook_AliceWonderland (for analysis of the single book). Run the cells in order and make sure to give each cell enough time to complete before moving on.
