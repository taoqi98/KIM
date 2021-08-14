# KIM
- Code of our KIM model

# Data Preparation
- If you want to condcut experiements based on this project, you can download MIND-Small dataset in https://msnews.github.io/index.html
- All data in MIND are stored in data-root-path
- We used the glove.840B.300d embedding vecrors in https://nlp.stanford.edu/projects/glove/
- The embedding file should be stored in embedding\_path\glove.840B.300d.txt
- The meta data of KG (including graph structre, and pre-trained transE embeddings) should be stored in KG\_root\_path

# Code Files
- preprocess.py: containing functions to preprocess data
- utils.py: containing some util functions
- generator.py: containing data generator for model training and evaluation
- models.py: containing codes for implementing the KIM model
- hypers.py: containing settings of hyper-parameters
- Main.ipynb: containing codes for model training and evaluation
- ProcessRawData.ipynb: using for converting raw MIND files (train/news.tsv, train/behaviors.tsv, dev/news.tsv, dev/behaviors.tsv), into files used in our codes (data_root_path/docs.tsv, train.tsv, test.tsv)

