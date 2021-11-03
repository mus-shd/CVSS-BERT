# CVSS-BERT
CVSS-BERT: Explainable Natural Language Processing to Determine the Severity of a Computer Security Vulnerability from its Description

This repository contains the code of the following research paper:

M. R. Shahid and H. Debar , "CVSS-BERT: Explainable Natural Language Processing to Determine the Severity of a Computer Security Vulnerability from its Description," *2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)*, 2021.

**Abstract:** When a new computer security vulnerability is publicly disclosed, only a textual description of it is available. Cybersecurity experts later provide an analysis of the severity of the vulnerability using the Common Vulnerability Scoring System (CVSS). Specifically, the different characteristics of the vulnerability are summarized into a vector (consisting of a set of metrics), from which a severity score is computed. However, because of the high number of vulnerabilities disclosed everyday this process requires lot of manpower, and several days may pass before a vulnerability is analyzed. We propose to leverage recent advances in the field of Natural Language Processing (NLP) to determine the CVSS vector and the associated severity score of a vulnerability from its textual description in an explainable manner. To this purpose, we trained multiple BERT classifiers, one for each metric composing the CVSS vector. Experimental results show that our trained classifiers are able to determine the value of the metrics of the CVSS vector with high accuracy. The severity score computed from the predicted CVSS vector is also very close to the real severity score attributed by a human expert. For explainability purpose, gradient-based input saliency method was used to determine the most relevant input words for a given prediction made by our classifiers. Often, the top relevant words include terms in agreement with the rationales of a human cybersecurity expert, making the explanation comprehensible for end-users.


The repo is organized as follows:
- data: contains the data used for the project.
- models: contains the trained models
- explainable_bert_classifier: a package that contains all the necessary codes within 3 modules (details provided below)
- app: Contains two sub-directory v1 and v2. v1 contains a Dockerfile two deploy the developed models in a containerized application using FastAPI. v2 also contains a docker-compose.yml file and allow the deployment of a stack of 2 containers, one to serve the models using FastAPI, the other to save the queries along with the predictions in a MongoDB database.
- demo_notebook.ipynb:
- train.py:


