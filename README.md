# DPProm
### DPProm: A Two-layer Predictor for Identifying Promoters and Their Types on Phage Genome Using Deep Learning
## Introduction
### Motivation:
With the number of phage genomes increasing, it is urgent to develop new bioinformatics methods for phage genome annotation. Promoter is a DNA region and important for gene transcriptional regulation. In the era of post-genomics, the availability of data made it possible to establish computational models for promoter identification with robustness.
### Results:
In this work, we proposed a two-layer model DPProm. On the first layer, for identifying the promoters, DPProm-1L was presented with a dual-channel deep neural network ensemble method fusing multi-view features, including sequence feature and handcrafted feature; on the second layer, for predicting promoter types (host or phage), DPProm-2L was proposed based on convolutional neural network (CNN). At the whole phage genome level, a novel sequence data processing workflow composed of sliding window module and merging sequences module was raised. Combined with the novel data processing workflow, DPProm could effectively decrease the false positives for promoter prediction on the whole phage genome.
### Related Files
dataencoder.py: encode the input sequences  
dataprocess.py: read sequences  
run_prokka.py: genome-wide annotations were performed using prokka tools  
cut_genome.py: the sequence of non-coding regions is intercepted from the annotation information, and the sliding window  
predict_independ.py: predict whether the input sequence is a promoter sequence  
merge_seqs.py: sequence merge the result of DPProm-1L and check the merge result  
cdhit.py: cdhit removes redundant sequences in the same non-coding region  
type.py: predicted promoter type: host or phage  
transfer_method:  transfer learning method is used to improve DPProm-1L  
## Installation
### Requirement
#### Linux： Ubuntu 16.04 LTS or later
#### python >= 3.6
