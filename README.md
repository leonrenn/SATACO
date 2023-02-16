# SATACO
On top tool for Simple Analysis (SA) implementing algorithms from Testing Analyses' COrrelations (TACO).
[SimpleAnalysis](https://simpleanalysis.docs.cern.ch)
[TACO](https://gitlab.com/t-a-c-o/)

## Purpose and General Functionality

Give a proof of concept of the paper: <br>
[Strength in numbers: optimal and scalable combination of LHC new-physics searches](https://arxiv.org/abs/2209.00025)
<br>

The paper states that to discover new physics beyond the SM, it will be necessary to combine multiple SR of different analyses, because new physics will not leave a siginificant signature in one channel but will disperse over many.<br>

The combination of SR is not trivial since the analyses are not statistically orthogonal and therefore have a nono zero correlation coefficient.<br>

Correlating SRs cannot be combined. The goal is now to define correlations matrices of the SRs and find the best combination of mulitple SRs. This process is not done combinatorically due to the increasingly fast rising numbers of possibilities, but via a longest path method in a _Directed Acyclic Graph (DAG)_. <br>

The program provides the user with visual, text and pure data results.<br>

## Get Started

### 1. Clone the git repository to your local machine

```sh
git clone https://github.com/leonrenn/SATACO.git
```


### 2. Change into the repository and install the requirements

To install the requirements do:
```sh
cd SATACO && pip install -r requirements.txt
```

## CLI

The tool is meant to be a CL tool. _SimpleAnalysis_ should be installed,<br>
to do real world applications. For testing and newcomers just provide the <br>
command 
```sh
python src/main.py -r data/
```

