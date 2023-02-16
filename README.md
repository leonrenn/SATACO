# SATACO

  _____      _______       _____ ____  
  / ____|  /\|__   __|/\   / ____/ __ \ 
 | (___   /  \  | |  /  \ | |   | |  | |
  \___ \ / /\ \ | | / /\ \| |   | |  | |
  ____) / ____ \| |/ ____ \ |___| |__| |
 |_____/_/    \_\_/_/    \_\_____\____/ 

2
On top tool for Simple Analysis (SA) implementing algorithms from 
Testing Analyses' COrrelations (TACO).
3
[SimpleAnalysis](https://simpleanalysis.docs.cern.ch)
4
[TACO](https://gitlab.com/t-a-c-o/)

## Requirements

To install the requirements do:
32
```sh
pip install -r requirements.txt
```

### 4. Start with the tool
For this step it is necessary that you already generated data with the SA tool.<br>
Do not forget to set the flag [-n] in SA which provides the analysis result with a
ntuple. After that you can move the data (.root-files) from the analysis into your 
SATACO repository.<br>

Now you can use the tool in two ways:
1. With the -r flag which you can provide with mulitple comma seperated root files:
35
​
41
```sh
python src/main.py -r data/Diphoton2016.root,data/DMbb2016.root
```

## Results

- event_SR.parquet (gzip compressed): Events in every single Signal Region that at least accepted one event of the data.
- SR_SR.parquet (gzip compressed): Matrix of Signal Regions. Elements where compared and afterwards summed up for different Signal Regions.
- correlations.parquet (gzip compressed): Pearson correlation matrix for the SR_SR matrix.
- SR_SR.png: Figure of the correlations between the Signal Regions.
- correlations_threshold.png: Figure of correlations of Signal Regions with applied threshold for dividing them into: correlated and uncorrelated.
- correlations_path.png: Figure of correaltion matrix with elements marked in green that are part of the best SR combination (top 1 path).
- signal_regions.txt: Text file of all signal regions used in the current analysis run.
- best_SR_comb.txt: Best combination of mulitple paths (sorted).

![correlations_path.png](correlations_path.png)
correlations_path.png

Shows all uncorrelated SRs in green that build up the best possible combination of all SRs.