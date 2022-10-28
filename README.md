# Splitting-Hairs-and-Network-Traces

This is the repository for the paper:

*Matthias Beckerle, Jonathan Magnusson, and Tobias Pulls. Splitting Hairs and
Network Traces: Improved Attacks Against Traffic Splitting as a Website
Fingerprinting Defense. WPES 2022.*

The following research artifacts are provided as-is and intended for research
purposes only. Our primarly goal with sharing these artifacts is to enable
others to reproduce our results. A secondary goal is to enable other researchers
to use Maturesc when evaluating Website Fingerprinting defenses.

## Maturesc
TODO: instructions for running Maturesc on a dataset with example terminal
output.

```
$ python3 maturesc.py -someargs dataset/folder
terminal output
terminal output
terminal output
terminal output
```

Explain output

## Datasets
Here you can download all the datasets packaged to work with our code:
- [CoMPS real-world (971M)](https://dart.cse.kau.se/Splitting-Hairs-and-Network-Traces/comps-rw.tar)
- [CoMPS simulated (5.6G)](https://dart.cse.kau.se/Splitting-Hairs-and-Network-Traces/comps-wang-x10.tar)
- [HyWF real-world (639M)](https://dart.cse.kau.se/Splitting-Hairs-and-Network-Traces/hywf-rw.tar)
- [HyWF simulated (5.3G)](https://dart.cse.kau.se/Splitting-Hairs-and-Network-Traces/hywf-wang-x10.tar)
- [TS-BWR5 real-world (484M)](https://dart.cse.kau.se/Splitting-Hairs-and-Network-Traces/ts-rw.tar)
- [TS-BWR5 simulated (6.0G)](https://dart.cse.kau.se/Splitting-Hairs-and-Network-Traces/ts-bwr5-wang-x10.tar)

Please note that the real-world datasets are repackaged datasets from the
respective authors. We share them here in the re-packaged format for sake of
transparency and towards reproducability of our results. The simulated datasets
are based on [Tao Wang](https://www.cs.sfu.ca/~taowang/wf/) et al.'s popular
[Wa-kNN dataset](https://www.cs.sfu.ca/~taowang/wf/data/knndata.zip).

### Files and Folders
Each unpacked dataset contains a data folder with the actual traces. The
structure here varies slightly depending on how we mapped the original
real-world structure. Both real-world and simulated datasets contain csv
folders. Inside each folder is ten csv files for ten folds. We used `fold-0.csv`
for the results in the paper. Entries in a csv file point to a data file and
documents its use. For example, let's look at CoMPS real-world:

```
$ head -n 4 comps-rw/csv-closed/fold-0.csv 
class,file,is_merged,is_split,is_empty,is_train,is_valid,is_test
0,0/0,False,True,False,True,False,False
0,0/1,False,True,False,True,False,False
0,0/2,False,True,False,True,False,False
```

Here, each entry in the csv specifies the class (website) and six true/false flags:
- is_merged: true if the trace is a merged trace of multiple paths (only used
  for training)
- is_split: true if the trace is split, i.e., from one path observed by our
  attacker. Typically the file path will encode which path it is from in the
  simulation run.
- is_empty: true if the trace is empty.
- is_train: true if the sample is for training.
- is_valid: true if the sample is for validation.
- is_test: true if the sample is for testing.

### Splitting Simulators
We implemented simulators for CoMPS, HyWF, and TS-BWR5. They can be found in the
[simulators/](https://github.com/m-bec/Splitting-Hairs-and-Network-Traces/tree/main/simulators)
folder. These simulators were used to generated the simulated datasets linked
above.