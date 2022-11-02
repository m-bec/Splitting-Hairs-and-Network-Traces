#!/usr/bin/env python
# coding: utf-8

# --------
# Maturesc 
# --------
# License: GPL 3
# --------
# Author: Matthias Beckerle
# Year: 2022 
# --------

#===========================
# Used DL libaries and versions for this script:
# fastai     : 2.6.0
# torch      : 1.11.0
#===========================

import argparse
import copy

from datetime import datetime

import fastai

from fastai.metrics import *
from fastai.vision.all import *
from fastai.vision.models.xresnet import _xresnet

import fastcore

import sklearn.metrics as skm
from sklearn.metrics import classification_report

import sys


# PRINT DATE AND LIB VERSIONS
#=========================================================
DATE = datetime.now().strftime('%Y_%m_%d_%H%M')
print("")
print("==============================================")
print(DATE)
print("---------")
print('fastai     :', fastai.__version__)
print('torch      :', torch.__version__)
print("---------")


# GLOBAL VARIABLES - NEED TO BE DEFINED BEFORE USAGE
#=========================================================
# DATA PATH
DATA_ROOT_DIR = "/mnt/ramdisk/split/"
DATA_SUB_DIR = "data"   # (DATA_ROOT_DIR, DATASET_NAME, DATA_SUB_DIR)

# INPUT CSV COLUMN NAMES
C_LABEL  = 'class'      # the real name of the class if available
C_LABEL2 = 'class'      # the numerical synonym of the class
C_FNAME  = 'file'       # filepath + filename
C_VALID  = 'is_valid'   # validation set indicator
C_TEST   = 'is_test'    # test set indicator

#Example code works with a csv file that looks like that:
#     class file  is_merged  is_split  is_empty  is_train  is_valid  is_test
#  0      0  0/0      False      True     False      True     False    False
#  1      0  0/1      False      True     False      True     False    False
#  2      0  0/2      False      True     False      True     False    False
#  3      0  0/3      False      True     False      True     False    False
#  4      0  0/4      False      True     False      True     False    False
#  ...
 

# COMMAND LINE PARAMETERS
#=========================================================
ap = argparse.ArgumentParser()

ap.add_argument("-dataset",                 required=True,                           help="e.g. comps-rw")
ap.add_argument("-csvdir",                  required=True,                           help="csv-closed, csv-open, or csv")

ap.add_argument("-csvfile",                 required=False,  default="fold-0.csv",   help="csv file name")
ap.add_argument("-resultdir",               required=False,  default="results",      help="results directory")


ap.add_argument("-device",   type = int,    required=False,  default=0,      help="GPU 0 or GPU 1")

ap.add_argument("-divide",   type = int,    required=False,  default=1,      help="Divide Dataset e.g. 1")

ap.add_argument("-lengthx",  type = int,    required=False,  default=40,     help="Trace length e.g. 70 for 70*70=4900")
ap.add_argument("-bs",       type = int,    required=False,  default=64,     help="Batch size")

ap.add_argument("-arch",                    required=False,  default="1d18", help="Architecture of xresnets e.g. 1d18 or 1d1c18")
ap.add_argument("-dropout",  type = float,  required=False,  default=0.5,    help="Dropout e.g. 0.5")
ap.add_argument("-type",                    required=False,  default="FOC",  help="FOC")
ap.add_argument("-epochs",   type = int,    required=False,  default=3,      help="n of epochs to train")
ap.add_argument("-lr",       type = float,  required=False,  default=0.02,   help="learning rate")

args = vars(ap.parse_args())


# COMMAND LINE PARAMETERS -> VARIABLES
#=========================================================
DATASET_NAME = args["dataset"]
CSV_DIR = args["csvdir"]
CSVFILE = args["csvfile"] 
RESULT_DIR = args["resultdir"]

CUDA_DEVICE = args["device"]
torch.cuda.set_device(CUDA_DEVICE)

LENGTHX = args["lengthx"]
LENGTH = LENGTHX*LENGTHX;

BS = args["bs"]

DIVIDE = args["divide"]
ARCH = args["arch"]

DROPOUT = args["dropout"]
TYPE = args["type"]
EPOCHS = args["epochs"]
LR = args["lr"]


# GENERATE SUBFOLDERS
#=========================================================
RESULT_SUBDIR = "_".join([DATASET_NAME, CSV_DIR]) 
NEWSUB = "".join([RESULT_DIR,"//",RESULT_SUBDIR]) 

if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)    
if not os.path.exists(NEWSUB): os.makedirs(NEWSUB)


# DEFINE FILE NAMES
#=========================================================
FN_PRE = "_".join([DATASET_NAME, CSV_DIR])
FN_SUBPRE = "".join([NEWSUB,"//", FN_PRE]) # with subfolder
FN_SUF = "_".join([str(LENGTH).zfill(5), str(BS), str(DIVIDE), ARCH, str(DROPOUT), TYPE, str(EPOCHS), str(LR), "", DATE])
FN_Model = "_".join([FN_PRE, "", FN_SUF])  # without subfolder - used for auto save 
FN = "".join([FN_SUBPRE, "__", FN_SUF])  # with subfolder

FN_HISTORY = "".join([FN_SUBPRE, "__HIST__", FN_SUF, ".csv"])


# HELPER PATH VARIABLES
#=========================================================
DIR = os.path.join(DATA_ROOT_DIR, DATASET_NAME, DATA_SUB_DIR)
CSVDIR = os.path.join(DATA_ROOT_DIR, DATASET_NAME, CSV_DIR)
path = Path(DIR)
csvpath = Path(CSVDIR)
Path.BASE_PATH = path


# CSV -> DataFrame; PREPROCESSING AND VARIANTS
#=========================================================
df_prefilter = pd.read_csv(csvpath/CSVFILE)
df_all = df_prefilter[(df_prefilter.is_merged==False)] 
df_train_all = df_all[(df_all.is_test==False)] 
df_active = None # see set_trainingset() or set_testset()


# PRINT INPUT OVERVIEW
#=========================================================
print("input data: ", path)
print("input csv:  ", csvpath/CSVFILE)
print("results:    ", Path(FN))
print("---------")
print(df_train_all.head())
print("---------")


# GLOBAL HELPER FUNCTION
#===========================
def items():                return df_active.index


# GLOBAL VARIABLE FUNCTIONS
#===========================
def fname_to_path(fname):   return (Path(f'{str(path)}/'+fname))
def file(index):            return fname_to_path(df_active.loc[index, C_FNAME]) 

def label(index):           return (df_active.loc[index, C_LABEL],df_active.loc[index, C_LABEL2])
def labels():               return zip(df_active[C_LABEL],df_active[C_LABEL2])


# TRACE PARSING
#===========================
def readlog1c(path):
    tmp = 0; i = 0; 
    data = np.zeros((1, LENGTH), dtype=np.float32)  
    with open(path, 'r') as f: 
          for line in f:
                parts = line.strip().split(); #[0]:time in sec; [1]:direction
                if i >= LENGTH:
                    break
                data[0][i] = float((float(parts[0])-float(tmp))*10000)*float(parts[1])  # add time in [0.1 ms] (from sec)  # TIME DIF *  DIRECTIONAL
                tmp = float(parts[0]) # store old time
                i = i + 1                   
    return torch.as_tensor(data) 

def readlog2c(path):
    tmp = 0; i = 0; 
    data = np.zeros((2, LENGTH), dtype=np.float32)  
    with open(path, 'r') as f: 
          for line in f:
                parts = line.strip().split(); #[0]:time in sec; [1]:direction including package size 
                if i >= LENGTH:
                    break
                data[0][i] = float((float(parts[0])-float(tmp))*10000) # add time in [0.1 ms] (from sec)  # TIME DIF
                data[1][i] = float(parts[1])                                                # DIRECTIONAL PACKET SIZE
                tmp = float(parts[0]) # store old time
                i = i + 1                   
    return torch.as_tensor(data) 

if "1d1c" in ARCH: readlog = readlog1c
else: readlog = readlog2c


# TRANSFORMATIONS
#===========================
class TraceTfm(ItemTransform):
    def setups(self, index):
        self.labeller = label
        self.vocab, self.o2i = uniqueify(labels(), sort=False, bidir=True)
        self.c = len(self.vocab)
    def encodes(self, o): return (readlog(file(o)),self.o2i[self.labeller(o)])
    def decodes(self, x): return None

def transformer():
    valid_idx = df_active[df_active[C_VALID]].index.values
    splitter = IndexSplitter(valid_idx)(items())
    tls = TfmdLists(items(), [TraceTfm()], splits=splitter)
    return tls

def testset_transformer():
    test_idx = df_active[df_active[C_TEST]].index.values
    test_splitter = IndexSplitter(test_idx)(items())    
    tls = TfmdLists(items(), [TraceTfm()], splits=test_splitter)
    return tls
 
def dataloader(tls):
    dls = tls.dataloaders(bs=BS)
    return dls


# ARCHITECTURE AND TRAINING
#===========================
def top_2_acc(inp, targ): return top_k_accuracy(inp, targ, k=2, axis=-1)
def top_5_acc(inp, targ): return top_k_accuracy(inp, targ, k=5, axis=-1)
metrics_used = [accuracy, top_2_acc, top_5_acc]

def a_1d18L(dls):  return Learner(dls, _xresnet(False, 1, [1,1,1,1],         c_in=2, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d18(dls):   return Learner(dls, _xresnet(False, 1, [2,2,2,2],         c_in=2, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d18D(dls):  return Learner(dls, _xresnet(False, 1, [2,2,2,2,1,1],     c_in=2, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d18DD(dls): return Learner(dls, _xresnet(False, 1, [2,2,2,2,1,1,1,1], c_in=2, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d34(dls):   return Learner(dls, _xresnet(False, 1, [3,4,6,3],         c_in=2, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d50(dls):   return Learner(dls, _xresnet(False, 4, [3,4,6,3],         c_in=2, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d101(dls):  return Learner(dls, _xresnet(False, 4, [3,4,23,3],        c_in=2, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d152(dls):  return Learner(dls, _xresnet(False, 4, [3,8,36,3],        c_in=2, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d1c18(dls): return Learner(dls, _xresnet(False, 1, [2,2,2,2],         c_in=1, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d1c34(dls): return Learner(dls, _xresnet(False, 1, [3,4,6,3],         c_in=1, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)
def a_1d1c50(dls): return Learner(dls, _xresnet(False, 4, [3,4,6,3],         c_in=1, n_out=dls.c, ndim=1), loss_func=CrossEntropyLossFlat(), metrics=metrics_used)

def train(learn, DROPOUT, TYPE, EPOCHS, LR, FN):
    learn.model[10].p = DROPOUT;
    cbs_used=[SaveModelCallback(monitor='accuracy', comp=np.greater, fname=FN_Model), ShowGraphCallback(), 
              CSVLogger(FN_HISTORY)]                            
    if TYPE =='FOC': learn.fit_one_cycle(EPOCHS, LR, cbs=cbs_used)    
    return learn

def set_trainingset(DIVIDE):
    global df_active
    # SET GLOBAL ACTIVE SET!
    df_train_div = df_train_all.iloc[0::DIVIDE]
    df_train_div = df_train_div.reset_index()
    df_train_div = df_train_div.rename(columns = {'index':'OrigIndex'})
    df_active = df_train_div
    # END SET GLOBAL ACTIVE SET!

def set_testset():
    global df_active
    # SET GLOBAL ACTIVE SET!
    df_temp = df_all[(df_all.is_valid==False)] 
    df_temp = df_temp.reset_index()
    df_temp = df_temp.rename(columns = {'index':'OrigIndex'})    
    df_active = df_temp
    # END SET GLOBAL ACTIVE SET!
    
def Test_results(learn):    
    # Change validation set to test set !NON REVERSABLE! 
    set_testset()
    testset_tls = testset_transformer()
    testset_dls = dataloader(testset_tls)
    testlearner = learn # copy.deepcopy(learn)
    testlearner.dls = testset_dls
    return testlearner
    
def set_arch(ARCH):
    tls = transformer()
    dls = dataloader(tls) 
    if ARCH == "1d18L":  learn = a_1d18L(dls)  
    if ARCH == "1d18":   learn = a_1d18(dls) 
    if ARCH == "1d18D":  learn = a_1d18D(dls)  
    if ARCH == "1d18DD": learn = a_1d18DD(dls)  
    if ARCH == "1d34":   learn = a_1d34(dls)
    if ARCH == "1d50":   learn = a_1d50(dls)
    if ARCH == "1d101":  learn = a_1d101(dls)
    if ARCH == "1d152":  learn = a_1d152(dls)
    if ARCH == "1d1c18": learn = a_1d1c18(dls)   
    if ARCH == "1d1c34": learn = a_1d1c34(dls)   
    if ARCH == "1d1c50": learn = a_1d1c50(dls)
    return learn

def do_learning(DIVIDE, ARCH, DROPOUT, TYPE, EPOCHS, LR, FN): 
    set_trainingset(DIVIDE)
    learn = set_arch(ARCH)    
    learn = train(learn, DROPOUT, TYPE, EPOCHS, LR, FN)
    return learn


# RESULT METRICS
#===========================
def show_result_metrics(learn):    
    vocab = learn.dls.vocab
    _,targs,decoded =learn.get_preds(with_decoded=True) 
    t,d = flatten_check(targs, decoded)
    print(classification_report(t, d, labels=list(learn.dls.o2i.values()), target_names=[str(v) for v in vocab], zero_division=0))
    
    
# MAIN 
#===========================    
learn = do_learning(DIVIDE, ARCH, DROPOUT, TYPE, EPOCHS, LR, FN)


# PRINT RESULTS
#===========================
print("---------")
show_result_metrics(learn)
print("---------")

res_Validation = learn.validate()
print("On validation set: valid_loss, accuracy, top_2_acc, top_5_acc")
print( res_Validation) # valid_loss, accuracy, top_2_acc, top_5_acc
print("---------")

testset_learner = Test_results(learn)
res_Test = testset_learner.validate()
print("On test set: valid_loss, accuracy, top_2_acc, top_5_acc")
print(res_Test) # valid_loss, accuracy, top_2_acc, top_5_acc
