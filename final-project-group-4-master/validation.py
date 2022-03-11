import argparse
import numpy as np
import pickle
import pandas as pd

# build parser 
parser = argparse.ArgumentParser(description='Validate a trained model for ece5307 project')
parser.add_argument("model_path",help="path to model file")
parser.add_argument("--Xtr_path",default="Xtr.csv",help="path to training-feature file")
parser.add_argument("--ytr_path",default="ytr.csv",help="path to training-label file")
parser.add_argument("--Xts_path",default="Xts.csv",help="path to test-feature file")
parser.add_argument("--yts_hat_path",default="yts_hat.csv",help="path to test-label-prediction file")

# parse input arguments
args = parser.parse_args()
model_path = args.model_path
Xtr_path = args.Xtr_path
ytr_path = args.ytr_path
Xts_path = args.Xts_path
yts_hat_path = args.yts_hat_path
if not yts_hat_path.endswith('.csv'): 
    print("Error: Argument of --yts_hat_path must end in .csv")
    quit()

# load data
Xtr = np.loadtxt(Xtr_path, delimiter=",")
ytr = np.loadtxt(ytr_path, delimiter=",")
Xts = np.loadtxt(Xts_path, delimiter=",")

# load model
if model_path.endswith('.json'):
    # XGBOOST
    from xgboost import XGBClassifier 
    model = XGBClassifier()
    model.load_model(model_path)
    ytr_hat = model.predict(Xtr)
    yts_hat = model.predict(Xts)
elif model_path.endswith('.pth'):
    # PYTORCH
    import torch 
    model = torch.jit.load(model_path)
    with torch.no_grad():
        ytr_hat = model(torch.Tensor(Xtr)).detach().numpy().argmax(axis=1)
        yts_hat = model(torch.Tensor(Xts)).detach().numpy().argmax(axis=1)
elif model_path.endswith('.pkl'): 
    # SKLEARN
    model = pickle.load(open(model_path, 'rb'))
    ytr_hat = model.predict(Xtr)
    yts_hat = model.predict(Xts)
else:
    print("Error: Unrecognized extension on model_path.  Should be .pkl for Sklearn models, or .pth for PyTorch models, or .json for XGBoost models")
    quit()


# print training accuracy 
acc = np.mean(ytr_hat==ytr)
print('training accuracy = ',acc)

# save test-label predictions in a csv file 
df = pd.DataFrame(data={'Id':np.arange(len(yts_hat)),
                        'Label':np.int64(yts_hat)})
df.to_csv(yts_hat_path, index=False)
print('test label predictions saved in',yts_hat_path)
