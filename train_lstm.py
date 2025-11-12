import argparse, torch, pandas as pd, numpy as np, json
from torch.utils.data import Dataset, DataLoader
from src.model_lstm import LSTMForecaster
from src.utils import make_sequences, standardize_train

class SeqDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

def train(args):
    df = pd.read_csv("data/sample_hurdat2_subset.csv", parse_dates=["datetime"])
    FEATURES = ["lat","lon","wind","pressure"]
    X,y = make_sequences(df, FEATURES, seq_len=args.seq_len)
    n=len(X); idx=np.random.permutation(n); tr=int(0.8*n)
    stdX,stdy,destdy,_,_ = standardize_train(X[:tr],y[:tr])
    Xtr,std_ytr = stdX(X[:tr]),stdy(y[:tr])
    Xva,std_yva = stdX(X[tr:]),stdy(y[tr:])
    train_loader = DataLoader(SeqDataset(Xtr,std_ytr),batch_size=args.batch,shuffle=True)
    val_loader   = DataLoader(SeqDataset(Xva,std_yva),batch_size=args.batch)
    model=LSTMForecaster(len(FEATURES),args.hidden,1,len(FEATURES))
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    loss_fn=torch.nn.MSELoss()
    for ep in range(args.epochs):
        model.train(); tot=0
        for xb,yb in train_loader:
            opt.zero_grad(); pred=model(xb); loss=loss_fn(pred,yb)
            loss.backward(); opt.step(); tot+=loss.item()*len(xb)
        print(f"Epoch {ep+1}: train loss {tot/len(train_loader.dataset):.4f}")
    json.dump({}, open("outputs/lstm_losses.json","w"))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--seq-len",type=int,default=4)
    p.add_argument("--hidden",type=int,default=64)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--lr",type=float,default=1e-3)
    train(p.parse_args())
