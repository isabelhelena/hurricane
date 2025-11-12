import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.utils import make_sequences, standardize_train, haversine_distance

SEQ_LEN = 4
FEATURES = ["lat","lon","wind","pressure"]

df = pd.read_csv("data/sample_hurdat2_subset.csv", parse_dates=["datetime"])
X, y = make_sequences(df, FEATURES, seq_len=SEQ_LEN)

n = len(X); idx = np.random.permutation(n)
train_size = int(0.8*n)
train_idx, val_idx = idx[:train_size], idx[train_size:]
X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

stdX, stdy, destdy, feat_mean, feat_std = standardize_train(X_train, y_train)
X_train_std, X_val_std = stdX(X_train), stdX(X_val)
y_train_std, y_val_std = stdy(y_train), stdy(y_val)
X_train_flat, X_val_flat = X_train_std.reshape(len(X_train_std), -1), X_val_std.reshape(len(X_val_std), -1)

N, D = X_train_flat.shape; M = y_train_std.shape[1]
W, b = np.zeros((D,M)), np.zeros(M)

def mse_loss(Xb,yb,W,b): return ((Xb@W+b - yb)**2).mean()

lr, epochs, batch_size = 0.05, 200, 32
train_losses, val_losses = [], []

for _ in range(epochs):
    perm = np.random.permutation(N)
    for i in range(0, N, batch_size):
        Xb, yb = X_train_flat[perm[i:i+batch_size]], y_train_std[perm[i:i+batch_size]]
        pred = Xb@W+b
        grad_pred = 2*(pred-yb)/len(Xb)
        W -= lr * Xb.T@grad_pred
        b -= lr * grad_pred.sum(0)
    train_losses.append(mse_loss(X_train_flat,y_train_std,W,b))
    val_losses.append(mse_loss(X_val_flat,y_val_std,W,b))

plt.plot(train_losses,label="train"); plt.plot(val_losses,label="val")
plt.legend(); plt.title("Baseline MSE Loss"); plt.savefig("figures/loss_curve.png")

y_pred = destdy(X_val_flat@W + b)
def mae(a,b): return np.mean(np.abs(a-b),0)
def rmse(a,b): return np.sqrt(np.mean((a-b)**2,0))
def r2(a,b): return 1 - ((a-b)**2).sum(0)/((a-a.mean(0))**2).sum(0)

metrics = {
 "MAE": dict(zip(FEATURES, map(float, mae(y_val, y_pred)))),
 "RMSE": dict(zip(FEATURES, map(float, rmse(y_val, y_pred)))),
 "R2": dict(zip(FEATURES, map(float, r2(y_val, y_pred)))),
 "Haversine_km_mean": float(haversine_distance(y_val[:,0],y_val[:,1],y_pred[:,0],y_pred[:,1]).mean())
}
os.makedirs("outputs", exist_ok=True)
json.dump(metrics, open("outputs/metrics.json","w"), indent=2)
