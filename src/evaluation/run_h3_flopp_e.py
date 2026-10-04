from pathlib import Path
import json
import random
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from src.models.spectral_tower import SpectralTower

SEEDS = [42, 43, 44, 45, 46]
X = np.load("data/processed/spectra/c4_dedup_train_spectra.npy")
y_raw = np.load("data/processed/spectra/c4_dedup_train_labels.npy")
groups = np.load("data/processed/spectra/c4_dedup_train_groups.npy")

classes = ["HDPE","LDPE","PET","PP","PS","PVC"]
label_to_id = {c:i for i,c in enumerate(classes)}
y = np.array([label_to_id[str(v)] for v in y_raw])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path("weights").mkdir(exist_ok=True)
results = []

for seed in SEEDS:
    print(f"\n===== SEED {seed} =====")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    split = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
    tr, va = next(split.split(X, y, groups))

    model = SpectralTower(num_classes=6).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-6)

    counts=np.bincount(y[tr],minlength=6)
    weights=len(tr)/(6*np.maximum(counts,1))
    lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))

    best=-1
    best_state=None
    patience=0

    tx=torch.from_numpy(X[tr]).float().to(device)
    ty=torch.from_numpy(y[tr]).long().to(device)
    vx=torch.from_numpy(X[va]).float().to(device)
    vy=torch.from_numpy(y[va]).long().to(device)

    for epoch in range(50):
        model.train()
        perm=torch.randperm(len(tr),device=device)

        for start in range(0,len(tr),32):
            idx=perm[start:start+32]
            opt.zero_grad()
            out=model(tx[idx])
            loss=lossfn(out,ty[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred=torch.argmax(model(vx),1).cpu().numpy()

        mf1=f1_score(y[va],pred,average="macro",zero_division=0)
        sched.step()

        if mf1>best:
            best=mf1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            patience=0
        else:
            patience+=1
            if patience>=10:
                break

    ckpt=Path(f"weights/h3_seed_{seed}_best.pth")
    torch.save(best_state,ckpt)

    model.load_state_dict(best_state)
    model.eval()

    # FLOPP-e
    from pathlib import Path as P
    from src.preprocessing.spectral_preprocess import preprocess_spectra

    c4=P("data/raw/ftir_plastic_c4/FTIR-PLASTIC-c4/FTIR_PLASTIC_c4_deduplicated.csv")
    import pandas as pd
    row=pd.read_csv(c4,nrows=1)
    grid=np.asarray(row[[c for c in row.columns if c.startswith("Data(x)")]].iloc[0],dtype=float)

    files=sorted(P("data/raw/flopp_e/FLOPP and FLOPP-e/FLOPP-e .csv").glob("*.CSV"))
    xx=[]; yy=[]

    for f in files:
        name=f.stem.upper()
        if name.startswith("PET "): lab=2
        elif name.startswith("PP "): lab=3
        elif name.startswith("PS "): lab=4
        elif name.startswith("PVC "): lab=5
        else: continue

        d=np.loadtxt(f,delimiter=",")
        r=np.interp(grid,d[:,0],d[:,1]).astype("float32")
        xx.append(r); yy.append(lab)

    xx=preprocess_spectra(np.stack(xx))
    with torch.no_grad():
        fp=torch.argmax(model(torch.from_numpy(xx).float().to(device)),1).cpu().numpy()

    cnn_f1=f1_score(yy,fp,average="macro",zero_division=0)

    svm=SVC(kernel="rbf",class_weight="balanced",random_state=seed)
    svm.fit(X[tr],y[tr])
    sp=svm.predict(xx)
    svm_f1=f1_score(yy,sp,average="macro",zero_division=0)

    print(f"Val Macro-F1: {best:.4f}")
    print(f"FLOPP-e CNN Macro-F1: {cnn_f1:.4f}")
    print(f"FLOPP-e SVM Macro-F1: {svm_f1:.4f}")

    results.append({"seed":seed,"val_cnn_f1":float(best),"flopp_e_cnn_f1":float(cnn_f1),"flopp_e_svm_f1":float(svm_f1)})

out=Path("results/hypothesis/h3_flopp_e_5seeds.json")
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(results,indent=2))
print("\n===== DONE =====")
print(json.dumps(results,indent=2))
