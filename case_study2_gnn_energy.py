#Case Study -2 – Project Code:

# ==========================================================
# Imports
# ==========================================================
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from tqdm import tqdm

# ==========================================================
# Config
# ==========================================================
L = 10
periodic = True
use_pos_encoding = True

batch_size = 4
lr = 1e-3
epochs = 200
edge_weight_factor = 5.0  # Last points weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Dataset (U/t (+noise)+ Energy)
# ==========================================================
U_t = [
    0.0, 0.060301508, 0.120603015, 0.180904523, 0.24120603,
    0.301507538, 0.361809045, 0.422110553, 0.48241206, 0.542713568,
    0.603015075, 0.663316583, 0.72361809, 0.783919598, 0.844221106,
    0.904522613, 0.964824121, 1.025125628, 1.085427136, 1.145728643,
    1.206030151, 1.266331658, 1.326633166, 1.386934673, 1.447236181,
    1.507537688, 1.567839196, 1.628140704, 1.688442211, 1.748743719,
    1.809045226, 1.869346734, 1.929648241, 1.989949749, 2.050251256,
    2.110552764, 2.170854271, 2.231155779, 2.291457286, 2.351758794,
    2.412060302, 2.472361809, 2.532663317, 2.592964824, 2.653266332,
    2.713567839, 2.773869347, 2.834170854, 2.894472362, 2.954773869,
    3.015075377, 3.075376884, 3.135678392, 3.195979899, 3.256281407,
    3.316582915, 3.376884422, 3.43718593, 3.497487437, 3.557788945,
    3.618090452, 3.67839196, 3.738693467, 3.798994975, 3.859296482,
    3.91959799, 3.979899497, 4.040201005, 4.100502513, 4.16080402,
    4.221105528, 4.281407035, 4.341708543, 4.40201005, 4.462311558,
    4.522613065, 4.582914573, 4.64321608, 4.703517588, 4.763819095,
    4.824120603, 4.884422111, 4.944723618, 5.005025126, 5.065326633,
    5.125628141, 5.185929648, 5.246231156, 5.306532663, 5.366834171,
    5.427135678, 5.487437186, 5.547738693, 5.608040201, 5.668341709,
    5.728643216, 5.788944724, 5.849246231, 5.909547739, 5.969849246,
    6.030150754, 6.090452261, 6.150753769, 6.211055276, 6.271356784,
    6.331658291, 6.391959799, 6.452261307, 6.512562814, 6.572864322,
    6.633165829, 6.693467337, 6.753768844, 6.814070352, 6.874371859,
    6.934673367, 6.994974874, 7.055276382, 7.115577889, 7.175879397,
    7.236180905, 7.296482412, 7.35678392, 7.417085427, 7.477386935,
    7.537688442, 7.59798995, 7.658291457, 7.718592965, 7.778894472,
    7.83919598, 7.899497487, 7.959798995, 8.020100503, 8.08040201,
    8.140703518, 8.201005025, 8.261306533, 8.32160804, 8.381909548,
    8.442211055, 8.502512563, 8.56281407, 8.623115578, 8.683417085,
    8.743718593, 8.804020101, 8.864321608, 8.924623116, 8.984924623,
    9.045226131, 9.105527638, 9.165829146, 9.226130653, 9.286432161,
    9.346733668, 9.407035176, 9.467336683, 9.527638191, 9.587939698,
    9.648241206, 9.708542714, 9.768844221, 9.829145729, 9.889447236,
    9.949748744, 10.01005025, 10.07035176, 10.13065327, 10.19095477,
    10.25125628, 10.31155779, 10.3718593, 10.4321608, 10.49246231,
    10.55276382, 10.61306533, 10.67336683, 10.73366834, 10.79396985,
    10.85427136, 10.91457286, 10.97487437, 11.03517588, 11.09547739,
    11.15577889, 11.2160804, 11.27638191, 11.33668342, 11.39698492,
    11.45728643, 11.51758794, 11.57788945, 11.63819095, 11.69849246,
    11.75879397, 11.81909548, 11.87939698, 11.93969849, 12.0
]



Energy = [
    -1.980686533, -2.096913142, -2.137606024, -2.182787696, -2.310626204,
    -2.291117157, -2.432958898, -2.439904726, -2.476481493, -2.549604139,
    -2.587061128, -2.588115039, -2.616988699, -2.654248153, -2.675096275,
    -2.720093986, -2.72522919, -2.779485013, -2.840630096, -2.860786323,
    -2.786840406, -2.831486724, -2.900465116, -2.826588031, -2.877765151,
    -2.925945209, -2.893618168, -2.917086412, -2.935573791, -2.955557042,
    -2.992191573, -2.974332422, -2.965422416, -3.022263356, -3.009447318,
    -3.05034633, -3.044674613, -3.006861659, -3.029115375, -3.043742277,
    -3.037228516, -3.053116556, -3.080963483, -3.106501102, -3.109283615,
    -3.042111602, -3.077750759, -3.092175645, -3.100383691, -3.098625391,
    -3.121384533, -3.182187015, -3.164135233, -3.104645802, -3.184779005,
    -3.174756083, -3.111386905, -3.182017202, -3.180301609, -3.156132891,
    -3.199693047, -3.17138208, -3.190408905, -3.17743952, -3.206567126,
    -3.169678769, -3.189487008, -3.174589817, -3.195299365, -3.180844815,
    -3.238475852, -3.166429777, -3.187862137, -3.240879395, -3.236178947,
    -3.243796403, -3.199761467, -3.240046635, -3.288265198, -3.236328589,
    -3.269436992, -3.200191389, -3.281755777, -3.229150558, -3.249926456,
    -3.221875401, -3.232450698, -3.292255405, -3.27938474, -3.288503435,
    -3.26425581, -3.225777343, -3.273040703, -3.282374728, -3.313231107,
    -3.29341127, -3.296923291, -3.291532041, -3.26406046, -3.31412441,
    -3.294191809, -3.276235742, -3.286492107, -3.224925591, -3.247366392,
    -3.226366932, -3.299300956, -3.271848391, -3.286939556, -3.305763392,
    -3.326000882, -3.308984472, -3.325790943, -3.364180261, -3.37596119,
    -3.302989661, -3.289698303, -3.293136385, -3.332065843, -3.301782705,
    -3.248417399, -3.326449519, -3.274718704, -3.350343172, -3.322087938,
    -3.323831142, -3.377516199, -3.309026811, -3.316590811, -3.343385489,
    -3.310500955, -3.326804191, -3.314359185, -3.332586271, -3.372610054,
    -3.356011557, -3.328690815, -3.280414205, -3.384877155, -3.346372044,
    -3.390288348, -3.34326014, -3.310919711, -3.31920912, -3.289167521,
    -3.355185857, -3.350513164, -3.41896896, -3.348875899, -3.331092607,
    -3.362435603, -3.333222177, -3.349453836, -3.353786993, -3.347172035,
    -3.383333337, -3.399434305, -3.303233881, -3.410545428, -3.367054646,
    -3.341802818, -3.324625524, -3.404753471, -3.389642462, -3.321200769,
    -3.298613965, -3.305837463, -3.354759642, -3.324537324, -3.399538267,
    -3.374292945, -3.42551362, -3.345201441, -3.329554232, -3.393365266,
    -3.360268813, -3.385387424, -3.381840029, -3.33758618, -3.357133299,
    -3.396517837, -3.38749666, -3.36680567, -3.448380699, -3.399501019,
    -3.40130488, -3.358094788, -3.374493082, -3.408125501, -3.376882576,
    -3.316364519, -3.375598031, -3.358521408, -3.393219803, -3.366530285,
    -3.351379125, -3.403608031, -3.418187838, -3.352161076, -3.419494277
]

df = pd.DataFrame({"U_t": U_t, "Energy": Energy})

# ==========================================================
# Scaling
# ==========================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

U_scaled = scaler_X.fit_transform(df[["U_t"]].values)
E_scaled = scaler_y.fit_transform(df[["Energy"]].values)

# ==========================================================
# Graph Builder
# ==========================================================
def build_graph(U):
    feats = []
    for i in range(L):
        f = [U]
        if use_pos_encoding:
            ang = 2*np.pi*i/L
            f += [np.sin(ang), np.cos(ang)]
        feats.append(f)

    x = torch.tensor(feats, dtype=torch.float)
    edges = [(i, i+1) for i in range(L-1)] + [(i+1, i) for i in range(L-1)]
    if periodic:
        edges += [(0, L-1), (L-1, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return Data(x=x, edge_index=edge_index)

# ==========================================================
# Custom Weighted Dataset
# ==========================================================
class HubbardDatasetWeighted(Dataset):
    def __init__(self, U, E, edge_points=None, weight_factor=5.0):
        super().__init__()
        self.U = U.flatten()
        self.E = E.flatten()
        self.edge_points = edge_points if edge_points is not None else []
        self.weights = np.ones(len(self.U))
        for i, u in enumerate(self.U):
            # edge points weight বৃদ্ধি
            if u in self.edge_points:
                self.weights[i] = weight_factor

    def len(self):
        return len(self.U)

    def get(self, idx):
        g = build_graph(self.U[idx])
        g.y = torch.tensor([self.E[idx]], dtype=torch.float)
        g.weight = torch.tensor([self.weights[idx]], dtype=torch.float)
        return g

# ==========================================================
# Load Dataset
# ==========================================================
edge_U = [11.9, 12.0]  # high-U points
dataset = HubbardDatasetWeighted(U_scaled, E_scaled, edge_points=edge_U, weight_factor=edge_weight_factor)

# Train/Validation Split
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==========================================================
# GNN Model
# ==========================================================
class GNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = GraphConv(in_ch, 64)
        self.conv2 = GraphConv(64, 64)
        self.conv3 = GraphConv(64, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, ei))
        x = torch.relu(self.conv2(x, ei))
        x = torch.relu(self.conv3(x, ei))
        x = global_mean_pool(x, b)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).view(-1)

model = GNN(in_ch=dataset[0].x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ==========================================================
# Training Loop with Weighted Loss
# ==========================================================
train_losses, val_losses = [], []

for epoch in range(1, epochs+1):
    # -------------------------
    # Training
    # -------------------------
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        # Weighted MSE
        loss = (batch.weight.view(-1) * (pred - batch.y.view(-1))**2).mean()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch.num_graphs
    avg_train_loss = total_train_loss / len(train_dataset)
    train_losses.append(avg_train_loss)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = (batch.weight.view(-1) * (pred - batch.y.view(-1))**2).mean()
            total_val_loss += loss.item() * batch.num_graphs
    avg_val_loss = total_val_loss / len(val_dataset)
    val_losses.append(avg_val_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Train Weighted MSE = {avg_train_loss:.6f} | Val Weighted MSE = {avg_val_loss:.6f}")

# ==========================================================
# Plot Training/Validation Loss
# ==========================================================
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, label='Train Weighted MSE')
plt.plot(range(1, epochs+1), val_losses, label='Validation Weighted MSE')
plt.xlabel('Epoch')
plt.ylabel('Weighted MSE')
plt.title('Training vs Validation Weighted MSE')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================================
# Prediction on Full Dataset
# ==========================================================
model.eval()
all_preds = []
with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=batch_size):
        batch = batch.to(device)
        pred = model(batch)
        all_preds.append(pred.cpu())

all_preds = torch.cat(all_preds).numpy()
Energy_pred = scaler_y.inverse_transform(all_preds.reshape(-1,1)).flatten()
Energy_true = scaler_y.inverse_transform(E_scaled).flatten()

df_results = pd.DataFrame({
    "U_t": df["U_t"],
    "Energy_true": Energy_true,
    "Energy_pred": Energy_pred
})

print("\nPredictions vs True Energy:")
print(df_results)

# ==========================================================
# Plot Predictions
# ==========================================================
plt.figure(figsize=(8,5))
plt.plot(df["U_t"], Energy_true, 'o-', label='True Energy')
plt.plot(df["U_t"], Energy_pred, 's--', label='Predicted Energy')
plt.xlabel('U/t')
plt.ylabel('Ground-state Energy per site [a.u.]')
plt.title('Hubbard Chain: True vs Predicted Energy')
plt.legend()
plt.grid(True)
plt.show()

