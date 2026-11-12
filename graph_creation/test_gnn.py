import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Crea un grafo custom
# 4 nodi con 3 features ciascuno
x = torch.tensor([
    [1, 0, 1],  # Nodo 0
    [0, 1, 0],  # Nodo 1  
    [1, 1, 0],  # Nodo 2
    [0, 0, 1]   # Nodo 3
], dtype=torch.float)

# Archi: 0-1, 1-2, 2-3, 3-0
edge_index = torch.tensor([
    [0, 1, 2, 3],  # Nodi sorgente
    [1, 2, 3, 0]   # Nodi destinazione
], dtype=torch.long)

# Labels per classificazione
y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

# Crea oggetto Data
data = Data(x=x, edge_index=edge_index, y=y)

print(f'Grafo custom: {data}')
print(f'Numero di nodi: {data.num_nodes}')
print(f'Numero di archi: {data.num_edges}')

# Modello semplice
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training su grafo custom
model = SimpleGNN(3, 16, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

model.eval()
with torch.no_grad():
    # Start timer
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    starter.record()
    
    out = model(data)
    ender.record()

    pred = out.argmax(dim=1)
    accuracy = (pred == data.y).sum().item() / data.num_nodes
    
    
    # Wait for GPU to finish
    torch.cuda.synchronize()
    
    evaluation_time = starter.elapsed_time(ender)  # Time in milliseconds
    
    print(f'Accuracy finale: {accuracy:.4f}')
    print(f'Evaluation time: {evaluation_time:.2f} ms')