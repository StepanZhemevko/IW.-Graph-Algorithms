
from flask import Flask, render_template
import io
import base64
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from networkx.algorithms import clustering
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

@app.route('/')
def report():
    log = io.StringIO()
    plots = []

    # ============================== 📦 1. ЗАВАНТАЖЕННЯ ГРАФА ==============================
    edges = []
    with open("insecta-ant-colony1.edges", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))

    G = nx.Graph()
    G.add_edges_from(edges)

    # ============================== 📊 2.1 ОПИС ГРАФА ==============================
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / num_nodes
    clustering_coeffs = clustering(G)
    avg_clustering = np.mean(list(clustering_coeffs.values()))

    print("\n=== 2.1 ПОВНИЙ ОПИС ДАНИХ ===", file=log)
    print(f"📌 Вузлів (антів): {num_nodes}", file=log)
    print(f"📌 Ребер (взаємодій): {num_edges}", file=log)
    print("📌 Тип: неорієнтований, незважений граф без самопетель", file=log)
    print("📌 Джерело: Network Repository", file=log)
    print("📌 Кожен вузол — мураха, ребро — контакт між мурахами", file=log)
    print("📌 Ознаки вузлів: ступінь, коефіцієнт кластеризації", file=log)

    # ============================== 🧠 2.2 ОПИС ЗАДАЧІ ==============================
    print("\n=== 2.2 ЗАДАЧА МОДЕЛЮВАННЯ ===", file=log)
    print("🎯 Задача: класифікація вузлів на 'високий' і 'низький' ступінь", file=log)
    print("🧠 Мітки генеруються за медіаною ступеня (1 — високий, 0 — низький)", file=log)
    print("🏗 Модель отримає лише 2 ознаки: ступінь та кластеризацію", file=log)
    print("🧪 Мета: протестувати GCN та GraphSAGE для цього поділу", file=log)

    # ============================== 📐 2.4 ДОДАТКОВА СТАТИСТИКА ==============================
    density = nx.density(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    components = nx.number_connected_components(G)
    print("\n=== 2.4 СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ ===", file=log)
    print(f"🔗 Щільність графа: {density:.4f}", file=log)
    print(f"🔄 Асортативність: {assortativity:.4f}", file=log)
    print(f"🧩 Компонент зв'язності: {components}", file=log)
    if nx.is_connected(G):
        print(f"📏 Діаметр: {nx.diameter(G)}", file=log)
        print(f"🛣 Середній шлях: {nx.average_shortest_path_length(G):.2f}", file=log)
    else:
        print("📏 Діаметр: граф не є зв'язним", file=log)
        print("🛣 Середній шлях: граф не є зв'язним", file=log)

    # ============================== 🧪 3.1 ВИБІР ПІДМНОЖИНИ ДАНИХ ==============================
    print("\n=== 3.1 ПІДМНОЖИНА ДАНИХ ДЛЯ НАВЧАННЯ ===", file=log)
    subset_fraction = 0.8
    subset_nodes = np.random.choice(list(G.nodes()), size=int(subset_fraction * num_nodes), replace=False)
    G_sub = G.subgraph(subset_nodes).copy()
    degrees_sub = dict(G_sub.degree())
    clustering_sub = clustering(G_sub)
    node_features = pd.DataFrame({
        'degree': pd.Series(degrees_sub),
        'clustering': pd.Series(clustering_sub)
    })
    median_degree = node_features['degree'].median()
    node_features['label'] = (node_features['degree'] > median_degree).astype(int)
    print(f"✅ Підмножина використана: {len(G_sub.nodes())} вузлів (80% від графа)", file=log)

    G = G_sub
    X = node_features[['degree', 'clustering']].values
    y = node_features['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ============================== 🖼 3.2 ВІЗУАЛІЗАЦІЯ ==============================
    def plot_to_base64():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')



    plt.figure(figsize=(8, 5))
    sns.histplot(node_features['degree'], bins=20, kde=False, color='skyblue')
    plt.title("Degree Distribution")
    plots.append(plot_to_base64())

    plt.figure(figsize=(8, 5))
    sns.histplot(node_features['clustering'], bins=20, kde=False, color='lightgreen')
    plt.title("Clustering Coefficient Distribution")
    plots.append(plot_to_base64())

    # ============================== 📊 3.3 РОЗПОДІЛ МІТОК ==============================
    print("\n=== 3.3 РОЗПОДІЛ МІТОК ===", file=log)
    counts = node_features['label'].value_counts()
    print(f"{'Label':<10} | {'Count':>5}", file=log)
    print(f"{'-'*18}", file=log)
    for index, value in counts.items():
        print(f"{index:<10} | {value:>5}", file=log)

    # ============================== 🧠 4.1 ПІДГОТОВКА ДАНИХ ДЛЯ GNN ==============================
    print("\n=== 4.1 ПІДГОТОВКА ДАНИХ ДЛЯ GNN ===", file=log)
    edge_index = torch.tensor([(list(G.nodes()).index(u), list(G.nodes()).index(v)) for u, v in G.edges()], dtype=torch.long).t().contiguous()
    x = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    num_nodes = x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=0.3, random_state=42)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    data = Data(x=x, edge_index=edge_index, y=y_tensor, train_mask=train_mask, test_mask=test_mask)

    def train_and_evaluate(model_class, name, hidden_dim, lr):
        print(f"\n🔧 Training {name} with hidden_dim={hidden_dim}, lr={lr}", file=log)
        model = model_class(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"{name} Epoch {epoch}, Loss: {loss.item():.4f}", file=log)
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1).numpy()
        y_true = data.y.numpy()
        print(f"✅ {name} Accuracy: Train = {accuracy_score(y_true[train_mask], pred[train_mask]):.2f}, Test = {accuracy_score(y_true[test_mask], pred[test_mask]):.2f}", file=log)

        report_dict = classification_report(y_true[test_mask], pred[test_mask], output_dict=True)
        print(f"\nРезультати класифікації ({name}):", file=log)
        print(f"{'Class':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}", file=log)
        print("-" * 40, file=log)
        for key in ['0', '1']:
            vals = report_dict[key]
            print(f"{key:<12} {vals['precision']:.2f}  {vals['recall']:.2f}  {vals['f1-score']:.2f}  {int(vals['support']):>5}", file=log)
        print("-" * 40, file=log)
        print(f"{'accuracy':<12} {'':6} {'':6} {report_dict['accuracy']:.2f}  {sum(report_dict[k]['support'] for k in ['0','1']):>5}", file=log)
        print(f"{'macro avg':<12} {report_dict['macro avg']['precision']:.2f}  {report_dict['macro avg']['recall']:.2f}  {report_dict['macro avg']['f1-score']:.2f}  {int(report_dict['macro avg']['support']):>5}", file=log)
        print(f"{'weighted avg':<12} {report_dict['weighted avg']['precision']:.2f}  {report_dict['weighted avg']['recall']:.2f}  {report_dict['weighted avg']['f1-score']:.2f}  {int(report_dict['weighted avg']['support']):>5}", file=log)

        tsne = TSNE(n_components=2, random_state=42)
        tsne_proj = tsne.fit_transform(out.detach().numpy())
        plt.figure(figsize=(8, 6))
        plt.title(f"t-SNE Embedding: {name}")
        scatter = plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=y_true, cmap="coolwarm", alpha=0.7)
        plt.colorbar(scatter, label='True Label')
        plt.grid(True)
        plots.append(plot_to_base64())

    class GCN(torch.nn.Module):
        def __init__(self, hidden_dim=16):
            super().__init__()
            self.conv1 = GCNConv(2, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, 2)
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    train_and_evaluate(GCN, "GCN", hidden_dim=16, lr=0.01)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, hidden_dim=32):
            super().__init__()
            self.conv1 = SAGEConv(2, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, 2)
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    train_and_evaluate(GraphSAGE, "GraphSAGE", hidden_dim=32, lr=0.005)

    class GAT(torch.nn.Module):
        def __init__(self, hidden_dim=8):
            super().__init__()
            self.conv1 = GATConv(2, hidden_dim, heads=4, dropout=0.6)
            self.conv2 = GATConv(hidden_dim * 4, 2, heads=1, concat=False, dropout=0.6)
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    train_and_evaluate(GAT, "GAT", hidden_dim=8, lr=0.005)

    return render_template("report.html", log=log.getvalue(), images=plots)

if __name__ == "__main__":
    app.run(debug=False)