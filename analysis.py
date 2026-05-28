"""
Social Network Conservation Analysis
=====================================
Analyzes conservation of "engagement level" on social network graphs
using the graph Laplacian and tension-based conservation ratios.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

np.random.seed(42)

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Helpers ──────────────────────────────────────────────────────────────────

def build_sbm(n_communities=5, size=20, p_in=0.3, p_out=0.05, seed=42):
    """Build a stochastic block model network with engagement attributes."""
    sizes = [size] * n_communities
    probs = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    # Assign engagement: community-specific normal distributions
    community_means = np.linspace(0.3, 0.9, n_communities)
    for node in G.nodes():
        block = G.nodes[node]['block']
        G.nodes[node]['engagement'] = np.clip(
            np.random.normal(community_means[block], 0.1), 0, 1
        )
        G.nodes[node]['community'] = block
    return G


def laplacian_and_fields(G):
    """Return Laplacian L, eigenvalues, eigenvectors, and attribute vector."""
    A = nx.to_numpy_array(G)
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    attr = np.array([G.nodes[n].get('engagement', 0.5) for n in G.nodes()])
    return L, eigenvalues, eigenvectors, attr


def conservation_ratio(L, attr):
    """
    Conservation ratio: ratio of Dirichlet energy (attribute variation across edges)
    to the total attribute energy. Lower ratio → more conserved (smoother).
    
    We define conservation = 1 - (dirichlet_energy / total_energy).
    High conservation ≈ attribute is smooth on the graph.
    """
    attr = attr.reshape(-1, 1)
    total_energy = float(attr.T @ attr)
    dirichlet = float(attr.T @ L @ attr)
    if total_energy < 1e-12:
        return 1.0
    ratio = dirichlet / total_energy
    return 1.0 - ratio  # higher = more conserved


def per_community_conservation(G, L, attr):
    """Compute conservation per community."""
    results = {}
    nodes = list(G.nodes())
    for c in sorted(set(G.nodes[n]['community'] for n in nodes)):
        members = [n for n in nodes if G.nodes[n]['community'] == c]
        idx = [list(G.nodes()).index(n) for n in members]
        # Subgraph Laplacian approximation: use diagonal blocks
        sub_attr = attr[idx]
        sub_L = L[np.ix_(idx, idx)]
        results[c] = conservation_ratio(sub_L, sub_attr)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE SOCIAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("1. BASELINE SOCIAL NETWORK — Community Health")
print("=" * 70)

G = build_sbm()
L, evals, evecs, attr = laplacian_and_fields(G)
overall_cons = conservation_ratio(L, attr)
print(f"Overall conservation: {overall_cons:.4f}")
print(f"Eigenvalues (first 6): {evals[:6].round(4)}")
print(f"Fiedler value (λ₂): {evals[1]:.4f}")

comm_cons = per_community_conservation(G, L, attr)
print("\nPer-community conservation:")
for c, val in comm_cons.items():
    label = "✓ HEALTHY" if val > 0.7 else "⚠ FRAGMENTED"
    print(f"  Community {c}: {val:.4f}  {label}")

# ── Plot 1: Network colored by conservation ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

pos = nx.spring_layout(G, seed=42)

# Color by community
colors_comm = [G.nodes[n]['community'] for n in G.nodes()]
nx.draw(G, pos, ax=axes[0], node_color=colors_comm, cmap='tab10',
        node_size=60, edge_color='#cccccc', width=0.3, with_labels=False)
axes[0].set_title("Network — Colored by Community", fontsize=13)

# Color by engagement
eng_colors = [G.nodes[n]['engagement'] for n in G.nodes()]
sc = nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=eng_colors,
                             cmap='RdYlGn', node_size=60)
nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color='#cccccc', width=0.3)
axes[1].set_title(f"Network — Colored by Engagement (Cons={overall_cons:.3f})", fontsize=13)
plt.colorbar(sc, ax=axes[1], label='Engagement')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "1_baseline_network.png"), dpi=150)
plt.close()
print("\n[Saved] 1_baseline_network.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ECHO CHAMBER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2. ECHO CHAMBER DETECTION")
print("=" * 70)

G2 = build_sbm(seed=123)
# Make community 0 an echo chamber: remove all external edges
nodes_c0 = [n for n in G2.nodes() if G2.nodes[n]['community'] == 0]
edges_to_remove = [(u, v) for u, v in G2.edges()
                   if (u in nodes_c0) != (v in nodes_c0)]
G2.remove_edges_from(edges_to_remove)
print(f"Removed {len(edges_to_remove)} edges to isolate community 0 (echo chamber)")

L2, evals2, evecs2, attr2 = laplacian_and_fields(G2)
comm_cons2 = per_community_conservation(G2, L2, attr2)

print("\nPer-community conservation (with echo chamber at community 0):")
for c, val in comm_cons2.items():
    tag = "🔒 ECHO CHAMBER" if c == 0 else ""
    label = "✓ HEALTHY" if val > 0.7 else "⚠ FRAGMENTED" if val < 0.5 else "→ MODERATE"
    print(f"  Community {c}: {val:.4f}  {label}  {tag}")

echo_cons = comm_cons2[0]
other_cons = np.mean([v for k, v in comm_cons2.items() if k != 0])
print(f"\nEcho chamber conservation: {echo_cons:.4f}")
print(f"Other communities avg:    {other_cons:.4f}")
print(f"→ Echo chamber has {'higher' if echo_cons > other_cons else 'lower'} conservation")
print("  (High conservation in an isolated community = homogeneity / echo chamber)")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
communities = sorted(comm_cons2.keys())
vals = [comm_cons2[c] for c in communities]
colors = ['#d32f2f' if c == 0 else '#4caf50' for c in communities]
bars = ax.bar(communities, vals, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=0.7, color='orange', linestyle='--', label='Healthy threshold')
ax.set_xlabel("Community")
ax.set_ylabel("Conservation Ratio")
ax.set_title("Echo Chamber Detection via Conservation")
ax.legend()
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "2_echo_chamber.png"), dpi=150)
plt.close()
print("[Saved] 2_echo_chamber.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. INFLUENCE PROPAGATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3. INFLUENCE PROPAGATION")
print("=" * 70)

G3 = build_sbm(seed=77)
nodes_list = list(G3.nodes())
# Seed node 0 with high engagement
seed_node = nodes_list[0]
G3.nodes[seed_node]['engagement'] = 1.0
print(f"Seeded node {seed_node} with engagement = 1.0")

conservation_over_time = []
attr_over_time = []
steps = 20
alpha = 0.1  # propagation rate

attr3 = np.array([G3.nodes[n]['engagement'] for n in nodes_list])
A3 = nx.to_numpy_array(G3)
D3 = np.diag(A3.sum(axis=1))
L3 = D3 - A3

for t in range(steps):
    # Diffusion step: dX/dt = -alpha * L @ X
    attr3 = attr3 - alpha * (L3 @ attr3) * 0.05
    attr3 = np.clip(attr3, 0, 1)
    c = conservation_ratio(L3, attr3)
    conservation_over_time.append(c)
    attr_over_time.append(attr3.copy())

print(f"Conservation: t=0 → {conservation_over_time[0]:.4f}, "
      f"t={steps-1} → {conservation_over_time[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(steps), conservation_over_time, 'b-o', markersize=4)
axes[0].set_xlabel("Propagation Step")
axes[0].set_ylabel("Conservation Ratio")
axes[0].set_title("Conservation During Influence Propagation")
axes[0].grid(True, alpha=0.3)

# Show how engagement spreads from seed
seed_idx = 0
neighbor_indices = list(G3.neighbors(seed_node))
neighbor_idx = [nodes_list.index(n) for n in neighbor_indices]
far_nodes = [i for i in range(len(nodes_list)) if i not in neighbor_idx and i != seed_idx]

axes[1].plot(range(steps), [a[seed_idx] for a in attr_over_time], 'r-', linewidth=2, label='Seed node')
if neighbor_idx:
    axes[1].plot(range(steps), [np.mean([a[i] for i in neighbor_idx]) for a in attr_over_time],
                 'g--', label='Seed neighbors (avg)')
if far_nodes:
    axes[1].plot(range(steps), [np.mean([a[i] for i in far_nodes[:20]]) for a in attr_over_time],
                 'b:', label='Distant nodes (avg)')
axes[1].set_xlabel("Propagation Step")
axes[1].set_ylabel("Engagement")
axes[1].set_title("Engagement Spread from Seed")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "3_influence_propagation.png"), dpi=150)
plt.close()
print("[Saved] 3_influence_propagation.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PLATFORM COMPARISON: Twitter vs Facebook
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("4. PLATFORM COMPARISON: Twitter-like vs Facebook-like")
print("=" * 70)

# Twitter: high between-community edges
G_twitter = build_sbm(p_in=0.3, p_out=0.15, seed=42)
L_tw, evals_tw, _, attr_tw = laplacian_and_fields(G_twitter)
cons_tw = conservation_ratio(L_tw, attr_tw)

# Facebook: low between-community edges
G_fb = build_sbm(p_in=0.3, p_out=0.02, seed=42)
L_fb, evals_fb, _, attr_fb = laplacian_and_fields(G_fb)
cons_fb = conservation_ratio(L_fb, attr_fb)

print(f"Twitter-like  (p_out=0.15): Conservation = {cons_tw:.4f}, λ₂ = {evals_tw[1]:.4f}")
print(f"Facebook-like (p_out=0.02): Conservation = {cons_fb:.4f}, λ₂ = {evals_fb[1]:.4f}")
print(f"\n→ {'Twitter' if cons_tw > cons_fb else 'Facebook'} has higher conservation")
print("  Higher between-community connectivity → engagement diffuses more → smoother")

# Per-community comparison
cc_tw = per_community_conservation(G_twitter, L_tw, attr_tw)
cc_fb = per_community_conservation(G_fb, L_fb, attr_fb)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

communities = sorted(cc_tw.keys())
x = np.arange(len(communities))
width = 0.35

axes[0].bar(x - width/2, [cc_tw[c] for c in communities], width,
            label='Twitter (p_out=0.15)', color='#1da1f2', edgecolor='black', linewidth=0.5)
axes[0].bar(x + width/2, [cc_fb[c] for c in communities], width,
            label='Facebook (p_out=0.02)', color='#1877f2', edgecolor='black', linewidth=0.5)
axes[0].set_xlabel("Community")
axes[0].set_ylabel("Conservation Ratio")
axes[0].set_title("Per-Community Conservation: Twitter vs Facebook")
axes[0].set_xticks(x)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Overall comparison
axes[1].bar(['Twitter\n(p_out=0.15)', 'Facebook\n(p_out=0.02)'],
            [cons_tw, cons_fb],
            color=['#1da1f2', '#1877f2'], edgecolor='black', linewidth=0.5)
for i, v in enumerate([cons_tw, cons_fb]):
    axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=11)
axes[1].set_ylabel("Overall Conservation Ratio")
axes[1].set_title("Platform-Level Conservation")
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "4_platform_comparison.png"), dpi=150)
plt.close()
print("[Saved] 4_platform_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. BOT DETECTION via Fiedler Vector
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("5. BOT DETECTION — Fiedler Vector Analysis")
print("=" * 70)
print("Hypothesis: 'The Fiedler vector of the social network Laplacian")
print("            separates authentic from inauthentic users.'")

G5 = build_sbm(seed=99)
nodes5 = list(G5.nodes())
n_original = len(nodes5)

# Inject 10 bot nodes
n_bots = 10
for i in range(n_bots):
    bot_id = n_original + i
    G5.add_node(bot_id, community=-1, engagement=0.5, is_bot=True)
    # Bots connect randomly (not community-structured)
    targets = np.random.choice(nodes5, size=np.random.randint(2, 8), replace=False)
    for t in targets:
        G5.add_edge(bot_id, t)

print(f"Injected {n_bots} bot nodes (total: {G5.number_of_nodes()}, "
      f"edges: {G5.number_of_edges()})")

L5, evals5, evecs5, attr5 = laplacian_and_fields(G5)
fiedler = evecs5[:, 1]  # second smallest eigenvector

# Analysis: do bots cluster in Fiedler space?
is_bot = np.array([G5.nodes[n].get('is_bot', False) for n in G5.nodes()])
bot_fiedler = fiedler[is_bot]
human_fiedler = fiedler[~is_bot]

print(f"\nFiedler value (λ₂): {evals5[1]:.4f}")
print(f"Bot Fiedler values:    mean={bot_fiedler.mean():.4f}, std={bot_fiedler.std():.4f}")
print(f"Human Fiedler values:  mean={human_fiedler.mean():.4f}, std={human_fiedler.std():.4f}")

# Can we separate with a threshold?
all_fiedler = fiedler
threshold = np.median(fiedler)
bot_above = (bot_fiedler > threshold).sum()
human_above = (human_fiedler > threshold).sum()
bot_below = (bot_fiedler <= threshold).sum()
human_below = (human_fiedler <= threshold).sum()

print(f"\nMedian threshold split:")
print(f"  Above median: {bot_above} bots, {human_above} humans")
print(f"  Below median: {bot_below} bots, {human_below} humans")

# Try optimal separation
from itertools import combinations
best_sep = 0
best_thresh = 0
for t in np.linspace(fiedler.min(), fiedler.max(), 200):
    pred_bot = (fiedler > t)
    accuracy = max((pred_bot == is_bot).mean(), (pred_bot != is_bot).mean())
    if accuracy > best_sep:
        best_sep = accuracy
        best_thresh = t

print(f"\nBest Fiedler threshold separation accuracy: {best_sep:.2%}")
print(f"  (at threshold = {best_thresh:.4f})")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Fiedler histogram
axes[0].hist(human_fiedler, bins=20, alpha=0.7, label='Humans', color='#4caf50', edgecolor='black')
axes[0].hist(bot_fiedler, bins=10, alpha=0.7, label='Bots', color='#f44336', edgecolor='black')
axes[0].axvline(x=best_thresh, color='black', linestyle='--', label=f'Best threshold ({best_thresh:.3f})')
axes[0].set_xlabel("Fiedler Vector Value")
axes[0].set_ylabel("Count")
axes[0].set_title("Fiedler Vector: Humans vs Bots")
axes[0].legend()

# 2. Network colored by Fiedler partition
pos5 = nx.spring_layout(G5, seed=42, k=0.8)
node_colors_fiedler = ['red' if G5.nodes[n].get('is_bot') else
                        ('#1da1f2' if fiedler[i] > best_thresh else '#4caf50')
                        for i, n in enumerate(G5.nodes())]
node_shapes = ['^' if G5.nodes[n].get('is_bot') else 'o' for n in G5.nodes()]

# Draw humans
human_nodes = [n for n in G5.nodes() if not G5.nodes[n].get('is_bot')]
human_colors = [node_colors_fiedler[i] for i, n in enumerate(G5.nodes()) if not G5.nodes[n].get('is_bot')]
nx.draw_networkx_nodes(G5, pos5, ax=axes[1], nodelist=human_nodes,
                        node_color=human_colors, node_size=40, node_shape='o')
# Draw bots
bot_nodes = [n for n in G5.nodes() if G5.nodes[n].get('is_bot')]
bot_colors = [node_colors_fiedler[i] for i, n in enumerate(G5.nodes()) if G5.nodes[n].get('is_bot')]
nx.draw_networkx_nodes(G5, pos5, ax=axes[1], nodelist=bot_nodes,
                        node_color=bot_colors, node_size=80, node_shape='^')
nx.draw_networkx_edges(G5, pos5, ax=axes[1], edge_color='#ddd', width=0.3)
axes[1].set_title("Fiedler Partition (▲ = bots)")

# 3. Engagement vs Fiedler
colors = ['red' if b else '#4caf50' for b in is_bot]
axes[2].scatter(fiedler[~is_bot], attr5[~is_bot], c='#4caf50', s=20, label='Humans', alpha=0.6)
axes[2].scatter(fiedler[is_bot], attr5[is_bot], c='red', s=60, marker='^', label='Bots', alpha=0.8)
axes[2].axvline(x=best_thresh, color='black', linestyle='--', alpha=0.5)
axes[2].set_xlabel("Fiedler Vector Value")
axes[2].set_ylabel("Engagement")
axes[2].set_title("Engagement vs Fiedler Value")
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, "5_bot_detection.png"), dpi=150)
plt.close()
print("[Saved] 5_bot_detection.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
print(f"""
1. COMMUNITY HEALTH:
   - Overall conservation on baseline SBM: {overall_cons:.4f}
   - Healthy communities (high internal consistency) show high conservation.
   - Fragmented communities (mixed engagement) show low conservation.

2. ECHO CHAMBER DETECTION:
   - Isolated community conservation: {comm_cons2[0]:.4f}
   - Connected communities average:   {other_cons:.4f}
   - Echo chambers show {'higher' if comm_cons2[0] > other_cons else 'lower'} conservation due to homogeneity.
   - Conservation combined with isolation (low external edges) = echo chamber signal.

3. INFLUENCE PROPAGATION:
   - Conservation evolves from {conservation_over_time[0]:.4f} to {conservation_over_time[-1]:.4f}
   - As influence spreads, the graph smooths → conservation {'increases' if conservation_over_time[-1] > conservation_over_time[0] else 'changes'}.

4. PLATFORM COMPARISON:
   - Twitter-like  (p_out=0.15): {cons_tw:.4f}
   - Facebook-like (p_out=0.02): {cons_fb:.4f}
   - {'Twitter' if cons_tw > cons_fb else 'Facebook'} has higher conservation.
   - More cross-community edges → better mixing → smoother attributes.

5. BOT DETECTION (HYPOTHESIS TEST):
   - Best Fiedler separation accuracy: {best_sep:.2%}
   - Bot Fiedler mean:  {bot_fiedler.mean():.4f}
   - Human Fiedler mean: {human_fiedler.mean():.4f}
   - VERDICT: The Fiedler vector shows {'moderate' if best_sep > 0.7 else 'some' if best_sep > 0.6 else 'limited'} ability to separate bots from humans.
   - Bots (random connections, uniform engagement) do tend to cluster in Fiedler space,
     but separation is not perfect — especially with few bots.

All plots saved to: {OUT}/
""")
