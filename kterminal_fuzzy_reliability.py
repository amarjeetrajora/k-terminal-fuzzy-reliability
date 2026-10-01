"""
K-Terminal Fuzzy Reliability of a 5-Node Network
=================================================
Based on: Chowdhury & Misra (1992) - "Evaluation of Fuzzy Reliability
of a Non-Series Parallel Network", Microelectron. Reliab., Vol 32, No 1/2.

Final VERSION:
- Uses proper Inclusion-Exclusion Principle (IEP) via alpha-cut level sets
- Handles shared-edge dependence between Steiner trees correctly
- Plots updated to reflect Rs support and core values
- Monte Carlo simulation added for validation of Rs interval
"""

import argparse
import itertools
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Output folder for figures and a global switch for interactive display.
OUTPUT_DIR = "results"
SHOW_PLOTS = True


# =============================================================================
# 1. FUZZY NUMBER REPRESENTATION
# =============================================================================

class TrapezoidalFuzzyNumber:
    """
    Trapezoidal fuzzy number (a1, a2, b2, b1).
    a1 = lower support, a2 = lower core, b2 = upper core, b1 = upper support
    """
    def __init__(self, a1, a2, b2, b1, name=""):
        assert 0 <= a1 <= a2 <= b2 <= b1 <= 1, (
            f"Invalid params ({a1},{a2},{b2},{b1})")
        self.a1 = a1
        self.a2 = a2
        self.b2 = b2
        self.b1 = b1
        self.name = name

    def membership(self, p):
        a1, a2, b2, b1 = self.a1, self.a2, self.b2, self.b1
        if p < a1 or p > b1:
            return 0.0
        elif a1 <= p < a2:
            return (p - a1) / (a2 - a1) if a2 > a1 else 1.0
        elif a2 <= p <= b2:
            return 1.0
        else:
            return (b1 - p) / (b1 - b2) if b1 > b2 else 1.0

    def multiply(self, other):
        return TrapezoidalFuzzyNumber(
            self.a1 * other.a1, self.a2 * other.a2,
            self.b2 * other.b2, self.b1 * other.b1,
            name=f"({self.name}*{other.name})")

    def get_alpha_cut(self, alpha):
        """Return [L(alpha), U(alpha)] crisp interval at given alpha level."""
        L = self.a1 + alpha * (self.a2 - self.a1)
        U = self.b1 - alpha * (self.b1 - self.b2)
        return L, U

    def __repr__(self):
        return (f"TFN(a1={self.a1:.4f}, a2={self.a2:.4f}, "
                f"b2={self.b2:.4f}, b1={self.b1:.4f})")

    def core_interval(self):
        return (self.a2, self.b2)

    def support_interval(self):
        return (self.a1, self.b1)


# =============================================================================
# 2. NETWORK DEFINITION
# =============================================================================

def build_network():
    edges = {
        'e1': (1, 2), 'e2': (1, 3), 'e3': (2, 4),
        'e4': (3, 4), 'e5': (2, 3), 'e6': (4, 5), 'e7': (1, 5),
    }
    fuzzy_probs = {
        'e1': TrapezoidalFuzzyNumber(0.70, 0.80, 0.90, 0.95, name='e1'),
        'e2': TrapezoidalFuzzyNumber(0.60, 0.75, 0.85, 0.92, name='e2'),
        'e3': TrapezoidalFuzzyNumber(0.65, 0.78, 0.88, 0.94, name='e3'),
        'e4': TrapezoidalFuzzyNumber(0.55, 0.70, 0.82, 0.90, name='e4'),
        'e5': TrapezoidalFuzzyNumber(0.72, 0.82, 0.91, 0.96, name='e5'),
        'e6': TrapezoidalFuzzyNumber(0.68, 0.79, 0.89, 0.95, name='e6'),
        'e7': TrapezoidalFuzzyNumber(0.50, 0.65, 0.80, 0.88, name='e7'),
    }
    terminals = {1, 3, 5}
    return edges, fuzzy_probs, terminals


# =============================================================================
# 3. GRAPH UTILITIES
# =============================================================================

def get_adjacency(edges, active_edges):
    adj = defaultdict(set)
    for e in active_edges:
        u, v = edges[e]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def terminals_connected(edges, active_edges, terminals):
    if not active_edges:
        return False
    adj = get_adjacency(edges, active_edges)
    start = next(iter(terminals))
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj[node] - visited)
    return terminals.issubset(visited)

def find_minimal_steiner_trees(edges, terminals):
    all_edges = list(edges.keys())
    steiner_trees = []
    for r in range(1, len(all_edges) + 1):
        for subset in itertools.combinations(all_edges, r):
            subset_set = set(subset)
            if terminals_connected(edges, subset_set, terminals):
                is_minimal = True
                for e in subset:
                    if terminals_connected(edges, subset_set - {e}, terminals):
                        is_minimal = False
                        break
                if is_minimal:
                    steiner_trees.append(frozenset(subset))
    # Sorted (by size, then edge names) so T1..T11 labels are reproducible
    # across runs (plain set ordering varies with Python hash randomisation).
    return sorted(set(steiner_trees), key=lambda t: (len(t), sorted(t)))


# =============================================================================
# 4.FUZZY RELIABILITY VIA ALPHA-CUT INCLUSION-EXCLUSION
#
# CORRECTION EXPLANATION:
# -------------------------
# The original code used:
#     Rs = 1 - PRODUCT( complement(P_Ti) )   for all trees Ti
#
# This formula assumes all Steiner tree failure events are INDEPENDENT.
# But trees share edges (e.g., T1 and T4 both contain e1, e4, e6),
# so their survival/failure probabilities are DEPENDENT.
#
# The correct formula is the Inclusion-Exclusion Principle (IEP):
#     Rs = SUM P(Si)
#          - SUM P(Si n Sj)
#          + SUM P(Si n Sj n Sk)
#          - ...
#
# where P(Si n Sj) = P(ALL edges in Ti UNION Tj survive)
# because Si AND Sj both surviving means every edge in both trees must work.
#
# Since direct fuzzy interval subtraction causes unbounded expansion,
# we apply IEP at each alpha-cut level (crisp arithmetic),
# then reconstruct the fuzzy number from the alpha-cut intervals.
# =============================================================================

def survival_prob_of_edge_set_at_alpha(edge_set, fuzzy_probs, alpha, bound):
    """
    Crisp probability that all edges in edge_set survive,
    evaluated at a given alpha-cut level and bound ('lower' or 'upper').
    """
    prob = 1.0
    for e in edge_set:
        L, U = fuzzy_probs[e].get_alpha_cut(alpha)
        prob *= (L if bound == 'lower' else U)
    return prob

def inclusion_exclusion_at_alpha(steiner_trees, fuzzy_probs, alpha, bound):
    """
    Correct IEP reliability at a single alpha-cut level.
    Rs(alpha, bound) = SUM_{S subset of trees, S nonempty}
                       (-1)^(|S|+1) * P(union edges of S survive)
    """
    M = len(steiner_trees)
    Rs_alpha = 0.0
    for size in range(1, M + 1):
        for combo in itertools.combinations(range(M), size):
            union_edges = set()
            for idx in combo:
                union_edges |= steiner_trees[idx]
            prob = survival_prob_of_edge_set_at_alpha(
                union_edges, fuzzy_probs, alpha, bound)
            sign = (-1) ** (size + 1)
            Rs_alpha += sign * prob
    return max(0.0, min(1.0, Rs_alpha))

def compute_kterminal_fuzzy_reliability_correct(steiner_trees, fuzzy_probs, num_alpha=11):
    """
    Compute K-terminal fuzzy system reliability using IEP
    over alpha-cut levels, then reconstruct the fuzzy number.
    """
    print("\n" + "="*65)
    print(" FUZZY RELIABILITY (IEP with Shared-Edge Dependence)")
    print("="*65)

    alphas = np.linspace(0, 1, num_alpha)
    lower_bounds = []
    upper_bounds = []

    print(f"\n  {'Alpha':>8} | {'L(alpha)':>12} | {'U(alpha)':>12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}")

    for alpha in alphas:
        L = inclusion_exclusion_at_alpha(steiner_trees, fuzzy_probs, alpha, 'lower')
        U = inclusion_exclusion_at_alpha(steiner_trees, fuzzy_probs, alpha, 'upper')
        lower_bounds.append(L)
        upper_bounds.append(U)
        print(f"  {alpha:>8.2f} | {L:>12.6f} | {U:>12.6f}")

    Rs_a1 = lower_bounds[0]
    Rs_a2 = lower_bounds[-1]
    Rs_b2 = upper_bounds[-1]
    Rs_b1 = upper_bounds[0]

    Rs = TrapezoidalFuzzyNumber(Rs_a1, Rs_a2, Rs_b2, Rs_b1, name="Rs")

    print(f"\n Rs Parameters:")
    print(f"  Support lower  a1 = {Rs_a1:.6f}")
    print(f"  Core lower     a2 = {Rs_a2:.6f}")
    print(f"  Core upper     b2 = {Rs_b2:.6f}")
    print(f"  Support upper  b1 = {Rs_b1:.6f}")
    print(f"\n  Interpretation:")
    print(f"  - Possible reliability range: [{Rs_a1:.4f}, {Rs_b1:.4f}]")
    print(f"  - 100% possibility core:      [{Rs_a2:.4f}, {Rs_b2:.4f}]")
    print(f"  - Best estimate (core mid):    {(Rs_a2 + Rs_b2)/2:.4f}")

    return Rs, alphas, lower_bounds, upper_bounds

def fuzzy_prob_of_tree(tree_edges, fuzzy_probs):
    """For plotting tree membership functions only."""
    edges_list = list(tree_edges)
    result = fuzzy_probs[edges_list[0]]
    for e in edges_list[1:]:
        result = result.multiply(fuzzy_probs[e])
    return result


# =============================================================================
# 5. MONTE CARLO VALIDATION
# =============================================================================

def monte_carlo_validate(edges, fuzzy_probs, terminals,
                         lower_bounds, upper_bounds, alphas,
                         N=100000, seed=42):
    """
    Monte Carlo simulation to validate the fuzzy Rs interval.

    For each alpha level:
      Step 1 — Sample each edge's crisp reliability uniformly from
               its alpha-cut interval [L_e(alpha), U_e(alpha)]
      Step 2 — Bernoulli trial: edge survives if random() < sampled probability
      Step 3 — BFS: check if all terminals {1,3,5} are mutually connected
      Step 4 — Compare simulated Rs against IEP bounds [L(alpha), U(alpha)]

    Parameters:
        N    : number of trials per alpha level (default 100,000)
        seed : random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    print("\n" + "="*65)
    print(" MONTE CARLO VALIDATION (N={:,} trials per alpha level)".format(N))
    print("="*65)
    print(f"\n  {'Alpha':>6} | {'Simulated Rs':>14} | "
          f"{'IEP Lower':>10} | {'IEP Upper':>10} | {'Within Bounds':>14}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}")

    sim_reliabilities = []
    within_bounds_flags = []

    for i, alpha in enumerate(alphas):
        successes = 0

        for _ in range(N):
            # Step 1: Sample crisp edge reliability from alpha-cut interval
            working_edges = set()
            for ename, fp in fuzzy_probs.items():
                L_e, U_e = fp.get_alpha_cut(alpha)
                p_sampled = random.uniform(L_e, U_e)

                # Step 2: Bernoulli trial
                if random.random() < p_sampled:
                    working_edges.add(ename)

            # Step 3: BFS connectivity check
            if terminals_connected(edges, working_edges, terminals):
                successes += 1

        simulated_Rs = successes / N
        sim_reliabilities.append(simulated_Rs)

        # Step 4: Compare against IEP bounds
        L_iep = lower_bounds[i]
        U_iep = upper_bounds[i]
        within = L_iep <= simulated_Rs <= U_iep
        within_bounds_flags.append(within)
        status = "YES  v" if within else "NO   x"
        print(f"  {alpha:>6.2f} | {simulated_Rs:>14.6f} | "
              f"{L_iep:>10.6f} | {U_iep:>10.6f} | {status:>14}")

    passed = sum(within_bounds_flags)
    total  = len(within_bounds_flags)
    print(f"\n  Validation Summary: {passed}/{total} alpha levels within IEP bounds")
    if passed == total:
        print("  RESULT: PASSED -- Monte Carlo confirms the fuzzy Rs interval.")
    else:
        print("  RESULT: PARTIAL -- Some alpha levels outside bounds. Review formula.")

    return sim_reliabilities


def plot_monte_carlo_validation(alphas, lower_bounds, upper_bounds,
                                sim_reliabilities,
                                save_path=os.path.join(OUTPUT_DIR, "kterminal_monte_carlo_validation.png")):
    """
    Plot IEP bounds vs Monte Carlo simulated reliability across alpha levels.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.fill_betweenx(alphas, lower_bounds, upper_bounds,
                     alpha=0.20, color='steelblue', label='IEP Fuzzy Rs interval')
    ax.plot(lower_bounds, alphas, 'b-', linewidth=2.2,
            label=f'IEP Lower L(a): [{min(lower_bounds):.4f} to {max(lower_bounds):.4f}]')
    ax.plot(upper_bounds, alphas, 'r-', linewidth=2.2,
            label=f'IEP Upper U(a): [{max(upper_bounds):.4f} to {min(upper_bounds):.4f}]')
    ax.plot(sim_reliabilities, alphas, 'go-', linewidth=1.8, markersize=7,
            label='Monte Carlo simulated Rs', zorder=5)

    for alpha, sim in zip(alphas, sim_reliabilities):
        ax.annotate(f'{sim:.4f}', xy=(sim, alpha),
                    xytext=(sim + 0.003, alpha + 0.025),
                    fontsize=7, color='darkgreen')

    ax.set_xlabel("System Reliability Rs", fontsize=11)
    ax.set_ylabel("Alpha Level (a)", fontsize=11)
    ax.set_title(
        "Monte Carlo Validation of Fuzzy Rs\n"
        "(Simulated points should lie within IEP shaded bounds)",
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.35)
    ax.set_ylim(-0.05, 1.15)
    x_min = max(0.0, min(lower_bounds) - 0.03)
    x_max = min(1.0, max(upper_bounds) + 0.03)
    ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Validation plot saved: {save_path}")
    _finish_figure()


def _finish_figure():
    """Show the figure interactively, or just close it when running headless."""
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# =============================================================================
# 6. VISUALISATION ( FOUR-PANEL PLOT)
# =============================================================================

def plot_results(edges, fuzzy_probs, terminals, steiner_trees,
                 tree_probs, Rs, alphas, lower_bounds, upper_bounds):
    """Four-panel plot with Rs and alpha-cut values."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Fuzzy Reliability of 5-Node K-Terminal Network\n"
        "(IEP -- Shared-Edge Dependence Accounted For)",
        fontsize=13, fontweight='bold'
    )

    p_vals = np.linspace(0, 1, 500)

    # Panel 1: Edge fuzzy probabilities
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(fuzzy_probs)))
    for (ename, fp), col in zip(fuzzy_probs.items(), colors):
        mu = [fp.membership(p) for p in p_vals]
        ax1.plot(p_vals, mu, label=ename, color=col, linewidth=1.8)
    ax1.set_xlabel("Probability p", fontsize=10)
    ax1.set_ylabel("Membership", fontsize=10)
    ax1.set_title("Fuzzy probabilities of edges", fontsize=11)
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_xlim(0, 1); ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Steiner tree probabilities
    ax2 = axes[0, 1]
    tree_colors = plt.cm.Set2(np.linspace(0, 1, len(steiner_trees)))
    for i, (tree, tp, col) in enumerate(zip(steiner_trees, tree_probs, tree_colors)):
        mu = [tp.membership(p) for p in p_vals]
        ax2.plot(p_vals, mu, label=f"T{i+1}", color=col, linewidth=1.8)
    ax2.set_xlabel("Probability p", fontsize=10)
    ax2.set_ylabel("Membership", fontsize=10)
    ax2.set_title("Fuzzy probability of each Steiner tree", fontsize=11)
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_xlim(0, 1); ax2.set_ylim(-0.05, 1.1)
    ax2.grid(True, alpha=0.3)

    # Panel 3:Rs membership function
    ax3 = axes[1, 0]
    x_lo = max(0.0, Rs.a1 - 0.03)
    x_hi = min(1.0, Rs.b1 + 0.02)
    p_zoom = np.linspace(x_lo, x_hi, 2000)
    mu_rs = [Rs.membership(p) for p in p_zoom]
    ax3.plot(p_zoom, mu_rs, color='navy', linewidth=2.5, label='Rs')
    ax3.fill_between(p_zoom, mu_rs, alpha=0.12, color='navy')
    ax3.axvline(Rs.a1, color='gray',  linestyle=':',  linewidth=1.2, label=f'a1={Rs.a1:.4f}')
    ax3.axvline(Rs.a2, color='green', linestyle='--', linewidth=1.5, label=f'a2={Rs.a2:.4f}')
    ax3.axvline(Rs.b2, color='red',   linestyle='--', linewidth=1.5, label=f'b2={Rs.b2:.4f}')
    ax3.axvline(Rs.b1, color='gray',  linestyle=':',  linewidth=1.2, label=f'b1={Rs.b1:.4f}')
    ax3.set_xlabel("System reliability p", fontsize=10)
    ax3.set_ylabel("Membership", fontsize=10)
    ax3.set_title("Fuzzy System Reliability Rs", fontsize=11)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.set_xlim(x_lo, x_hi); ax3.set_ylim(-0.05, 1.15)
    ax3.grid(True, alpha=0.4)
    ax3.annotate('', xy=(Rs.b1, -0.03), xytext=(Rs.a1, -0.03),
                 arrowprops=dict(arrowstyle='<->', color='navy', lw=1.5))
    ax3.text((Rs.a1+Rs.b1)/2, -0.047, f'Support [{Rs.a1:.4f}, {Rs.b1:.4f}]',
             ha='center', fontsize=7.5, color='navy')
    ax3.annotate('', xy=(Rs.b2, 1.08), xytext=(Rs.a2, 1.08),
                 arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
    ax3.text((Rs.a2+Rs.b2)/2, 1.10, f'Core [{Rs.a2:.4f}, {Rs.b2:.4f}]',
             ha='center', fontsize=7.5, color='darkgreen')

    # Panel 4: Alpha-cut analysis
    ax4 = axes[1, 1]
    x_min = max(0.0, min(lower_bounds)-0.02)
    x_max = min(1.0, max(upper_bounds)+0.02)
    ax4.fill_betweenx(alphas, lower_bounds, upper_bounds,
                      alpha=0.25, color='steelblue', label='Alpha-cut interval')
    ax4.plot(lower_bounds, alphas, 'b-', linewidth=2.2,
             label=f'L(a): [{min(lower_bounds):.4f} to {max(lower_bounds):.4f}]')
    ax4.plot(upper_bounds, alphas, 'r-', linewidth=2.2,
             label=f'U(a): [{min(upper_bounds):.4f} to {max(upper_bounds):.4f}]')
    for a_level, ls in [(0.0,':'),(0.5,'--'),(1.0,':')]:
        idx = int(round(a_level*(len(alphas)-1)))
        lb, ub = lower_bounds[idx], upper_bounds[idx]
        ax4.axhline(a_level, color='gray', linestyle=ls, linewidth=0.8, alpha=0.6)
        ax4.annotate(f'a={a_level:.1f}:[{lb:.4f},{ub:.4f}]',
                     xy=(lb, a_level), xytext=(x_min+0.01, a_level+0.06),
                     fontsize=7.5, color='dimgray',
                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax4.set_xlabel("System reliability", fontsize=10)
    ax4.set_ylabel("Alpha level", fontsize=10)
    ax4.set_title("Alpha-cut Analysis of Rs", fontsize=11)
    ax4.legend(fontsize=7.5, loc='upper left')
    ax4.set_xlim(x_min, x_max); ax4.set_ylim(-0.05, 1.15)
    ax4.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "kterminal_fuzzy_reliability.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {path}")
    _finish_figure()


# =============================================================================
# 7. NETWORK DIAGRAM
# =============================================================================

def plot_network(edges, terminals, steiner_trees):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    pos = {1:(2,4), 2:(1,2.5), 3:(3,2.5), 4:(2,1), 5:(2,3)}
    steiner_edge_set = set()
    for tree in steiner_trees: steiner_edge_set |= tree

    for ename, (u, v) in edges.items():
        x1,y1=pos[u]; x2,y2=pos[v]
        col = '#2196F3' if ename in steiner_edge_set else '#BDBDBD'
        lw  = 2.5       if ename in steiner_edge_set else 1.2
        ax.plot([x1,x2],[y1,y2],'-',color=col,linewidth=lw,zorder=1)
        ax.annotate(ename,((x1+x2)/2,(y1+y2)/2),fontsize=9,ha='center',
                    bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none'))

    for node,(x,y) in pos.items():
        col = '#FF5722' if node in terminals else '#78909C'
        ax.scatter(x,y,s=700,c=col,zorder=3,edgecolors='black',linewidths=1.5)
        ax.text(x,y,str(node),ha='center',va='center',
                fontsize=11,fontweight='bold',color='white',zorder=4)

    ax.legend(handles=[
        mpatches.Patch(color='#FF5722', label='Terminal node (K={1,3,5})'),
        mpatches.Patch(color='#78909C', label='Intermediate node'),
        mpatches.Patch(color='#2196F3', label='Edge in >=1 Steiner tree'),
    ], loc='lower right', fontsize=9)
    ax.set_title("5-Node K-Terminal Network\nTerminal set K = {1, 3, 5}",
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0,4); ax.set_ylim(0,5); ax.axis('off')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "kterminal_network_diagram.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Network diagram saved: {path}")
    _finish_figure()


# =============================================================================
# 8. MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="K-terminal fuzzy reliability (IEP + alpha-cuts + Monte Carlo).")
    parser.add_argument("--trials", type=int, default=100000,
                        help="Monte Carlo trials per alpha level (default: 100000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--no-show", action="store_true",
                        help="save figures without opening plot windows")
    parser.add_argument("--output-dir", default="results",
                        help="folder for saved figures (default: results)")
    return parser.parse_args()


def main():
    global OUTPUT_DIR, SHOW_PLOTS
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    SHOW_PLOTS = not args.no_show
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*65)
    print(" K-TERMINAL FUZZY RELIABILITY --IEP + MC VALIDATION")
    print("="*65)

    edges, fuzzy_probs, terminals = build_network()

    print(f"\nNetwork: 5 nodes, {len(edges)} edges")
    print(f"Terminal set K = {terminals}")
    print(f"\nEdge fuzzy probabilities (a1, a2, b2, b1):")
    for e, fp in fuzzy_probs.items():
        u, v = edges[e]
        print(f"  {e} ({u}-{v}): ({fp.a1}, {fp.a2}, {fp.b2}, {fp.b1})")

    print(f"\nFinding minimal Steiner trees for K = {terminals}...")
    steiner_trees = find_minimal_steiner_trees(edges, terminals)
    print(f"Found {len(steiner_trees)} minimal Steiner trees:")
    for i, tree in enumerate(steiner_trees):
        print(f"  T{i+1}: {{{', '.join(sorted(tree))}}}")

    # Tree probs for plotting Panel 2 only
    tree_probs = [fuzzy_prob_of_tree(t, fuzzy_probs) for t in steiner_trees]

    # Reliability via IEP
    Rs, alphas, lower_bounds, upper_bounds = compute_kterminal_fuzzy_reliability_correct(
        steiner_trees, fuzzy_probs, num_alpha=11)

    print("\n" + "="*65)
    print(" FINAL RESULT SUMMARY")
    print("="*65)
    print(f"\n  Rs = (a1={Rs.a1:.4f}, a2={Rs.a2:.4f}, b2={Rs.b2:.4f}, b1={Rs.b1:.4f})")
    print(f"  Support: [{Rs.a1:.4f}, {Rs.b1:.4f}]")
    print(f"  Core:    [{Rs.a2:.4f}, {Rs.b2:.4f}]")
    print(f"  Best estimate: {(Rs.a2+Rs.b2)/2:.4f}")

    # Monte Carlo validation
    sim_reliabilities = monte_carlo_validate(
        edges, fuzzy_probs, terminals,
        lower_bounds, upper_bounds, alphas,
        N=args.trials, seed=args.seed)

    # Plots
    print("\nGenerating plots...")
    plot_network(edges, terminals, steiner_trees)
    plot_results(edges, fuzzy_probs, terminals, steiner_trees,
                 tree_probs, Rs, alphas, lower_bounds, upper_bounds)
    plot_monte_carlo_validation(alphas, lower_bounds, upper_bounds,
                                sim_reliabilities,
                                save_path=os.path.join(OUTPUT_DIR,
                                                        "kterminal_monte_carlo_validation.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
