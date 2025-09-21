# =============================================================================
# IMPORTANT NOTE ON DEPENDENCIES
# =============================================================================
# To run this script, the following Python libraries must be installed
# in your environment. You can install them using pip:
#
# pip install numpy
# pip install matplotlib
#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# PART 1: SIMULATION CORE
# =============================================================================

def setup_grid_and_maps(grid_size):
    """
    Creates the grid structure (nodes, edges) and the necessary
    mapping dictionaries to map coordinates to array indices.
    """
    nodes = []
    node_map = {}
    idx = 0
    for y in range(grid_size):
        for x in range(grid_size):
            nodes.append((x, y))
            node_map[(x, y)] = idx
            idx += 1

    edges = []
    # This loop structure ensures a consistent ordering for edges
    for y in range(grid_size):
        for x in range(grid_size):
            if x < grid_size - 1:
                edges.append(((x, y), (x + 1, y)))
            if y < grid_size - 1:
                edges.append(((x, y), (x, y + 1)))
            
    return nodes, edges, node_map

def run_simulation(grid_size=10, alpha=1.6, beta=1.0, A=0.5, omega=0.2, dt=0.05, steps=400, freeze_step=250, static=False):
    """
    Runs the simulation of the coupled differential equations.
    """
    if static:
        A, omega, freeze_step = 0, 0, None

    nodes, edges, node_map = setup_grid_and_maps(grid_size)
    num_nodes, num_edges = len(nodes), len(edges)

    Phi = np.zeros(num_nodes)
    c = np.full(num_edges, 0.1)
    source_idx = node_map[(0, 0)]
    sink_idx = node_map[(grid_size - 1, grid_size - 1)]
    c_history = np.zeros((steps, num_edges))
    frozen_edges_indices = set()

    for t in range(steps):
        time = t * dt
        Phi[source_idx] = 1.0 + A * np.sin(omega * time)
        Phi[sink_idx] = 0.0

        dPhi_dt = np.zeros(num_nodes)
        for i, (u, v) in enumerate(edges):
            u_idx, v_idx = node_map[u], node_map[v]
            flux = c[i] * (Phi[v_idx] - Phi[u_idx])
            dPhi_dt[u_idx] += flux
            dPhi_dt[v_idx] -= flux
        Phi += dPhi_dt * dt

        dc_dt = np.zeros(num_edges)
        for i in range(num_edges):
            if i not in frozen_edges_indices:
                u, v = edges[i]
                u_idx, v_idx = node_map[u], node_map[v]
                potential_diff = np.abs(Phi[u_idx] - Phi[v_idx])
                dc_dt[i] = alpha * potential_diff - beta * c[i]
        c += dc_dt * dt
        c = np.maximum(c, 0.01)

        if t == freeze_step and freeze_step is not None:
            strongest_edge_idx = np.argmax(c)
            frozen_edges_indices.add(strongest_edge_idx)

        if freeze_step is not None and t > freeze_step + 50:
            A = 0
            Phi[source_idx] = 0
        
        c_history[t] = c.copy()

    return nodes, edges, c_history, frozen_edges_indices

# =============================================================================
# PART 2: VISUALIZATION
# =============================================================================

def create_and_save_plots():
    """
    Main function that starts simulations and generates all five figures.
    """
    print("Starting simulations for the figures...")
    grid_size = 10
    steps_static = 200
    steps_dynamic = 400
    
    nodes, edges, c_history_dynamic, frozen_indices = run_simulation(grid_size=grid_size, steps=steps_dynamic, freeze_step=250)
    _, _, c_history_static, _ = run_simulation(grid_size=grid_size, steps=steps_static, static=True)
    
    x_coords, y_coords = zip(*nodes)
    source_coords = (0, 0)
    sink_coords = (grid_size - 1, grid_size - 1)

    # --- Figure 1: Static Emergence ---
    print("Creating Figure 1...")
    fig1, axes1 = plt.subplots(1, 4, figsize=(22, 5.5), facecolor='white')
    steps_to_plot = [0, 50, 100, 199]
    for i, step in enumerate(steps_to_plot):
        ax = axes1[i]
        c_vals = c_history_static[step]
        max_c = c_vals.max() if c_vals.max() > 0 else 1.0
        for j, edge in enumerate(edges):
            (x1, y1), (x2, y2) = edge
            lw = 0.5 + 5 * (c_vals[j] / max_c)
            alpha_val = 0.1 + 0.9 * (c_vals[j] / max_c)
            ax.plot([x1, x2], [-y1, -y2], color='black', linewidth=lw, alpha=alpha_val, zorder=1)
        ax.scatter(x_coords, [-y for y in y_coords], c='lightgray', s=30, zorder=2)
        ax.scatter(source_coords[0], -source_coords[1], c='#d62728', s=120, zorder=3)
        ax.scatter(sink_coords[0], -sink_coords[1], c='#1f77b4', s=120, zorder=3)
        ax.set_title(f'Timestep t = {step}', fontsize=14)
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig('figure_1_en.png', dpi=300, facecolor='white')
    plt.close(fig1)

    # --- Figure 2: Dynamic Heartbeat ---
    print("Creating Figure 2...")
    fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5.5), facecolor='white')
    steps_to_plot_dynamic = [150, 165, 180, 195]
    max_c_cycle = c_history_dynamic[150:200].max()
    for i, step in enumerate(steps_to_plot_dynamic):
        ax = axes2[i]
        c_vals = c_history_dynamic[step]
        for j, edge in enumerate(edges):
            (x1, y1), (x2, y2) = edge
            lw = 0.5 + 5 * (c_vals[j] / max_c_cycle)
            alpha_val = 0.1 + 0.9 * (c_vals[j] / max_c_cycle)
            ax.plot([x1, x2], [-y1, -y2], color='black', linewidth=lw, alpha=alpha_val, zorder=1)
        ax.scatter(x_coords, [-y for y in y_coords], c='lightgray', s=30, zorder=2)
        ax.scatter(source_coords[0], -source_coords[1], c='#d62728', s=120, zorder=3)
        ax.scatter(sink_coords[0], -sink_coords[1], c='#1f77b4', s=120, zorder=3)
        ax.set_title(f'Timestep t = {step} (Pulsating)', fontsize=14)
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig('figure_2_en.png', dpi=300, facecolor='white')
    plt.close(fig2)

    # --- Figure 3: The Act of Creation ---
    print("Creating Figure 3...")
    fig3, axes3 = plt.subplots(1, 4, figsize=(22, 5.5), facecolor='white')
    steps_to_plot_freeze = [249, 251, 300, 399]
    titles_freeze = ["t=249 (Before Freeze)", "t=251 (After Freeze)", "t=300 (Pulsating with Memory)", "t=399 (Memory Persists)"]
    max_c_freeze = c_history_dynamic[240:300].max()
    frozen_color = '#FF007F' # Rose pink for high visibility
    for i, step in enumerate(steps_to_plot_freeze):
        ax = axes3[i]
        c_vals = c_history_dynamic[step]
        for j, edge in enumerate(edges):
            (x1, y1), (x2, y2) = edge
            lw = 0.5 + 5 * (c_vals[j] / max_c_freeze)
            alpha_val = 0.1 + 0.9 * (c_vals[j] / max_c_freeze)
            color = 'black'
            
            is_frozen = j in frozen_indices and step > 249
            
            if is_frozen:
                color = frozen_color
                lw = max(lw, 4.5) # Make the frozen edge substantially thicker
                alpha_val = 1.0     # Make it fully opaque
                
            z_order = 5 if is_frozen else 1
            
            ax.plot([x1, x2], [-y1, -y2], color=color, linewidth=lw, alpha=alpha_val, zorder=z_order)

        ax.scatter(x_coords, [-y for y in y_coords], c='lightgray', s=30, zorder=2)
        ax.scatter(source_coords[0], -source_coords[1], c='#d62728', s=120, zorder=3)
        ax.scatter(sink_coords[0], -sink_coords[1], c='#1f77b4', s=120, zorder=3)
        ax.set_title(titles_freeze[i], fontsize=14)
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig('figure_3_en.png', dpi=300, facecolor='white')
    plt.close(fig3)

    # --- Figure 4: Quantitative Analysis ---
    print("Creating Figure 4: Quantitative Analysis...")
    fig4, ax4 = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Heuristically find an edge that is part of the final path
    # For a grid starting at (0,0), an edge like ((0,0), (1,0)) is a good candidate.
    # This might need adjustment if the grid or source/sink changes.
    path_edge_idx = -1
    for i, edge in enumerate(edges):
        if (edge[0] == (0,0) and edge[1] == (1,0)) or (edge[0] == (1,0) and edge[1] == (0,0)):
           path_edge_idx = i
           break
    if path_edge_idx == -1: path_edge_idx = 0 # Fallback

    strongest_edge_history = c_history_static[:, path_edge_idx]
    
    other_edges_mask = np.ones(c_history_static.shape[1], dtype=bool)
    other_edges_mask[path_edge_idx] = False
    average_other_history = c_history_static[:, other_edges_mask].mean(axis=1)
    
    time_axis = np.arange(steps_static)
    
    ax4.plot(time_axis, strongest_edge_history, label='Conductivity of a Path Edge', color='#1f77b4', lw=2.5)
    ax4.plot(time_axis, average_other_history, label='Average Conductivity of Other Edges', color='#ff7f0e', lw=2.5, linestyle='--')
    
    ax4.set_xlabel('Simulation Timestep (t)', fontsize=14)
    ax4.set_ylabel('Conductivity (c)', fontsize=14)
    ax4.set_title('Quantitative Analysis of the "Winner-Takes-it-All" Effect', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=12)
    ax4.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax4.set_yscale('log')
    ax4.set_ylim(bottom=0.01)
    
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig('figure_4_en.png', dpi=300, facecolor='white')
    plt.close(fig4)


    # --- Figure 5: Genesis Experiment Schematic ---
    print("Creating Figure 5: Schematic of the Experiment...")
    fig5, ax5 = plt.subplots(figsize=(8, 6.5), facecolor='white')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 8)
    ax5.axis('off')
    box_style = dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='steelblue', lw=2)
    arrow_style = dict(facecolor='#9467bd', edgecolor='#9467bd', arrowstyle='->, head_width=0.4, head_length=0.8', lw=2.5)
    ax5.add_patch(patches.Rectangle((1, 1), 8, 1.2, color='lightsteelblue'))
    ax5.text(5, 1.6, 'Glass Substrate', ha='center', va='center', fontsize=12)
    for j in range(6):
        ax5.add_patch(patches.Rectangle((1.5 + j * 1.2, 2.2), 0.2, 0.8, color='gold'))
    ax5.text(5, 3.25, 'Micro-electrodes (Au/Ti)', ha='center', va='center', fontsize=12)
    ax5.add_patch(patches.Rectangle((1, 4), 8, 2, color='mediumseagreen', alpha=0.4))
    ax5.text(5, 5, 'Active Medium (e.g., Hydrogel)', ha='center', va='center', bbox=box_style, fontsize=12)
    ax5.add_patch(patches.Rectangle((3, 0.5), 4, 0.5, color='slategray'))
    ax5.text(5, 0.75, 'Piezo Transducer (Floquet Drive)', ha='center', va='center', color='white', fontsize=12)
    ax5.annotate('UV Laser ("Freeze" Operator)', xy=(5, 4.5), xytext=(5, 7.5),
                arrowprops=arrow_style, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='lavender', ec='#9467bd', lw=1.5), fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig('figure_5_en.png', dpi=300, facecolor='white')
    plt.close(fig5)

    print("All five figures have been successfully generated and saved.")

if __name__ == '__main__':
    create_and_save_plots()

