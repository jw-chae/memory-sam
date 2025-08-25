#!/usr/bin/env python3
"""
Memory SAM Pipeline Diagram Generator
Creates a professional pipeline diagram for the Memory SAM algorithm paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path

def create_memory_sam_pipeline():
    """Create the Memory SAM pipeline diagram for the paper"""
    
    # Set up the figure with high DPI for publication quality
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'feature': '#FFF2CC',
        'memory': '#D5E8D4',
        'matching': '#E1D5E7',
        'clustering': '#FFE6CC',
        'sam': '#F8CECC',
        'output': '#D5E8D4'
    }
    
    # Define box properties
    box_props = {
        'boxstyle': 'round,pad=0.1',
        'facecolor': 'white',
        'edgecolor': 'black',
        'linewidth': 1.5
    }
    
    # 1. Input Stage
    input_box = FancyBboxPatch((0.5, 4.5), 1.5, 0.8, **box_props)
    input_box.set_facecolor(colors['input'])
    ax.add_patch(input_box)
    ax.text(1.25, 4.9, 'Input Image\n$I \\in \\mathbb{R}^{H_0 \\times W_0 \\times 3}$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # 2. Feature Extraction Stage
    feature_box = FancyBboxPatch((2.5, 4.5), 1.5, 0.8, **box_props)
    feature_box.set_facecolor(colors['feature'])
    ax.add_patch(feature_box)
    ax.text(3.25, 4.9, 'DINOv3 Feature\nExtraction\n$F = \\Phi(I)$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # 3. Memory System
    memory_box = FancyBboxPatch((4.5, 4.5), 1.5, 0.8, **box_props)
    memory_box.set_facecolor(colors['memory'])
    ax.add_patch(memory_box)
    ax.text(5.25, 4.9, 'Memory System\n$\\{I^{(m)}, M^{(m)}, F^{(m)}\\}$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # 4. Sparse Matching
    matching_box = FancyBboxPatch((6.5, 4.5), 1.5, 0.8, **box_props)
    matching_box.set_facecolor(colors['matching'])
    ax.add_patch(matching_box)
    ax.text(7.25, 4.9, 'Sparse Matching\n$S_{ij} = \\tilde{X}_i \\cdot \\tilde{X}^{(m)}_j$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # 5. K-Means Clustering
    clustering_box = FancyBboxPatch((2.5, 3.0), 1.5, 0.8, **box_props)
    clustering_box.set_facecolor(colors['clustering'])
    ax.add_patch(clustering_box)
    ax.text(3.25, 3.4, 'K-Means\nClustering\n$\\hat{\\mathcal{P}}_{\\mathrm{fg}}$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # 6. SAM2 Segmentation
    sam_box = FancyBboxPatch((4.5, 3.0), 1.5, 0.8, **box_props)
    sam_box.set_facecolor(colors['sam'])
    ax.add_patch(sam_box)
    ax.text(5.25, 3.4, 'SAM2\nSegmentation\n$g(I, \\mathcal{U})$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # 7. Output
    output_box = FancyBboxPatch((6.5, 3.0), 1.5, 0.8, **box_props)
    output_box.set_facecolor(colors['output'])
    ax.add_patch(output_box)
    ax.text(7.25, 3.4, 'Output Mask\n$\\hat{M}$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # 8. Background Matching (Optional)
    bg_box = FancyBboxPatch((2.5, 1.5), 1.5, 0.8, **box_props)
    bg_box.set_facecolor(colors['matching'])
    ax.add_patch(bg_box)
    ax.text(3.25, 1.9, 'Background\nMatching\n$\\mathcal{M}_{\\mathrm{bg}}$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # 9. Coordinate Transformation
    coord_box = FancyBboxPatch((4.5, 1.5), 1.5, 0.8, **box_props)
    coord_box.set_facecolor(colors['feature'])
    ax.add_patch(coord_box)
    ax.text(5.25, 1.9, 'Coordinate\nTransformation\n$(u,v) \\rightarrow (x,y)$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # 10. Prompt Generation
    prompt_box = FancyBboxPatch((6.5, 1.5), 1.5, 0.8, **box_props)
    prompt_box.set_facecolor(colors['clustering'])
    ax.add_patch(prompt_box)
    ax.text(7.25, 1.9, 'Prompt\nGeneration\n$\\mathcal{U} = \\{(p,y_p)\\}$', 
             ha='center', va='center', fontsize=10, weight='bold')
    
    # Add arrows showing the main flow
    arrows = [
        # Main flow
        ((2.0, 4.9), (2.5, 4.9)),  # Input to Feature
        ((4.0, 4.9), (4.5, 4.9)),  # Feature to Memory
        ((6.0, 4.9), (6.5, 4.9)),  # Memory to Matching
        ((7.25, 4.7), (3.25, 3.8)),  # Matching to Clustering
        ((4.0, 3.4), (4.5, 3.4)),  # Clustering to SAM2
        ((6.0, 3.4), (6.5, 3.4)),  # SAM2 to Output
        
        # Background flow
        ((7.25, 4.3), (3.25, 2.3)),  # Matching to Background
        ((4.0, 2.3), (4.5, 2.3)),  # Background to Coordinate
        ((6.0, 2.3), (6.5, 2.3)),  # Coordinate to Prompt
        ((7.25, 2.3), (5.25, 3.8)),  # Prompt to SAM2
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    # Add mathematical notation boxes
    math_boxes = [
        (0.5, 0.5, 'Normalization: $\\tilde{X}_i = \\frac{X_i}{\\|X_i\\|_2 + \\varepsilon}$'),
        (3.0, 0.5, 'Foreground: $\\mathcal{I}_{\\mathrm{fg}} = \\{i \\mid M[u(i), v(i)]=1\\}$'),
        (5.5, 0.5, 'Similarity: $S_{ij} \\geq \\tau_{\\mathrm{fg}}$'),
        (8.0, 0.5, 'Selection: $\\ell^\\star = \\arg\\max_\\ell s_\\ell$')
    ]
    
    for x, y, text in math_boxes:
        math_box = FancyBboxPatch((x, y), 2.0, 0.4, 
                                 boxstyle='round,pad=0.05',
                                 facecolor='lightgray', 
                                 edgecolor='black', 
                                 linewidth=1)
        ax.add_patch(math_box)
        ax.text(x + 1.0, y + 0.2, text, ha='center', va='center', 
                fontsize=9, style='italic')
    
    # Add title
    ax.text(5.0, 5.8, 'Memory SAM: Sparse Keypoint-based Prompt Generation Pipeline', 
             ha='center', va='center', fontsize=16, weight='bold')
    
    # Add subtitle
    ax.text(5.0, 5.5, 'Memory-guided Segmentation via Sparse Feature Matching and K-Means Clustering', 
             ha='center', va='center', fontsize=12, style='italic')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Processing'),
        patches.Patch(color=colors['feature'], label='Feature Extraction'),
        patches.Patch(color=colors['memory'], label='Memory System'),
        patches.Patch(color=colors['matching'], label='Sparse Matching'),
        patches.Patch(color=colors['clustering'], label='Clustering'),
        patches.Patch(color=colors['sam'], label='SAM2 Model'),
        patches.Patch(color=colors['output'], label='Output Generation')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def create_detailed_flow_diagram():
    """Create a detailed flow diagram showing the mathematical pipeline"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define stages
    stages = [
        (1, 7, 'Input Image\n$I \\in \\mathbb{R}^{H_0 \\times W_0 \\times 3}$', 'lightblue'),
        (3, 7, 'Feature Extraction\n$F = \\Phi(I) \\in \\mathbb{R}^{C \\times H \\times W}$\n$X = \\mathrm{flat}(F) \\in \\mathbb{R}^{N \\times C}$', 'lightgreen'),
        (5, 7, 'Normalization\n$\\tilde{X}_i = \\frac{X_i}{\\|X_i\\|_2 + \\varepsilon}$', 'lightyellow'),
        (7, 7, 'Memory Search\n$\\{I^{(m)}, M^{(m)}, F^{(m)}\\}$', 'lightcoral'),
        (9, 7, 'Sparse Matching\n$S_{ij} = \\tilde{X}_i \\cdot \\tilde{X}^{(m)}_j$', 'lightpink'),
        (11, 7, 'Coordinate Mapping\n$(u,v) \\rightarrow (x,y)$', 'lightsteelblue'),
        (1, 5, 'Foreground Selection\n$\\mathcal{I}_{\\mathrm{fg}} = \\{i \\mid M[u(i), v(i)]=1\\}$', 'lightgreen'),
        (3, 5, 'Background Selection\n$\\mathcal{I}_{\\mathrm{bg}} = \\{i \\mid M[u(i), v(i)]=0\\}$', 'lightcoral'),
        (5, 5, 'Matching Filtering\n$\\mathcal{M}_{\\mathrm{fg}} = \\{(i,j) \\mid S_{ij} \\geq \\tau_{\\mathrm{fg}}\\}$', 'lightyellow'),
        (7, 5, 'Candidate Collection\n$\\mathcal{P}_{\\mathrm{fg}} = \\{(x_i, y_i)\\}$', 'lightblue'),
        (9, 5, 'K-Means Clustering\n$\\min \\sum_{i} \\|Z_i - \\mu_{c(i)}\\|_2^2$', 'lightpink'),
        (11, 5, 'Representative Points\n$\\hat{\\mathcal{P}}_{\\mathrm{fg}}$', 'lightgreen'),
        (1, 3, 'Prompt Construction\n$\\mathcal{U} = \\{(p, y_p)\\}$\n$y_p = 1$ for foreground', 'lightyellow'),
        (3, 3, 'SAM2 Input\n$g(I, \\mathcal{U})$', 'lightblue'),
        (5, 3, 'Mask Prediction\n$\\{\\hat{M}_\\ell\\}_{\\ell=1}^L$', 'lightgreen'),
        (7, 3, 'Score Calculation\n$s_\\ell \\in [0,1]$', 'lightcoral'),
        (9, 3, 'Best Mask Selection\n$\\ell^\\star = \\arg\\max_\\ell s_\\ell$', 'lightpink'),
        (11, 3, 'Final Output\n$\\hat{M} = \\hat{M}_{\\ell^\\star}$', 'lightsteelblue'),
        (1, 1, 'Memory Update\nStore $(I, \\hat{M}, F)$', 'lightyellow'),
        (3, 1, 'Quality Assessment\nIoU, Dice Score', 'lightblue'),
        (5, 1, 'Visualization\nOverlay, Sparse Matches', 'lightgreen'),
        (7, 1, 'Result Storage\nStructured Output', 'lightcoral'),
        (9, 1, 'Performance Metrics\nSpeed, Accuracy', 'lightpink'),
        (11, 1, 'System Optimization\nMemory, Computation', 'lightsteelblue')
    ]
    
    # Create stage boxes
    for x, y, text, color in stages:
        box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                            boxstyle='round,pad=0.05',
                            facecolor=color, 
                            edgecolor='black', 
                            linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=8, weight='bold')
    
    # Add flow arrows
    flow_arrows = [
        # Main horizontal flow
        ((1.4, 7), (2.6, 7)), ((3.4, 7), (4.6, 7)), ((5.4, 7), (6.6, 7)),
        ((7.4, 7), (8.6, 7)), ((9.4, 7), (10.6, 7)),
        
        # Vertical connections
        ((1, 6.7), (1, 5.3)), ((3, 6.7), (3, 5.3)), ((5, 6.7), (5, 5.3)),
        ((7, 6.7), (7, 5.3)), ((9, 6.7), (9, 5.3)), ((11, 6.7), (11, 5.3)),
        
        # Horizontal flow in middle row
        ((1.4, 5), (2.6, 5)), ((3.4, 5), (4.6, 5)), ((5.4, 5), (6.6, 5)),
        ((7.4, 5), (8.6, 5)), ((9.4, 5), (10.6, 5)),
        
        # Vertical connections to bottom
        ((1, 4.7), (1, 3.3)), ((3, 4.7), (3, 3.3)), ((5, 4.7), (5, 3.3)),
        ((7, 4.7), (7, 3.3)), ((9, 4.7), (9, 3.3)), ((11, 4.7), (11, 3.3)),
        
        # Bottom row flow
        ((1.4, 3), (2.6, 3)), ((3.4, 3), (4.6, 3)), ((5.4, 3), (6.6, 3)),
        ((7.4, 3), (8.6, 3)), ((9.4, 3), (10.6, 3)),
        
        # Final vertical connections
        ((1, 2.7), (1, 1.3)), ((3, 2.7), (3, 1.3)), ((5, 2.7), (5, 1.3)),
        ((7, 2.7), (7, 1.3)), ((9, 2.7), (9, 1.3)), ((11, 2.7), (11, 1.3)),
        
        # Bottom row horizontal flow
        ((1.4, 1), (2.6, 1)), ((3.4, 1), (4.6, 1)), ((5.4, 1), (6.6, 1)),
        ((7.4, 1), (8.6, 1)), ((9.4, 1), (10.6, 1))
    ]
    
    for start, end in flow_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc="black", lw=1.5)
        ax.add_patch(arrow)
    
    # Add mathematical formulas
    formulas = [
        (6, 0.5, 'Final Similarity: $\\mathrm{sim} = \\frac{1}{|\\mathcal{I}_{\\mathrm{fg}}|}\\sum_{i\\in\\mathcal{I}_{\\mathrm{fg}}}\\max_{j\\in\\mathcal{J}_{\\mathrm{fg}}} S_{ij}$'),
        (6, 0.2, 'Complexity: $O(|\\mathcal{I}|\\,|\\mathcal{J}|\\,C)$ for matching, $O(K T n)$ for K-Means')
    ]
    
    for x, y, text in formulas:
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", 
                                                      facecolor="white", 
                                                      edgecolor="black"))
    
    # Add title
    ax.text(6, 8.5, 'Memory SAM: Detailed Mathematical Pipeline Flow', 
             ha='center', va='center', fontsize=18, weight='bold')
    
    # Add subtitle
    ax.text(6, 8.2, 'Complete End-to-End Pipeline with Mathematical Formulations', 
             ha='center', va='center', fontsize=14, style='italic')
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the pipeline diagrams"""
    
    # Create output directory
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    # Generate main pipeline diagram
    print("Creating main pipeline diagram...")
    fig1 = create_memory_sam_pipeline()
    fig1.savefig(output_dir / "memory_sam_pipeline_main.png", 
                  dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'memory_sam_pipeline_main.png'}")
    
    # Generate detailed flow diagram
    print("Creating detailed flow diagram...")
    fig2 = create_detailed_flow_diagram()
    fig2.savefig(output_dir / "memory_sam_pipeline_detailed.png", 
                  dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'memory_sam_pipeline_detailed.png'}")
    
    # Generate high-resolution versions for publication
    print("Creating publication-quality versions...")
    fig1.savefig(output_dir / "memory_sam_pipeline_main.pdf", 
                  bbox_inches='tight', facecolor='white')
    fig2.savefig(output_dir / "memory_sam_pipeline_detailed.pdf", 
                  bbox_inches='tight', facecolor='white')
    print(f"Saved PDF versions for publication")
    
    plt.show()
    print("Pipeline diagrams generated successfully!")

if __name__ == "__main__":
    main()
