"""
Generate CN2VF-Net architecture visualization
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 26)
    ax.axis('off')
    
    # Title
    ax.text(5, 25, 'CN2VF-Net Architecture', 
            fontsize=24, fontweight='bold', ha='center')
    ax.text(5, 24.3, '1,264,759 Parameters | 448×448 Input', 
            fontsize=12, ha='center', style='italic', color='gray')
    
    y_pos = 23
    
    # Helper function to draw a box
    def draw_box(y, width, height, color, label, details='', text_color='white'):
        x = 5 - width/2
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(5, y + height/2 + 0.15, label,
               fontsize=12, fontweight='bold', ha='center', va='center', color=text_color)
        if details:
            ax.text(5, y + height/2 - 0.25, details,
                   fontsize=9, ha='center', va='center', color=text_color, style='italic')
    
    # Helper function to draw arrow
    def draw_arrow(y_start, y_end, label=''):
        arrow = FancyArrowPatch((5, y_start), (5, y_end),
                               arrowstyle='->', mutation_scale=30, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)
        if label:
            ax.text(5.5, (y_start + y_end)/2, label,
                   fontsize=8, ha='left', va='center', style='italic')
    
    # Input
    draw_box(y_pos, 4, 0.8, '#FF6B35', 'INPUT IMAGE', '448×448×3')
    y_pos -= 1.2
    draw_arrow(y_pos + 1.2, y_pos)
    
    # Stem
    draw_box(y_pos, 4, 0.8, '#667eea', 'STEM LAYER', 'Conv 3×3, stride=2 → 224×224×24')
    y_pos -= 1.2
    draw_arrow(y_pos + 1.2, y_pos)
    
    # CNN Stage 1
    draw_box(y_pos, 4, 1.0, '#f093fb', 'CNN STAGE 1', '3× InvertedResidual\n112×112×24 → 56×56×40')
    y_pos -= 1.4
    draw_arrow(y_pos + 1.4, y_pos)
    
    # CNN Stage 2
    draw_box(y_pos, 4, 1.0, '#f5576c', 'CNN STAGE 2', '3× InvertedResidual\n56×56×40 → 28×28×48')
    c2_y = y_pos  # Save for fusion
    y_pos -= 1.4
    draw_arrow(y_pos + 1.4, y_pos)
    
    # CNN Stage 3
    draw_box(y_pos, 4, 1.0, '#fa709a', 'CNN STAGE 3', '3× InvertedResidual\n28×28×48 → 28×28×80')
    c3_y = y_pos  # Save for fusion
    y_pos -= 1.4
    draw_arrow(y_pos + 1.4, y_pos)
    
    # Patch Embedding
    draw_box(y_pos, 4, 0.8, '#4facfe', 'PATCH EMBEDDING', 'Conv 1×1 → 784×128 tokens')
    y_pos -= 1.2
    draw_arrow(y_pos + 1.2, y_pos)
    
    # ViT Stage 1
    draw_box(y_pos, 4, 1.0, '#00f2fe', 'TRANSFORMER STAGE 1', '2× MHSA Blocks (4 heads)\n784 tokens × 128 dim')
    y_pos -= 1.4
    draw_arrow(y_pos + 1.4, y_pos)
    
    # Token Downsampling
    draw_box(y_pos, 4, 0.8, '#43e97b', 'TOKEN DOWNSAMPLING', 'Conv 3×3, stride=2 → 196×160 tokens')
    y_pos -= 1.2
    draw_arrow(y_pos + 1.2, y_pos)
    
    # ViT Stage 2
    draw_box(y_pos, 4, 1.0, '#38f9d7', 'TRANSFORMER STAGE 2', '2× MHSA Blocks (5 heads)\n196 tokens × 160 dim')
    t2_y = y_pos  # Save for fusion
    y_pos -= 1.4
    
    # Draw fusion arrows from c2, c3, t2
    fusion_y = y_pos
    # Arrow from t2
    draw_arrow(t2_y, fusion_y + 0.8)
    # Arrow from c3 (curved)
    ax.annotate('', xy=(5, fusion_y + 0.8), xytext=(7, c3_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black',
                               connectionstyle="arc3,rad=.3"))
    # Arrow from c2 (curved)
    ax.annotate('', xy=(5, fusion_y + 0.8), xytext=(7.5, c2_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black',
                               connectionstyle="arc3,rad=.5"))
    
    # Multi-Scale Fusion
    draw_box(y_pos, 4.5, 1.0, '#fee140', 'MULTI-SCALE FUSION', 
             'Concat (c2, c3, t2↑) + Conv\n28×28×224', text_color='black')
    y_pos -= 1.4
    draw_arrow(y_pos + 1.4, y_pos)
    
    # Detection Head
    draw_box(y_pos, 4, 1.2, '#764ba2', 'DETECTION HEAD', 
             'Global Avg Pool + FC Layers\nClassification (3) + BBox (4)')
    y_pos -= 1.6
    draw_arrow(y_pos + 1.6, y_pos)
    
    # Output - split into two boxes
    output_y = y_pos
    # Classification box
    draw_box(output_y, 2, 0.8, '#4CAF50', 'Classification', 
             'Fire/Smoke/Neutral')
    # BBox box
    draw_box(output_y, 2, 0.8, '#FF9800', 'Bounding Box', 
             '[x, y, w, h]')
    # Adjust positions
    ax.patches[-2].set_x(2)  # Move classification left
    ax.patches[-1].set_x(6)  # Move bbox right
    ax.texts[-4].set_x(3)
    ax.texts[-3].set_x(3)
    ax.texts[-2].set_x(7)
    ax.texts[-1].set_x(7)
    
    # Add legend
    legend_y = 0.5
    ax.text(1, legend_y + 1.2, 'Component Legend:', fontsize=10, fontweight='bold')
    
    components = [
        ('#FF6B35', 'Input/Output'),
        ('#667eea', 'Stem'),
        ('#f093fb', 'CNN Stages'),
        ('#4facfe', 'Patch Embed'),
        ('#00f2fe', 'Transformers'),
        ('#fee140', 'Fusion'),
        ('#764ba2', 'Detection'),
    ]
    
    for i, (color, name) in enumerate(components):
        y = legend_y - i * 0.25
        small_box = FancyBboxPatch((0.5, y), 0.3, 0.15,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor='black')
        ax.add_patch(small_box)
        ax.text(1, y + 0.075, name, fontsize=8, va='center')
    
    # Add parameter info box
    info_y = 1
    info_text = [
        'Total Parameters: 1,264,759',
        'CNN Params: ~500K',
        'ViT Params: ~600K',
        'Fusion + Head: ~163K'
    ]
    
    ax.text(7.5, info_y + 1.2, 'Parameter Distribution:', 
            fontsize=10, fontweight='bold')
    for i, text in enumerate(info_text):
        ax.text(7.5, info_y - i * 0.25, text, fontsize=8, family='monospace')
    
    plt.tight_layout()
    plt.savefig('cn2vf_net_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print("✓ Architecture diagram saved as: cn2vf_net_architecture.png")
    plt.show()

if __name__ == "__main__":
    create_architecture_diagram()
