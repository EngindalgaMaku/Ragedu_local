"""
Profesyonel RAG Altyapısı Görselleştirme
Python ile NetworkX ve Matplotlib kullanarak makale kalitesinde diagramlar
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

# Türkçe font desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300  # Yüksek çözünürlük

def create_architecture_diagram():
    """Genel Sistem Mimarisi Diagramı"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Renkler
    colors = {
        'frontend': '#E1F5FF',
        'gateway': '#FFF4E1',
        'aprag': '#E8F5E9',
        'doc': '#F3E5F5',
        'model': '#FFF3E0',
        'rerank': '#FCE4EC',
        'storage': '#E0F2F1',
        'external': '#F5F5F5'
    }
    
    # Frontend Layer
    frontend_box = FancyBboxPatch((1, 8.5), 2, 1, 
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['frontend'],
                                  edgecolor='black', linewidth=2)
    ax.add_patch(frontend_box)
    ax.text(2, 9, 'Frontend\nNext.js\n:3000', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # API Gateway
    gateway_box = FancyBboxPatch((4, 8.5), 2, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['gateway'],
                                 edgecolor='black', linewidth=2)
    ax.add_patch(gateway_box)
    ax.text(5, 9, 'API Gateway\nFastAPI\n:8000', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Core Services
    services = [
        ('APRAG\n:8007', 1, 6, colors['aprag']),
        ('Document\nProcessing\n:8080', 4, 6, colors['doc']),
        ('Model\nInference\n:8002', 7, 6, colors['model']),
        ('Reranker\n:8008', 1, 4, colors['rerank']),
        ('Auth\n:8006', 4, 4, colors['gateway']),
    ]
    
    for label, x, y, color in services:
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    # Storage
    storage_box = FancyBboxPatch((7, 4), 1.6, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['storage'],
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(storage_box)
    ax.text(7.8, 4.6, 'ChromaDB\n:8004', 
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Bağlantılar
    arrows = [
        # Frontend -> Gateway
        ((2, 8.5), (4, 9)),
        # Gateway -> Services
        ((5, 8.5), (1.8, 6.6)),
        ((5, 8.5), (4, 6.6)),
        ((5, 8.5), (4, 4.6)),
        # APRAG -> Document Processing
        ((1.8, 6), (3.2, 6)),
        # Document Processing -> Model
        ((5.6, 6), (6.2, 6)),
        # Document Processing -> Reranker
        ((4, 5.4), (1.8, 4.6)),
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', 
                               mutation_scale=20,
                               linewidth=1.5,
                               color='gray',
                               alpha=0.7)
        ax.add_patch(arrow)
    
    # Başlık
    ax.text(5, 9.8, 'RAG Education Assistant - System Architecture', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path('diagrams')
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'architecture_diagram.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'architecture_diagram.pdf', 
               bbox_inches='tight', facecolor='white')
    print("✅ Architecture diagram oluşturuldu!")
    plt.close()

def create_query_flow_diagram():
    """Query Flow Sequence Diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Aktörler (dikey)
    actors = [
        ('User', 1, 9),
        ('Frontend', 2.5, 9),
        ('Gateway', 4, 9),
        ('APRAG', 5.5, 9),
        ('DocProc', 7, 9),
        ('ChromaDB', 8.5, 9),
    ]
    
    # Aktör kutuları
    for name, x, y in actors:
        circle = Circle((x, y), 0.3, facecolor='lightblue', 
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', 
               fontsize=9, fontweight='bold')
        # Dikey çizgi
        ax.plot([x, x], [y-0.3, 1], 'k-', linewidth=1)
    
    # Adımlar (yatay oklar)
    steps = [
        (1, 'Query', 8.5, 2.5, 4),
        (2, 'Proxy', 7.5, 4, 5.5),
        (3, 'Topic\nClassify', 6.5, 5.5, 7),
        (4, 'Retrieve', 5.5, 7, 8.5),
        (5, 'Rerank', 4.5, 7, 5.5),
        (6, 'Generate', 3.5, 5.5, 4),
        (7, 'Response', 2.5, 4, 2.5),
    ]
    
    y_pos = 8
    for step_num, label, start_x, end_x, text_x in steps:
        # Ok
        arrow = FancyArrowPatch((start_x, y_pos), (end_x, y_pos),
                               arrowstyle='->', 
                               mutation_scale=15,
                               linewidth=2,
                               color='blue',
                               alpha=0.7)
        ax.add_patch(arrow)
        # Etiket
        ax.text(text_x, y_pos + 0.2, f'{step_num}. {label}', 
               ha='center', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='yellow', alpha=0.5))
        y_pos -= 1
    
    ax.text(5, 9.5, 'RAG Query Flow Sequence', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path('diagrams')
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'query_flow_diagram.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'query_flow_diagram.pdf', 
               bbox_inches='tight', facecolor='white')
    print("✅ Query flow diagram oluşturuldu!")
    plt.close()

def create_hybrid_retrieval_diagram():
    """Hybrid Retrieval System Diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Query
    query_box = FancyBboxPatch((3.5, 9), 3, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='#FFE0B2',
                              edgecolor='black', linewidth=2)
    ax.add_patch(query_box)
    ax.text(5, 9.4, "User Query", ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Semantic Search
    sem_box = FancyBboxPatch((1, 7), 2.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor='#E3F2FD',
                             edgecolor='black', linewidth=1.5)
    ax.add_patch(sem_box)
    ax.text(2.25, 7.75, 'Semantic\nSearch\n(Vector)', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # BM25 Search
    bm25_box = FancyBboxPatch((6.5, 7), 2.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor='#FFF3E0',
                             edgecolor='black', linewidth=1.5)
    ax.add_patch(bm25_box)
    ax.text(7.75, 7.75, 'BM25\nSearch\n(Keyword)', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Fusion
    fusion_box = FancyBboxPatch((3.5, 5), 3, 1,
                               boxstyle="round,pad=0.1",
                               facecolor='#E8F5E9',
                               edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5, 5.5, 'Hybrid Fusion\n(0.7 Semantic + 0.3 BM25)', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Reranking
    rerank_box = FancyBboxPatch((3.5, 3), 3, 1,
                               boxstyle="round,pad=0.1",
                               facecolor='#FCE4EC',
                               edgecolor='black', linewidth=2)
    ax.add_patch(rerank_box)
    ax.text(5, 3.5, 'Reranking\n(Alibaba GTE-Rerank-v2)', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Final Results
    final_box = FancyBboxPatch((3.5, 1), 3, 1,
                              boxstyle="round,pad=0.1",
                              facecolor='#4CAF50',
                              edgecolor='black', linewidth=2)
    ax.add_patch(final_box)
    ax.text(5, 1.5, 'Top-K Results\n(5 chunks)', 
           ha='center', va='center', fontsize=11, 
           fontweight='bold', color='white')
    
    # Oklar
    arrows = [
        ((5, 9), (2.25, 7)),
        ((5, 9), (7.75, 7)),
        ((2.25, 7), (4, 5.5)),
        ((7.75, 7), (6, 5.5)),
        ((5, 5), (5, 4)),
        ((5, 3), (5, 2)),
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', 
                               mutation_scale=20,
                               linewidth=2,
                               color='gray',
                               alpha=0.7)
        ax.add_patch(arrow)
    
    ax.text(5, 9.8, 'Hybrid Retrieval System', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path('diagrams')
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'hybrid_retrieval_diagram.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'hybrid_retrieval_diagram.pdf', 
               bbox_inches='tight', facecolor='white')
    print("✅ Hybrid retrieval diagram oluşturuldu!")
    plt.close()

if __name__ == "__main__":
    print("🎨 Profesyonel RAG Diagramları Oluşturuluyor...\n")
    
    create_architecture_diagram()
    create_query_flow_diagram()
    create_hybrid_retrieval_diagram()
    
    print("\n✅ Tüm diagramlar 'diagrams' klasöründe oluşturuldu!")
    print("📄 Formatlar: PNG (300 DPI) ve PDF (vektör)")



