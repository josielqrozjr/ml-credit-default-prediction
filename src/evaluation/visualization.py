"""
Geração de Gráficos e Visualizações do Benchmark
------------------------------------------------
Lê os resultados consolidados das Fases 2, 4 e 5 e gera gráficos
acadêmicos em alta resolução para o TCC.
"""

import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurações de estilo para gráficos mais acadêmicos e limpos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

def plot_phase2_ranking():
    """Gera o gráfico de barras horizontais do Campeonato Aberto (Fase 2)."""
    file_path = RESULTS_DIR / "phase2_benchmark_ranking.csv"
    if not file_path.exists():
        print("Arquivo da Fase 2 não encontrado.")
        return
        
    df = pd.read_csv(file_path).sort_values(by="AMEX Score (OOF)", ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df["Modelo"], df["AMEX Score (OOF)"], color=sns.color_palette("viridis", len(df)))
    
    plt.title("Ranking de Baseline - Fase 2 (AMEX Score)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("AMEX Score Global", fontsize=14)
    plt.ylabel("Modelos", fontsize=14)
    plt.xlim(0.5, 0.82) # Ajusta o eixo X para dar zoom nas diferenças reais
    
    # Adiciona os valores nas barras
    for bar in bars:
        plt.text(
            bar.get_width() + 0.005, 
            bar.get_y() + bar.get_height()/2, 
            f'{bar.get_width():.4f}', 
            va='center', ha='left', fontsize=11, fontweight='bold'
        )
        
    output_path = PLOTS_DIR / "01_phase2_ranking.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo: {output_path}")

def plot_amex_evolution():
    """Gera um gráfico de barras mostrando a evolução do Score nas fases."""
    
    # Extrai a melhor nota da Fase 2
    f2_path = RESULTS_DIR / "phase2_benchmark_ranking.csv"
    f2_score = pd.read_csv(f2_path)["AMEX Score (OOF)"].max() if f2_path.exists() else 0.7872
    
    # Extrai a nota do Voting Classifier da Fase 4
    f4_path = RESULTS_DIR / "phase4_ensembles_ranking.csv"
    f4_score = pd.read_csv(f4_path).loc[1, "AMEX Score"] if f4_path.exists() else 0.7920
    
    # Extrai a nota final da Fase 5
    f5_path = RESULTS_DIR / "phase5_final_blind_test.csv"
    f5_score = pd.read_csv(f5_path)["AMEX_Score"].max() if f5_path.exists() else 0.7931

    fases = ["Fase 2\n(Melhor Baseline)", "Fase 4\n(Voting OOF)", "Fase 5\n(Teste Cego)"]
    scores = [f2_score, f4_score, f5_score]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(fases, scores, color=['#A9A9A9', '#4682B4', '#2E8B57'], width=0.5)
    
    plt.title("Evolução Preditiva do AMEX Score", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("AMEX Score", fontsize=14)
    plt.ylim(0.7800, 0.7960) # Zoom intenso para ver a evolução dos milésimos
    
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.0005, 
            f'{bar.get_height():.4f}', 
            va='bottom', ha='center', fontsize=12, fontweight='bold'
        )
        
    output_path = PLOTS_DIR / "02_amex_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo: {output_path}")

def plot_confusion_matrix():
    """Lê a matriz de confusão da Fase 5 e gera um Heatmap."""
    file_path = RESULTS_DIR / "phase5_final_blind_test.csv"
    if not file_path.exists():
        print("Arquivo da Fase 5 não encontrado.")
        return
        
    df = pd.read_csv(file_path)
    # Converte a string da matriz para uma lista python de verdade
    cm_str = df["Confusion_Matrix"].iloc[0]
    cm = np.array(ast.literal_eval(cm_str))
    
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                     annot_kws={"size": 16, "weight": "bold"})
    
    plt.title("Matriz de Confusão: Voting Classifier\n(Base de Teste: 91.783 clientes)", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Previsão do Modelo", fontsize=12, fontweight='bold')
    plt.ylabel("Realidade (Gabarito)", fontsize=12, fontweight='bold')
    
    ax.set_xticklabels(['Bom Pagador (0)', 'Inadimplente (1)'])
    ax.set_yticklabels(['Bom Pagador (0)', 'Inadimplente (1)'], rotation=0)
    
    output_path = PLOTS_DIR / "03_confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo: {output_path}")

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Gerando gráficos para o TCC...")
    plot_phase2_ranking()
    plot_amex_evolution()
    plot_confusion_matrix()
    print("Processo finalizado com sucesso!")

if __name__ == "__main__":
    main()