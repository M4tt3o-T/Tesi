import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Evaluator:
    
    def initialize_log_file(self, filepath, title):
        """Crea la cartella se non esiste e inizializza il file di log con un'intestazione."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"{title}\n")
            file.write("=" * 130 + "\n")
            
    def _format_time(self, seconds):
        """Formatta i secondi in stringa HH:MM:SS."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def log_suite_result(self, filepath, model_name, results, tempo_totale):
        """Calcola la media dell'accuratezza e salva il risultato formattato nel file di log."""
        accuracies = [res['accuracy'] for res in results]
        mean_acc = np.mean(accuracies) * 100
        std_acc = np.std(accuracies) * 100
        
        acc_str = f"{mean_acc:>5.1f}% ± {std_acc:>4.1f}%"
        tempo_str = self._format_time(tempo_totale)
        
        riga_risultato = f"{model_name:<40} -> Accuracy: {acc_str:<15} | Tempo: {tempo_str}"
        
        with open(filepath, "a", encoding="utf-8") as file:
            file.write(riga_risultato + "\n")
            
    def log_hierarchical_result(self, filepath, model_name, results, tempo_totale):
        """Versione specifica per i modelli gerarchici (salva 3 metriche diverse)."""
        acc_globale = [res['accuracy'] for res in results]
        acc_s1 = [res['accuracy_s1'] for res in results]
        acc_s2 = [res['accuracy_s2'] for res in results]
        
        acc_globale_str = f"{(np.mean(acc_globale)*100):>5.1f}% ± {(np.std(acc_globale)*100):>4.1f}%"
        acc_s2_str = f"{(np.mean(acc_s2)*100):>5.1f}% ± {(np.std(acc_s2)*100):>4.1f}%"
        
        tempo_str = self._format_time(tempo_totale)
        
        riga_risultato = f"{model_name:<40} -> Accuracy globale: {acc_globale_str:<15} | Accuracy stage 2: {acc_s2_str:<15} | Tempo: {tempo_str}"
        
        with open(filepath, "a", encoding="utf-8") as file:
            file.write(riga_risultato + "\n")

    def generate_top5(self, log_file, output_file):
        """Legge il log standard, estrae le percentuali e salva le prime 5."""
        if not os.path.exists(log_file):
            print(f"Errore: Il file {log_file} non esiste.")
            return

        risultati = []
        pattern_acc = re.compile(r"Accuracy:\s*([0-9\.]+)%")

        with open(log_file, "r", encoding="utf-8") as f:
            for linea in f:
                match = pattern_acc.search(linea)
                if match:
                    acc_val = float(match.group(1))
                    risultati.append((acc_val, linea.strip()))

        risultati_ordinati = sorted(risultati, key=lambda x: x[0], reverse=True)[:5]

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== TOP 5 CONFIGURAZIONI ===\n" + "=" * 130 + "\n")
            for i, (_, riga) in enumerate(risultati_ordinati, 1):
                f.write(f"{i}° Posto | {riga}\n")

    def generate_top5_hierarchical(self, log_file, output_file):
        """Legge il log gerarchico, estrae l'Accuracy dello STAGE 2 e salva le prime 5."""
        if not os.path.exists(log_file):
            return

        risultati = []
        pattern_acc_s2 = re.compile(r"Accuracy stage 2:\s*([0-9\.]+)%")

        with open(log_file, "r", encoding="utf-8") as f:
            for linea in f:
                match = pattern_acc_s2.search(linea)
                if match:
                    acc_s2_val = float(match.group(1))
                    risultati.append((acc_s2_val, linea.strip()))

        risultati_ordinati = sorted(risultati, key=lambda x: x[0], reverse=True)[:5]

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== TOP 5 CONFIGURAZIONI (Ordinate per Stage 2) ===\n" + "=" * 130 + "\n")
            for i, (_, riga) in enumerate(risultati_ordinati, 1):
                f.write(f"{i}° Posto | {riga}\n")

    def save_confusion_matrix(self, results, classes, title, filepath):
        """Calcola la matrice media e la sua deviazione standard, poi salva l'immagine png."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        all_cms = []
        for res in results:
            cm = confusion_matrix(res['y_true'], res['y_pred'], labels=classes, normalize='true')
            all_cms.append(cm * 100)
            
        mean_cm = np.mean(all_cms, axis=0)
        std_cm = np.std(all_cms, axis=0)
        
        # Preparazione etichette interne della matrice
        annot = np.empty_like(mean_cm, dtype=object)
        n = len(classes)
        for i in range(n):
            for j in range(n):
                if mean_cm[i, j] >= 0.1:
                    annot[i, j] = f"{mean_cm[i, j]:.1f}\n±{std_cm[i, j]:.1f}"
                else:
                    annot[i, j] = ""

        # Recupera l'accuratezza media dai risultati per il titolo
        mean_acc = np.mean([res['accuracy'] for res in results]) * 100
        std_acc = np.std([res['accuracy'] for res in results]) * 100

        # Disegno grafico
        fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
        sns.heatmap(mean_cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                    xticklabels=classes, yticklabels=classes, ax=ax, 
                    annot_kws={"size": 9}, linewidths=0.5, linecolor="lightgray")
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        accuracy_string = f"{mean_acc:>5.1f}% ± {std_acc:>4.1f}%"
        titolo_grafico = f"10-Fold CV: {title}\nAccuracy: {accuracy_string}"
        plt.title(titolo_grafico, fontsize=14, fontweight='bold', pad=20)
        plt.ylabel("Etichetta Reale", fontweight='bold', fontsize=12)
        plt.xlabel("Etichetta Predetta", fontweight='bold', fontsize=12)
        
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        plt.close(fig)