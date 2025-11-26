"""
Módulo de análisis y evaluación para autoencoders
Incluye funciones para análisis detallado de errores y comparación entre modelos
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import wandb


def compute_reconstruction_errors(model, dataloader, device='mps' if torch.mps.is_available() else 'cpu'):
    """
    Calcula errores de reconstrucción por imagen
    
    Returns:
        dict con arrays de:
        - errors: error MSE por imagen
        - labels: etiquetas de clase
        - images: imágenes originales
        - reconstructions: reconstrucciones
    """
    model.eval()
    model = model.to(device)
    
    all_errors = []
    all_labels = []
    all_images = []
    all_reconstructions = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            reconstructions = model(images)
            
            # Error por imagen (MSE pixel-wise)
            errors = torch.mean((images - reconstructions) ** 2, dim=[1, 2, 3])
            
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_images.append(images.cpu())
            all_reconstructions.append(reconstructions.cpu())
    
    return {
        'errors': np.concatenate(all_errors),
        'labels': np.concatenate(all_labels),
        'images': torch.cat(all_images),
        'reconstructions': torch.cat(all_reconstructions)
    }


def compute_errors_by_defect_type(model, dataset_path, class_names, transform, 
                                   device='mps' if torch.mps.is_available() else 'cpu'):
    """
    Calcula errores separados para imágenes buenas vs cada tipo de defecto
    """
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import os
    import glob
    
    model.eval()
    model = model.to(device)
    
    results = {}
    
    class SimpleImageDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = image_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transform(img)
    
    for class_name in class_names:
        results[class_name] = {}
        test_path = os.path.join(dataset_path, class_name, 'test')
        
        if not os.path.exists(test_path):
            continue
        
        # Obtener todas las subcarpetas (good y tipos de defectos)
        subfolders = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(test_path, subfolder)
            
            # Obtener todas las imágenes de esta carpeta
            image_paths = glob.glob(os.path.join(subfolder_path, '*.png')) + \
                         glob.glob(os.path.join(subfolder_path, '*.jpg'))
            
            if not image_paths:
                continue
            
            results[class_name][subfolder] = []
            
            # Crear dataset y dataloader
            dataset = SimpleImageDataset(image_paths, transform)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Calcular errores
            with torch.no_grad():
                for images in loader:
                    images = images.to(device)
                    recons = model(images)
                    errors = torch.mean((images - recons) ** 2, dim=[1, 2, 3])
                    results[class_name][subfolder].extend(errors.cpu().numpy())
        
        print(f"  {class_name}: {', '.join(results[class_name].keys())}")
    
    return results


def plot_error_histograms_by_defect(error_dict, class_name, save_path=None):
    """
    Genera histogramas comparando errores de imágenes buenas vs defectuosas por tipo
    
    Args:
        error_dict: dict con estructura {defect_type: [errors]}
        class_name: nombre de la clase
        save_path: ruta para guardar la figura (opcional)
    """
    n_defects = len(error_dict) - 1  # -1 por 'good'
    
    if n_defects == 0:
        print(f"No hay defectos para la clase {class_name}")
        return None
    
    fig, axes = plt.subplots(1, n_defects + 1, figsize=(5 * (n_defects + 1), 4))
    if n_defects == 0:
        axes = [axes]
    
    # Plot global: good vs all defects
    ax = axes[0]
    good_errors = np.array(error_dict['good'])
    all_defect_errors = []
    for key, errors in error_dict.items():
        if key != 'good':
            all_defect_errors.extend(errors)
    
    ax.hist(good_errors, bins=30, alpha=0.6, label='Buenas', color='green')
    ax.hist(all_defect_errors, bins=30, alpha=0.6, label='Defectuosas', color='red')
    ax.set_xlabel('Error de Reconstrucción (MSE)')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'{class_name}\nBuenas vs Todas las Defectuosas')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plots individuales por tipo de defecto
    plot_idx = 1
    for defect_type, errors in error_dict.items():
        if defect_type == 'good':
            continue
        
        ax = axes[plot_idx]
        ax.hist(good_errors, bins=30, alpha=0.6, label='Buenas', color='green')
        ax.hist(errors, bins=30, alpha=0.6, label=defect_type, color='red')
        ax.set_xlabel('Error de Reconstrucción (MSE)')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'{class_name}\nBuenas vs {defect_type}')
        ax.legend()
        ax.grid(alpha=0.3)
        plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_models_reconstruction(models_dict, dataloader, n_samples=8, 
                                   device='mps' if torch.mps.is_available() else 'cpu'):
    """
    Compara reconstrucciones de múltiples modelos lado a lado
    
    Args:
        models_dict: dict {model_name: model}
        dataloader: dataloader con imágenes
        n_samples: número de imágenes a comparar
    """
    # Obtener un batch de imágenes
    images, _ = next(iter(dataloader))
    images = images[:n_samples].to(device)
    
    n_models = len(models_dict)
    fig, axes = plt.subplots(n_models + 1, n_samples, figsize=(n_samples * 2, (n_models + 1) * 2))
    
    # Fila 0: Imágenes originales
    for i in range(n_samples):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2  # De [-1, 1] a [0, 1]
        axes[0, i].imshow(np.clip(img, 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12, rotation=0, ha='right')
    
    # Filas siguientes: Reconstrucciones de cada modelo
    for model_idx, (model_name, model) in enumerate(models_dict.items(), start=1):
        model.eval()
        model = model.to(device)
        
        with torch.no_grad():
            recons = model(images)
        
        for i in range(n_samples):
            recon = recons[i].cpu().permute(1, 2, 0).numpy()
            recon = (recon + 1) / 2
            axes[model_idx, i].imshow(np.clip(recon, 0, 1))
            axes[model_idx, i].axis('off')
            if i == 0:
                axes[model_idx, i].set_ylabel(model_name, fontsize=12, rotation=0, ha='right')
    
    plt.tight_layout()
    return fig


def analyze_loss_function_impact(results_dict):
    """
    Analiza el impacto de cada función de pérdida
    
    Args:
        results_dict: dict con estructura {loss_name: {'train_loss': [], 'val_loss': [], 'final_error': float}}
    
    Returns:
        DataFrame con resumen estadístico
    """
    import pandas as pd
    
    summary = []
    for loss_name, metrics in results_dict.items():
        summary.append({
            'Función de Pérdida': loss_name,
            'Train Loss Final': metrics.get('train_loss', [0])[-1] if metrics.get('train_loss') else 'N/A',
            'Val Loss Final': metrics.get('val_loss', [0])[-1] if metrics.get('val_loss') else 'N/A',
            'Error de Reconstrucción (Test)': metrics.get('final_error', 'N/A'),
            'Convergencia': 'Rápida' if len(metrics.get('train_loss', [])) < 15 else 'Normal'
        })
    
    df = pd.DataFrame(summary)
    return df


def plot_comparison_dashboard(models_results, save_path=None):
    """
    Genera un dashboard completo comparando múltiples modelos
    
    Args:
        models_results: dict con {model_name: {
            'train_losses': [...],
            'val_losses': [...],
            'test_errors': {...}
        }}
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Curvas de entrenamiento
    ax1 = fig.add_subplot(gs[0, :2])
    for model_name, results in models_results.items():
        epochs = range(len(results['train_losses']))
        ax1.plot(epochs, results['train_losses'], label=f'{model_name} (train)', alpha=0.7)
        ax1.plot(epochs, results['val_losses'], label=f'{model_name} (val)', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Curvas de Entrenamiento y Validación')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(alpha=0.3)
    
    # 2. Errores finales comparados
    ax2 = fig.add_subplot(gs[0, 2])
    model_names = list(models_results.keys())
    final_errors = [results['test_errors']['mean'] for results in models_results.values()]
    ax2.bar(range(len(model_names)), final_errors, color='skyblue')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('Error Medio de Reconstrucción')
    ax2.set_title('Comparación de Error Final')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Distribución de errores por modelo
    ax3 = fig.add_subplot(gs[1, :])
    all_errors = []
    all_labels = []
    for model_name, results in models_results.items():
        errors = results['test_errors']['all_errors']
        all_errors.extend(errors)
        all_labels.extend([model_name] * len(errors))
    
    import pandas as pd
    df = pd.DataFrame({'Error': all_errors, 'Modelo': all_labels})
    
    for model_name in model_names:
        model_errors = df[df['Modelo'] == model_name]['Error']
        ax3.hist(model_errors, bins=40, alpha=0.5, label=model_name)
    
    ax3.set_xlabel('Error de Reconstrucción (MSE)')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Errores por Modelo')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Tabla resumen
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    table_data = []
    for model_name, results in models_results.items():
        table_data.append([
            model_name,
            f"{results['train_losses'][-1]:.4f}",
            f"{results['val_losses'][-1]:.4f}",
            f"{results['test_errors']['mean']:.4f}",
            f"{results['test_errors']['std']:.4f}"
        ])
    
    table = ax4.table(cellText=table_data,
                      colLabels=['Modelo', 'Train Loss', 'Val Loss', 'Error Medio', 'Error Std'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def log_final_comparison_to_wandb(models_results, project_name="tarea05_autoencoder"):
    """
    Registra una comparación final completa en WandB
    """
    # Crear un nuevo run para el análisis final
    run = wandb.init(project=project_name, name="final_comparison", job_type="analysis")
    
    # 1. Tabla comparativa
    comparison_data = []
    for model_name, results in models_results.items():
        comparison_data.append([
            model_name,
            results['train_losses'][-1],
            results['val_losses'][-1],
            results['test_errors']['mean'],
            results['test_errors']['std']
        ])
    
    table = wandb.Table(
        columns=['Modelo', 'Train Loss Final', 'Val Loss Final', 'Error Medio Test', 'Error Std Test'],
        data=comparison_data
    )
    run.log({"model_comparison_table": table})
    
    # 2. Dashboard visual
    fig = plot_comparison_dashboard(models_results)
    run.log({"comparison_dashboard": wandb.Image(fig)})
    plt.close(fig)
    
    # 3. Métricas individuales
    for model_name, results in models_results.items():
        run.log({
            f"{model_name}_final_train_loss": results['train_losses'][-1],
            f"{model_name}_final_val_loss": results['val_losses'][-1],
            f"{model_name}_test_error_mean": results['test_errors']['mean'],
            f"{model_name}_test_error_std": results['test_errors']['std']
        })
    
    run.finish()
    print(f"✓ Comparación final registrada en WandB: {run.url}")