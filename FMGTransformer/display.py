import matplotlib.pyplot as plt
import numpy as np



def plot_prediction_results(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))


    ax.scatter(y_true, y_pred, c='#1f77b4', label='Predicted',
               alpha=0.8, edgecolors='w', s=30)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
            'r--', label='Ideal line (y = x)', linewidth=2)

    ax.set_title("Test Set: Predicted vs True", fontsize=12)
    ax.set_xlabel("True Values", fontsize=10)
    ax.set_ylabel("Predicted Values", fontsize=10)
    ax.legend()

    plt.tight_layout()
    plt.show()






def plot_loss_and_r2_curves(train_losses, train_r2_scores,  epochs, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_r2_scores, label='Train R²', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Training and Validation R² Curve')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/r2_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np




def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'R2': round(r2, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
    }






