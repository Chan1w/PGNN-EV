import matplotlib.pyplot as plt
import numpy as np
import os

def draw_SOH(self, preds, test_full_trues, folder_path, pic_name, file_name, save=True, window_size=10):
    plt.figure(figsize=(15, 5))

    plt.plot(range(len(test_full_trues)), test_full_trues, label='Test True Values', color='blue', linewidth=1.5)

    shifted_preds = [np.nan] * window_size + list(preds)

    if len(shifted_preds) < len(test_full_trues):
        shifted_preds += [np.nan] * (len(test_full_trues) - len(shifted_preds))
    else:
        shifted_preds = shifted_preds[:len(test_full_trues)]

    plt.plot(
        range(len(test_full_trues)),
        shifted_preds,
        label='Predicted Values',
        color='red',
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=6,
        markerfacecolor='red',
        markeredgecolor='darkred',
        alpha=0.7
    )

    plt.axvline(x=window_size, color='black',
                linestyle='--', alpha=0.9, label=f'Prediction Start (window={window_size})')

    plt.legend()
    plt.title(pic_name)
    plt.xlabel('Test Cycle Index')
    plt.ylabel('Capacity(Ah)')
    plt.xlabel('Time(week)')
    if self.args.battery == "VST":
        plt.ylim(100, 150)
    if self.args.battery == "B20":
        plt.ylim(103, 137)
    if self.args.battery == "LP":
        plt.ylim(150, 250)
    plt.grid(True)

    if save:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, file_name + '.png'), dpi=300, bbox_inches='tight')
    plt.show()
