# visualization.py

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


def plot_curves(
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    save_dir
):

    epochs = range(
        1,
        len(train_loss) + 1
    )

    # Loss
    plt.figure()

    plt.plot(
        epochs,
        train_loss,
        label='train'
    )

    plt.plot(
        epochs,
        val_loss,
        label='validation'
    )

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.savefig(
        save_dir / 'loss.png',
        dpi=200
    )

    plt.close()


    # Accuracy
    plt.figure()

    plt.plot(
        epochs,
        train_acc,
        label='train'
    )

    plt.plot(
        epochs,
        val_acc,
        label='validation'
    )

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.savefig(
        save_dir / 'accuracy.png',
        dpi=200
    )

    plt.close()


def plot_decision_boundary(
    net,
    X,
    y,
    device,
    save_dir
):

    X_np = X.numpy()
    y_np = y.numpy()

    x_min = X_np[:, 0].min() - 0.5
    x_max = X_np[:, 0].max() + 0.5

    y_min = X_np[:, 1].min() - 0.5
    y_max = X_np[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(
            x_min,
            x_max,
            0.02
        ),
        np.arange(
            y_min,
            y_max,
            0.02
        )
    )

    grid = torch.tensor(
        np.c_[
            xx.ravel(),
            yy.ravel()
        ],
        dtype=torch.float32
    ).to(device)

    net.eval()

    with torch.no_grad():

        pred = net(grid).argmax(
            dim=1
        )

    pred = (
        pred.cpu()
        .numpy()
        .reshape(xx.shape)
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.contourf(
        xx,
        yy,
        pred,
        alpha=0.3,
        levels=[
            -0.5,
            0.5,
            1.5
        ]
    )

    plt.scatter(
        X_np[:, 0],
        X_np[:, 1],
        c=y_np,
        edgecolors='k',
        s=20
    )

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary')

    plt.savefig(
        save_dir / 'decision_boundary.png',
        dpi=200
    )

    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_dir
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1]
    )

    disp.plot(
        values_format='d'
    )

    plt.title(
        'Confusion Matrix'
    )

    plt.savefig(
        save_dir / 'confusion_matrix.png',
        dpi=200
    )

    plt.close()


def show_misclassified(
    X,
    y_true,
    y_pred,
    save_dir,
    n=5
):

    wrong = np.where(
        y_true != y_pred
    )[0]

    wrong = wrong[:n]


    # --------------------------------------------------------
    # 可视化错误样本
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y_true,
        alpha=0.25
    )

    if len(wrong) > 0:

        plt.scatter(
            X[wrong, 0],
            X[wrong, 1],
            facecolors='none',
            edgecolors='k',
            s=120
        )

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Misclassified Samples')

    plt.savefig(
        save_dir / 'misclassified.png',
        dpi=200
    )

    plt.close()


    # --------------------------------------------------------
    # 保存错误样本信息
    # --------------------------------------------------------

    with open(
        save_dir / 'misclassified.txt',
        'w',
        encoding='utf-8'
    ) as f:

        for i in wrong:

            f.write(
                f'index={i}, '
                f'x=({X[i, 0]:.3f}, '
                f'{X[i, 1]:.3f}), '
                f'true={y_true[i]}, '
                f'pred={y_pred[i]}\n'
            )


    print(
        '\nMisclassified samples:'
    )

    for i in wrong:

        print(
            f'index={i}, '
            f'x=({X[i, 0]:.3f}, '
            f'{X[i, 1]:.3f}), '
            f'true={y_true[i]}, '
            f'pred={y_pred[i]}'
        )