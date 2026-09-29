# train.py

import time
from pathlib import Path

import torch
from torch import nn

from model import load_data, make_net, init_weights
from visualization import (
    plot_curves,
    plot_decision_boundary,
    plot_confusion_matrix,
    show_misclassified
)


OUTPUT_DIR = Path('outputs')

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


device = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)


def evaluate(net, data_iter, loss):

    net.eval()

    total_loss = 0
    correct = 0
    total = 0

    predictions = []
    labels = []

    with torch.no_grad():

        for X, y in data_iter:

            X = X.to(device)
            y = y.to(device)

            y_hat = net(X)

            l = loss(y_hat, y)

            pred = y_hat.argmax(dim=1)

            total_loss += l.item() * y.numel()
            correct += (pred == y).sum().item()
            total += y.numel()

            predictions.append(pred.cpu())
            labels.append(y.cpu())

    return (
        total_loss / total,
        correct / total,
        torch.cat(predictions),
        torch.cat(labels)
    )


def train(
    train_iter,
    val_iter,
    hidden_units,
    lr,
    num_epochs,
    save_path
):

    net = make_net(hidden_units)
    net.apply(init_weights)
    net = net.to(device)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr
    )

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    best_val_loss = float('inf')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for epoch in range(num_epochs):

        net.train()

        total_loss = 0
        correct = 0
        total = 0

        for X, y in train_iter:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_hat = net(X)

            l = loss(y_hat, y)

            l.backward()

            optimizer.step()

            pred = y_hat.argmax(dim=1)

            total_loss += l.item() * y.numel()
            correct += (pred == y).sum().item()
            total += y.numel()

        epoch_train_loss = total_loss / total
        epoch_train_acc = correct / total

        (
            epoch_val_loss,
            epoch_val_acc,
            _,
            _
        ) = evaluate(
            net,
            val_iter,
            loss
        )

        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:

            best_val_loss = epoch_val_loss

            torch.save(
                net.state_dict(),
                save_path
            )

    if device.type == 'cuda':
        torch.cuda.synchronize()

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    else:
        peak_memory = 0

    train_time = (
        time.perf_counter()
        - start_time
    )

    return (
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        train_time,
        peak_memory
    )


def run_experiment(
    name,
    train_iter,
    val_iter,
    test_iter,
    X_test,
    y_test,
    hidden_units,
    lr,
    num_epochs=100
):

    save_dir = OUTPUT_DIR / name

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = save_dir / 'model.pt'

    (
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        train_time,
        peak_memory
    ) = train(
        train_iter,
        val_iter,
        hidden_units,
        lr,
        num_epochs,
        model_path
    )

    net = make_net(hidden_units)

    net.load_state_dict(
        torch.load(model_path)
    )

    net = net.to(device)

    loss = nn.CrossEntropyLoss()

    (
        test_loss,
        test_acc,
        test_preds,
        test_labels
    ) = evaluate(
        net,
        test_iter,
        loss
    )

    plot_curves(
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        save_dir
    )

    plot_decision_boundary(
        net,
        X_test,
        y_test,
        device,
        save_dir
    )

    plot_confusion_matrix(
        test_labels.numpy(),
        test_preds.numpy(),
        save_dir
    )

    show_misclassified(
        X_test.numpy(),
        test_labels.numpy(),
        test_preds.numpy(),
        save_dir
    )

    with open(
        save_dir / 'result.txt',
        'w',
        encoding='utf-8'
    ) as f:

        f.write(f'hidden_units = {hidden_units}\n')
        f.write(f'learning_rate = {lr}\n')
        f.write(f'epochs = {num_epochs}\n')
        f.write(f'train_time = {train_time:.3f}s\n')
        f.write(f'peak_memory = {peak_memory:.2f}MB\n')
        f.write(f'test_loss = {test_loss:.4f}\n')
        f.write(f'test_accuracy = {test_acc:.4f}\n')

    print(
        f'\n{name}'
    )

    print(
        f'train time: {train_time:.3f}s'
    )

    print(
        f'peak memory: {peak_memory:.2f}MB'
    )

    print(
        f'test loss: {test_loss:.4f}'
    )

    print(
        f'test accuracy: {test_acc:.4f}'
    )


def main():

    (
        X,
        y,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_iter,
        val_iter,
        test_iter
    ) = load_data(
        batch_size=64
    )

    run_experiment(
        'baseline',
        train_iter,
        val_iter,
        test_iter,
        X_test,
        y_test,
        hidden_units=16,
        lr=0.01
    )

    run_experiment(
        'width_32',
        train_iter,
        val_iter,
        test_iter,
        X_test,
        y_test,
        hidden_units=32,
        lr=0.01
    )

    run_experiment(
        'lr_0.005',
        train_iter,
        val_iter,
        test_iter,
        X_test,
        y_test,
        hidden_units=16,
        lr=0.005
    )


if __name__ == '__main__':
    main()