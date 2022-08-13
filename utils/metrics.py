from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple


def show_results(labels: List[int], preds: List[int]):
    title = _str_metrics(labels, preds)
    plot_confusion_matrix(labels, preds, title=title)


def _str_metrics(labels: List[int], preds: List[int]):
    report = classification_report(labels, preds, zero_division=1, output_dict=True)
    acc = report['accuracy']
    startegy = 'weighted avg'
    f1 = report[startegy]['f1-score']
    prec = report[startegy]['precision']
    rec = report[startegy]['recall']
    title = f'accuracy: {acc:.3f}, precision:{prec:.3f}, recall: {rec:.3f}, f1: {f1:.3f}'
    return title


def plot_confusion_matrix(
        labels: List[int], preds: List[int],
        figsize: Tuple[int, int] = (15, 30),
        title: str = 'Confusion Matrix'
) -> None:
    cm = confusion_matrix(labels, preds)
    df_to_plot = pd.DataFrame(cm)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df_to_plot, annot=True, fmt="d")
    ax.set(xlabel='Predicted', ylabel='True')
    plt.title(title, fontsize=25)
    plt.show();