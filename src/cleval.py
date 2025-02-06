import sys

import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

import argparse
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import logging
import json
import csv

import random

import itertools

logging.basicConfig(level=logging.INFO, format='')


"""
Author: Matthijs Westera

Simple CLI to compare precomputed lists of predictions vs targets, logging a simple classification report, optionally a 
PDF with plots, and outputting a classification report as json dict.

Examples:

$ cleval preds.txt targs.txt --pdf report.pdf

Or with piping:

$ cat preds.txt | cleval - targs.txt --pdf report.pdf > metrics.json

"""


# TODO: Second choices, proba, and write examples to html/pdf

def main():
    parser = argparse.ArgumentParser(description='Compare precomputed lists of predictions vs targets, logging a simple classification report, optionally PDF, and outputting json.')
    parser.add_argument('true', type=argparse.FileType('r'), help='File containing predicted classes, or - for stdin.')
    parser.add_argument('pred', type=argparse.FileType('r'), nargs='?', default=None, help='File containing true classes, or - for stdin; default None, exploring only predictions.')
    parser.add_argument('--multi', action='store_true', help='To assume potentially multiple labels per item (multi-label classification), in which case true and pred files contain csv rows; inferred from true otherwise.')
    parser.add_argument('--prob', action='store_true', help='If pred file contains raw probabilities instead of labels; inferred from pred otherwise.')
    parser.add_argument('--labels', nargs='*', type=str, default=[], help='Classification labels (separated by spaces); inferred from true otherwise.')
    parser.add_argument('--pdf', type=str, default=None, help='Path to write PDF report to.')

    args = parser.parse_args()

    file_true = sys.stdin if args.true == '-' else args.true
    file_pred = sys.stdin if args.pred == '-' else args.pred

    y_true = [set(row) for row in csv.reader(file_true)]

    if not args.multi and any(len(row) > 1 for row in y_true):
        logging.warning('--multi is inferred.')
        args.multi = True

    if not args.multi:
        assert all(len(l) == 1 for l in y_true)
        y_true = list(itertools.chain(*y_true))

    # Read y_pred, try if probabilities:
    y_pred = None
    if file_pred:
        y_pred = [list(row) for row in csv.reader(file_pred)]
        try:
            if not args.labels:
                first_row = y_pred[0]
                y_pred = y_pred[1:]
            y_pred = [[float(i) for i in row] for row in y_pred]
            if not args.prob:
                logging.warning('--prob is inferred.')
                args.prob = True
            if not args.labels:
                logging.warning('--labels inferred from first row of predictions.')
                args.labels = first_row
        except ValueError as e:
            if args.prob:
                raise e
            y_pred = [set(row) for row in y_pred]

        if not args.multi and not args.prob:
            assert all(len(l) == 1 for l in y_pred)
            y_pred = list(itertools.chain(*y_pred))

        assert len(y_true) == len(y_pred)

    # Infer labels if still needed:
    if not args.labels:
        args.labels = sorted(set(y_true)) if not args.multi else sorted(set(itertools.chain(*y_true)))
        logging.warning(f'--labels are inferred: {args.labels}')

    result = evaluate(y_true, y_pred, labels=args.labels, is_multilabel=args.multi, is_probs=args.prob, probs_threshold=.5, output_pdf=args.pdf)

    print(json.dumps(result))


def evaluate(y_true, y_pred, labels, is_multilabel=False, is_probs=False, probs_threshold=.5, output_pdf=None) -> dict:

    if y_pred is None:
        y_pred = y_true

    if is_probs:
        def probs_to_choices(probs):
            indexed_probs = sorted((prob, i) for i, prob in enumerate(probs))
            if is_multilabel:
                return {labels[i] for prob, i in indexed_probs if prob >= probs_threshold}
            else:
                best_index = indexed_probs[-1][1]
                return labels[best_index]

        y_pred_probs = y_pred
        y_pred = [probs_to_choices(probs) for probs in y_pred_probs]

    report = make_classification_report(y_true, y_pred, is_multilabel=is_multilabel, labels=labels)
    logging.info(report)

    confusion_matrix_df = make_confusion_matrix(y_true, y_pred, is_multilabel=is_multilabel, labels=labels)
    logging.info('Confusion table:\n')
    logging.info(confusion_matrix_df)

    if output_pdf:
        write_pdf_report(y_true, y_pred, confusion_matrix_df, labels, output_pdf)

    return classification_report(y_true, y_pred, target_names=labels, output_dict=True)


def make_classification_report(y_true, y_pred, is_multilabel, labels) -> str:
    if is_multilabel:
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(y_true)
        y_pred = mlb.transform(y_pred)

    report = classification_report(y_true, y_pred, target_names=labels)
    return report


def make_confusion_matrix(y_true, y_pred, is_multilabel, labels) -> pandas.DataFrame:
    if is_multilabel:
        y_true, y_pred = collapse_multilabel(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    conf_matrix_df.index.name = 'True:'
    conf_matrix_df.columns.name = 'Predicted:'
    return conf_matrix_df


def collapse_multilabel(y_true, y_pred):
    """
    >>> collapse_multilabel([{1, 2}, {2, 3, 4}], [{1, 3}, {4, 5}])
    ([1, 2, 4, 3], [1, 3, 4, 5])
    """
    new_items = []
    for t, p in zip(y_true, y_pred):
        t_minus_p = sorted(a for a in t if a not in p)
        p_minus_t = sorted(a for a in p if a not in t)
        for l in t & p:
            new_items.append((l, l))
        for l1, l2 in itertools.product(t_minus_p, p_minus_t):
            new_items.append((l1, l2))
    return tuple(map(list, zip(*new_items)))


def write_pdf_report(y_true, y_pred, conf_matrix, labels, out_path) -> None:

    # TODO: Does creating a report have to be this ugly?

    def plot_to_image(plt):
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return ImageReader(buf)

    pdf_filename = out_path
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2.0, height - 40, "Report generated by ClEval")
    c.setFont("Helvetica", 12)

    text = c.beginText(50, height - 100)
    text.textLines(classification_report(y_true, y_pred, labels=labels))
    c.drawText(text)

    # Histogram
    combined_data = pd.DataFrame({
        'y_true': pd.Categorical(y_true, categories=labels, ordered=True),
        'y_pred': pd.Categorical(y_pred, categories=labels, ordered=True),
    })

    plt.figure(figsize=(13.5, 5))
    sns.histplot(data=combined_data.melt(), x='value', hue='variable', multiple='dodge', shrink=0.8)
    plt.legend(['Targets', 'Predictions'])
    plt.title('Histogram'); plt.xlabel('Class'); plt.ylabel('Frequency'); plt.xticks(labels)
    hist_image = plot_to_image(plt)
    c.drawImage(hist_image, 30, height - 460, width=540, height=200)
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(13.5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True');
    conf_matrix_image = plot_to_image(plt)
    c.drawImage(conf_matrix_image, 30, height - 700, width=540, height=200)
    plt.close()

    c.showPage()
    c.save()
    logger.info(f"\nPDF report saved as {pdf_filename}")



if __name__ == '__main__':
    main()
