import sys

import pandas
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
import textwrap

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

"""  # TODO Update instructions and README.md


# TODO: Second choices, proba, and write examples to html/pdf

def main():
    parser = argparse.ArgumentParser(description='Compare precomputed lists of predictions vs targets, logging a simple classification report, optionally PDF, and outputting json.')
    parser.add_argument('--csv', type=argparse.FileType('r'), nargs='?', help='.csv file with columns --true, --pred, --prob, --item')
    parser.add_argument('--true', type=str, required=False, default=None, help='If --csv, column name of true labels (,-separated); else, file containing true classes; if not given, exploring only predictions.')
    parser.add_argument('--pred', type=str, required=False, default=None, help='If --csv, column name of predicted labels (,-separated); else, .csv file containing predicted classes.')
    parser.add_argument('--prob', type=str, required=False, default=None, help='If --csv, column name of predicted probabilities (,-separated); else, .csv file containing probabilities.')
    parser.add_argument('--text', type=str, required=False, default=None, help='If --csv, column name of categorized content (e.g., text or sentence categorized); else file containing items one per line.')

    parser.add_argument('--multi', action='store_true', help='To assume potentially multiple labels per item (multi-label classification), in which case true and pred files contain csv rows; inferred from --true otherwise.')
    parser.add_argument('--labels', nargs='*', type=str, default=[], help='Classification labels (separated by spaces); inferred from --true or --prob otherwise.')
    parser.add_argument('--pdf', type=str, default=None, help='Path to write PDF report to.')

    args = parser.parse_args()

    y_true = y_pred = y_prob = texts = None

    if args.csv and args.prob and not args.labels:
        raise ValueError('--labels must be specified when --prob comes from column in --csv.')

    if args.csv:
        df = pandas.read_csv(args.csv)
        y_true = [row.split(',') for row in df[args.true]] if args.true else None
        y_pred = [row.split(',') for row in df[args.pred]] if args.pred else None
        y_prob = [[float(i) for i in row.split(',')] for row in df[args.prob]] if args.prob else None
        texts = df[args.text] if args.text else None
    else:
        if args.true:
            with open(args.true, 'r') as file:
                y_true = [set(row) for row in csv.reader(file)]
        if args.pred:
            with open(args.pred, 'r') as file:
                y_pred = [set(row) for row in csv.reader(file)]
        if args.prob:
            with open(args.prob, 'r') as file:
                y_prob = [row for row in csv.reader(file)]
            if not args.labels:
                args.labels = y_prob[0]
                y_prob = y_prob[1:]
                logging.warning(f'--labels inferred from first row of probabilities: {args.labels}')
            y_prob = [[float(i) for i in row] for row in y_prob]
        if args.text:
            with open(args.text, 'r') as file:
                texts = [line.strip() for line in file]

    if not args.multi and any(len(row) > 1 for row in y_true):
        logging.warning('--multi is inferred from --true.')
        args.multi = True

    if not args.multi:
        if y_true:
            assert all(len(l) == 1 for l in y_true)
            y_true = list(itertools.chain(*y_true))
        if y_pred:
            assert all(len(l) == 1 for l in y_pred)
            y_pred = list(itertools.chain(*y_pred))

    # Infer labels if still needed:
    if not args.labels:
        args.labels = sorted(set(y_true)) if not args.multi else sorted(set(itertools.chain(*y_true)))
        logging.warning(f'--labels are inferred: {args.labels}')

    if y_prob and not y_pred:
        y_pred = probs_to_choices(y_prob, args.labels, is_multilabel=args.multi, probs_threshold=.5)  # TODO: Expose probs threshold as arg?

    if y_true and y_pred:
        assert len(y_true) == len(y_pred)
    if y_true and y_prob:
        assert len(y_true) == len(y_prob)
    if y_pred and y_prob:
        assert len(y_pred) == len(y_prob)

    record_for_pdf = {
    }

    result, confusion_mtrx = evaluate_categorical(y_true, y_pred, labels=args.labels, is_multilabel=args.multi)
    print(json.dumps(result))

    if y_prob:  # TODO do probabilistic analysis; confidence and stuff?
        pass
        # evaluate_probabilistic(y_true, y_pred, y_prob, labels=args.labels, is_multilabel=args.multi, record=record_for_pdf)

    if texts and y_pred and y_true:
        print_errors(y_true, y_pred, texts, y_prob, labels=args.labels)

    if args.pdf:    # TODO To be removed/refactored
        write_pdf_report(y_true, y_pred, confusion_mtrx.values, args.labels, args.pdf)


def probs_to_choices(y_pred_probs: list[list[float]], labels: list[str], is_multilabel: bool = False, probs_threshold: float = .5):

    def probs_to_choices(probs):
        indexed_probs = sorted((prob, i) for i, prob in enumerate(probs))
        if is_multilabel:
            return {labels[i] for prob, i in indexed_probs if prob >= probs_threshold}
        else:
            best_index = indexed_probs[-1][1]
            return labels[best_index]

    y_pred = [probs_to_choices(probs) for probs in y_pred_probs]
    return y_pred


def evaluate_categorical(y_true: list[set[str]] | list[str],
                         y_pred: list[set[str]] | list[str] | None,
                         labels: list[str],
                         is_multilabel: bool = False,
                         record: dict = None) -> tuple[dict, pandas.DataFrame]:

    if y_pred is None:
        y_pred = y_true

    report, scores_dict = make_classification_report(y_true, y_pred, is_multilabel=is_multilabel, labels=labels)
    logging.info(report)

    confusion_matrix_df = make_confusion_matrix(y_true, y_pred, is_multilabel=is_multilabel, labels=labels)
    logging.info('Confusion table:\n')
    logging.info(confusion_matrix_df)

    return scores_dict, confusion_matrix_df


def print_errors(y_true, y_pred, texts, y_prob, labels):
    df = pandas.DataFrame({'true': y_true, 'pred': y_pred, 'text': texts,'prob': y_prob})
    df['correct'] = df['true'] == df['pred']
    df = df.loc[~df['correct']]
    for label, group in df.groupby('true'):
        print(f'\n\n## Actual label: {label}')
        for i, row in group.iterrows():
            text = textwrap.fill(row["text"], initial_indent='> ', subsequent_indent='> ')
            if row['prob']:
                pred_prob = row['prob'][labels.index(row['pred'])]
                true_prob = row['prob'][labels.index(row['true'])]
                pred_prob_str = f' ({pred_prob:.2f})'
                true_prob_str = f' ({true_prob:.2f})'
            logging.info(f'\nPredicted: {row["pred"]}{pred_prob_str} / actual {row["true"]}{true_prob_str}\n{text}')


def make_classification_report(y_true, y_pred, is_multilabel: bool, labels: list[str]) -> tuple[str, dict]:
    if is_multilabel:
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(y_true)
        y_pred = mlb.transform(y_pred)

    report = classification_report(y_true, y_pred, target_names=labels)
    scores_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    return report, scores_dict


def make_confusion_matrix(y_true, y_pred, is_multilabel: bool, labels: list[str]) -> pandas.DataFrame:
    if is_multilabel:
        y_true, y_pred = collapse_multilabel(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    conf_matrix_df = pandas.DataFrame(conf_matrix, index=labels, columns=labels)
    conf_matrix_df.index.name = 'Actual:'
    conf_matrix_df.columns.name = 'Predicted:'
    return conf_matrix_df


def collapse_multilabel(y_true: list[set], y_pred: list[set]) -> tuple[list, list]:
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
    combined_data = pandas.DataFrame({
        'y_true': pandas.Categorical(y_true, categories=labels, ordered=True),
        'y_pred': pandas.Categorical(y_pred, categories=labels, ordered=True),
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
    logging.info(f"\nPDF report saved as {pdf_filename}")



if __name__ == '__main__':
    main()
