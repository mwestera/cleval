# clEval: Command-line tool for evaluating classifier predictions vs. targets 

### Author: Matthijs Westera

Simple CLI to compare precomputed lists of predictions vs targets, logging a simple classification report, optionally a 
PDF with plots, and outputting a classification report as json dict.

## Install

```bash
$ pipx install git+https://github.com/mwestera/cleval
```

This will make the command `cleval` available.

## Examples:

```bash
$ cleval preds.txt targs.txt --pdf report.pdf
```

Or with piping:

```bash
$ cat preds.txt | cleval - targs.txt --pdf report.pdf > metrics.json
```
