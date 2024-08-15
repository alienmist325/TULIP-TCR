# TULIP-TCR

This is a fork of https://github.com/barthelemymp/TULIP-TCR, with some additional scripts to make predictions easier to run, and some of the prediction data used in https://github.com/alienmist325/tumour_immune_interactions. This fork doesn't make use of any of the retraining or finetuning. See `ignored.md` for all the directories and folders that are unused.

## Setting up

No additional setup is required beyond cloning the repository.

The repository has been tested with Python 3.10.

## Creating a valid input

This must be a `.csv` file, with columns like those seen in `data/input_1.1.csv`. Data can be marked as missing with  `<MIS>`

## Running the predictor

This is done by executing `predict.py`. Feel free to write a script yourself, and if not, utilise `predictions.bat` (Windows only)
The main command is:

```py
py -3.10 predict.py --test_dir {path_to_input} --modelconfig configs/shallow.config.json --output output/
```

I have placed all inputs under `data` with a filename of the format `input_{id}.csv`, and have given a description of each input in `data/readme.md`.