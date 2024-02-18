set /p input_id="Enter the input id: ":
set "input=data/input_%input_id%.csv"
echo %input%


py -3.10 predict.py --test_dir %input% --modelconfig configs/shallow.config.json --output output/