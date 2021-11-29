import json
import numpy as np
import pandas as pd

from likelihood import str_to_lst, predFunc_guess


def process_csv(fname: str = "Experiment2a_data.csv"):
    df = pd.read_csv(fname)
    inputs = list(df["records.input_sequence"])
    acc = list(df["records.accuracy"])

    input_stripped = []
    acc_stripped = []
    preds_stripped = []

    for i in range(len(inputs)):
        minlength = min(len(inputs[i]), len(acc[i]))
        stripped_input = inputs[i][:minlength]
        stripped_acc = acc[i][:minlength]
        predstr = ""
        for j in range(len(stripped_input)):
            predstr += str((int(stripped_input[j]) + int(stripped_acc[j])) % 2)

        input_stripped.append(stripped_input)
        acc_stripped.append(stripped_acc)
        preds_stripped.append(predstr)

    df["accuracy"] = acc_stripped
    df["input"] = input_stripped
    df["pred"] = preds_stripped

    outname = fname.split(".")[0] + "_clean.csv"
    df.to_csv(outname)

    return df


def main():
    pass


if __name__ == "__main__":
    main()


# Keys of dicts: ['screens', 'session', 'location_country_name',
# 'window_size_y', 'bootout_code', 'browser_version', 'date', 'os_version',
# 'input_sequence', 'os_name', 'isComplete', 'max_progress', 'R_map',
# 'window_size_x', 'location_region', 'location_city', 'isMTurk', 'input_type',
# 'device_input', 'id', 'user', 'browser_name']
