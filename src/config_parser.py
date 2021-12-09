import json
import platform

import os
def get_config(attributes, recursive=True):
    if platform.system() == 'Darwin':  # macosx
        config_path = '/Users/ducanhnguyen/Documents/mydeepconcolic/src/config_osx.json'
    elif platform.system() == 'Linux':  # hpc
        config_path = './config_hpc.json'
    else:
        config_path = "C:/Users/ducanhnguyen/PycharmProjects/mydeepconcolic/src/config_win.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

        con = config
        for attribute in attributes:
            con = con[attribute]

        if isinstance(con, str) and recursive:
            DATASET_SIGNAL = '{dataset}'
            if DATASET_SIGNAL in con:
                con = con.replace(DATASET_SIGNAL, get_config(['dataset'], False))

            BASE_PATH = '{base_path}'
            if BASE_PATH in con:
                con = con.replace(BASE_PATH, get_config(['base_path'], False))

        return con


if __name__ == '__main__':
    print(get_config(["solver", "z3_solution_parser_command"]))
