import json

def get_config(attributes, config_path = './config_osx.json', recursive = True):
    with open(config_path, 'r') as f:
        config = json.load(f)

        con = config
        for attribute in attributes:
            con = con[attribute]

        if isinstance(con,str) and recursive:
            DATASET_SIGNAL = '{dataset}'
            if DATASET_SIGNAL in con:
                con = con.replace(DATASET_SIGNAL, get_config(['dataset'], config_path, False))

            BASE_PATH = '{base_path}'
            if BASE_PATH in con:
                con = con.replace(BASE_PATH, get_config(['base_path'], config_path, False))

        return con

if __name__ == '__main__':
    print(get_config(["solver", "z3_solution_parser_command"]))