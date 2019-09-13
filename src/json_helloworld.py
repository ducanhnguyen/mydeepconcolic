import json

def get_config(attributes, config_path = './config.json'):
    DATASET_SIGNAL = '{dataset}'
    with open(config_path, 'r') as f:
        config = json.load(f)

        con = config
        for attribute in attributes:
            con = con[attribute]

        if isinstance(con,str):
            if DATASET_SIGNAL in con:
                con = con.replace(DATASET_SIGNAL, get_config(['dataset']))

        return con

if __name__ == '__main__':
    print(get_config(["solver", "z3_solution_parser_command"]))