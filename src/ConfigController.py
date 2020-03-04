import json
from python_json_config import ConfigBuilder

'''
Used for reading, writing configurations from file
'''
class ConfigController:
    def __init__(self):
        pass

    def get_config(self, attribute, config_path='./config_ubuntu.json', recursive=True):
        # create config parser
        builder = ConfigBuilder()
        # parse config
        config = builder.parse_config(config_path)
        con = config.get(attribute)

        if isinstance(con, str) and recursive:
            DATASET_SIGNAL = '{dataset}'
            if DATASET_SIGNAL in con:
                con = con.replace(DATASET_SIGNAL, self.get_config(['dataset'], config_path, False))

            BASE_PATH = '{base_path}'
            if BASE_PATH in con:
                con = con.replace(BASE_PATH, self.get_config(['base_path'], config_path, False))
        return con

    def update_and_write(self, key, value, config_path='./config_ubuntu.json', ):
        # create config parser
        builder = ConfigBuilder()
        # parse config
        config = builder.parse_config(config_path)
        # access elements
        config.update(key, value)
        # create a beautiful json file
        updated_json = config.to_json()
        beautiful_updated_json = json.loads(updated_json)
        beautiful_updated_json = json.dumps(beautiful_updated_json, indent=4)

        with open(config_path, mode='w') as f:
            f.write(beautiful_updated_json)

        return beautiful_updated_json


if __name__ == '__main__':
    jsoncontroller = ConfigController()
    print(jsoncontroller.get_config("files.new_image_file_path"))
    #jsoncontroller.update_and_write(key="attacked_neuron.lower_layer_index", value = 20)