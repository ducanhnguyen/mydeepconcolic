import tensorflow as tf

from src.ConfigController import ConfigController

'''
Store configuration of a thread
'''
class ThreadConfig:

    def __init__(self):
        pass

    def load_config_from_file(self):
        assert (self.config_path != None)
        assert (self.thread_idx != None and self.thread_idx >= 0)

        self.graph = tf.get_default_graph()
        self.thread_name = f'thread_{self.thread_idx}'
        self.should_plot = True

        self.dataset = self.get_config(["dataset"])

        self.feature_lower_bound = self.get_config(["constraint_config", "feature_lower_bound"])
        self.feature_upper_bound = self.get_config(["constraint_config", "feature_upper_bound"])
        self.delta_lower_bound = self.get_config(["constraint_config", "delta_lower_bound"])
        self.delta_upper_bound = self.get_config(["constraint_config", "delta_upper_bound"])
        self.delta_prefix_name = self.get_config(["constraint_config", "delta_prefix_name"])
        self.feature_input_type = self.get_config(["constraint_config", "feature_input_type"])

        self.attacked_neuron = dict()
        self.attacked_neuron['enabled'] = self.get_config(["attacked_neuron", "enabled"])
        self.attacked_neuron['layer_index'] = self.get_config(["attacked_neuron", "layer_index"])
        self.attacked_neuron['neuron_index'] = self.get_config(["attacked_neuron", "neuron_index"])
        self.attacked_neuron['lower_bound'] = self.get_config(["attacked_neuron", "lower_bound"])
        self.attacked_neuron['upper_bound'] = self.get_config(["attacked_neuron", "upper_bound"])

        self.seed_file = self.get_config(["files", "seed_file_path"], recursive = True)
        self.true_label_seed_file = self.get_config(["files", "true_label_seed_file_path"], recursive = True)
        self.seed_index_file = self.get_config(["files", "seed_index_file_path"], recursive = True)
        self.analyzed_seed_index_file_path = self.get_config(["files", "analyzed_seed_index_file_path"], recursive = True)
        self.selected_seed_index_file_path = self.get_config(["files", "selected_seed_index_file_path"], recursive = True)
        self.new_image_file_path = self.get_config(["files", "new_image_file_path"], recursive = True)
        self.comparison_file_path = self.get_config(["files", "comparison_file_path"], recursive = True)
        self.new_csv_image_file_path = self.get_config(["files", "new_csv_image_file_path"], recursive = True)
        self.old_csv_image_file_path = self.get_config(['files', 'old_csv_image_file_path'], recursive = True)
        self.old_image_file_path = self.get_config(['files', 'old_image_file_path'], recursive = True)
        self.full_comparison = self.get_config(['files', 'full_comparison'], recursive = True)

        self.constraints_file = self.get_config(["z3", "constraints_file_path"], recursive = True)
        self.z3_solution_file = self.get_config(["z3", "z3_solution_path"], recursive = True)
        self.z3_normalized_output_file = self.get_config(["z3", "z3_normalized_solution_path"], recursive = True)
        self.z3_path = self.get_config(["z3", "z3_solver_path"], recursive = True)
        self.z3_solution_parser_command = self.get_config(["z3", "z3_solution_parser_command"], recursive = True)
        self.z3_time_out = self.get_config(["z3", "time_out"])

    def get_config(self, arr, recursive=False):
        con = ConfigController().get_config(attribute=arr, config_path=self.config_path, recursive=recursive)

        if isinstance(con, str) and recursive:
            # replacement
            if hasattr(self, 'seed_index'):
                con = con.replace('{seed_index}', str(self.getSeedIndex()))

            # replacement
            if hasattr(self, 'delta_upper_bound'):
                con = con.replace('{delta}', str(self.delta_upper_bound))

            # replacement
            if hasattr(self, 'thread_idx'):
                con = con.replace('{thread_idx}', str(self.thread_idx))

            # replacement
            from random import randint
            con = con.replace('{time}', str(randint(0, 10000)))

            # replacement
            if hasattr(self, "attacked_neuron"):
                con = con.replace('{layer_idx}', str(self.attacked_neuron['layer_index']))
            if hasattr(self, "attacked_neuron"):
                con = con.replace('{unit_idx}', str(self.attacked_neuron['neuron_index']))

        return con

    def setSeedIndex(self, seed_index):
        self.seed_index = seed_index

    def getSeedIndex(self):
        return self.seed_index

    def setId(self, thread_idx):
        self.thread_idx = thread_idx

    def setConfigFile(self, config_path):
        self.config_path = config_path