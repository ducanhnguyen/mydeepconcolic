class covered_layer:
    def __init__(self):
        self.__index = None
        self.__layer = None
        self.__input = None
        self.__output = None

    def set_index(self, index):
        self.__index = index

    def get_index(self):
        return self.__index

    def set_layer(self, layer):
        self.__layer = layer

    def get_layer(self):
        return self.__layer

    def __repr__(self):
        return "[layer name = '{0}', index = {1}, input = {2}, output = {3}]\n".format(self.get_layer().name,
                                                                                       self.get_index(),
                                                                                       self.get_input(),
                                                                                       self.get_output())

    def get_input(self):
        return self.get_layer().input

    def get_output(self):
        return self.get_layer().output

    def get_output_shape(self):
        return self.get_layer().output_shape
