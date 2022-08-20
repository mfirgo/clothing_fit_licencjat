from hierarchical_size_model_new import *
from hierarchical_model_return_status import *

class HierarchicalFullModel:
    def __init__(self, size_model, return_status_model):
        self.size_model = size_model
        self.return_status_model = return_status_model

    def predict(self, test):
        