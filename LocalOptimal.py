import numpy as np

class LocalOptimal(object):
    def __init__(self, infl_model, score_layer, beam_width=100):
        self.infl_model = infl_model
        self.score_layer = score_layer
        self.beam_width = beam_width