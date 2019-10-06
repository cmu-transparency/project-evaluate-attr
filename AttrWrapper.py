from explainer.Influence import KerasInflExp
import explainer as E
import explainer.Influence as Infl
import explainer.QuantityInterests as Q
import explainer.ExpertUnit as EU
import keras.backend as K
import explainer.QuantityInterests as Q
from explainer.Attribution import KerasAttr
import numpy as np


class AttributionWrapper:
    def __init__(self, model, qoi, batch_size, mul_with_input):
        self.batch_size = batch_size
        self.qoi = qoi
        self.mul_with_input = mul_with_input
        self.attr_fn = KerasAttr(model, qoi, verbose=False)

    def set_qoi(self, new_qoi):
        self.qoi = new_qoi

    def __call__(self, x):
        raise NotImplementedError


class SaliencyMapWrapper(AttributionWrapper):
    def __init__(self, model, qoi, batch_size=16, mul_with_input=True):
        super(SaliencyMapWrapper, self).__init__(model, qoi, batch_size,
                                                 mul_with_input)

    def __call__(self, X):
        return self.attr_fn.saliency_map(X,
                                         batch_size=self.batch_size,
                                         mul_with_input=self.mul_with_input)


class IntergratedGradWrapper(AttributionWrapper):
    def __init__(self, model, qoi, batch_size=16, mul_with_input=True):
        super(IntergratedGradWrapper, self).__init__(model, qoi, batch_size,
                                                     mul_with_input)

    def __call__(self, X):
        return self.attr_fn.integrated_grad(X,
                                            batch_size=self.batch_size,
                                            mul_with_input=self.mul_with_input)


class SmoothGradWrapper(AttributionWrapper):
    def __init__(self, model, qoi, batch_size=16, mul_with_input=True):
        super(SmoothGradWrapper, self).__init__(model, qoi, batch_size,
                                                mul_with_input)

    def __call__(self, X):
        return self.attr_fn.integrated_grad(X,
                                            batch_size=self.batch_size,
                                            mul_with_input=self.mul_with_input)


class GradCAMWrapper(AttributionWrapper):
    def __init__(self,
                 model,
                 qoi,
                 layer_name,
                 channel_first=False,
                 batch_size=16,
                 mul_with_input=True):
        super(GradCAMWrapper, self).__init__(model, qoi, batch_size,
                                             mul_with_input)
        self.layer_name = layer_name
        self.channel_first = channel_first

    def __call__(self, X):
        return self.attr_fn.gradcam(X,
                                    layer_name=self.layer_name,
                                    batch_size=self.batch_size,
                                    channel_first=self.channel_first)


class NeuronInflWrapper(AttributionWrapper):
    def __init__(self,
                 model,
                 qoi,
                 layer_name,
                 channel_first=False,
                 topK=10,
                 batch_size=16,
                 mul_with_input=True):
        super(NeuronInflWrapper, self).__init__(model, qoi, batch_size,
                                                mul_with_input)
        self.layer_name = layer_name
        self.channel_first = channel_first
        self.topK = topK
        self.attr_fn = KerasInflExp(model,
                                    channel_first=self.channel_first,
                                    verbose=False)
        self.experts = EU.Expert("Neuron")
        self.top_idx = np.arange(self.topK)

    def set_doi(self, doi):
        self.attr_fn.set_doi_type(doi)

    def find_experts(self, X_test, from_layer='fc6'):
        raw_infl = self.attr_fn.internal_infl(X_test,
                                              from_layer=from_layer,
                                              wrt_layer=self.layer_name,
                                              interest_mask=self.qoi)
        self.experts(raw_infl)
        self.tops = self.experts.get_tops()
        self.inter_unit_wts = self.experts.get_inter_unit_wts(self.top_idx)

    def __call__(self, X):
        return self.attr_fn.multi_unit_visualizaiton(
            X,
            self.layer_name,
            self.tops[:self.topK],
            inter_unit_wts=self.inter_unit_wts,
            infl_as_wts=False,
            multiply_with_input=self.mul_with_input)


class ChannelInflWrapper(AttributionWrapper):
    def __init__(self,
                 model,
                 qoi,
                 layer_name,
                 channel_first=False,
                 topK=3,
                 batch_size=16,
                 mul_with_input=True):
        super(ChannelInflWrapper, self).__init__(model, qoi, batch_size,
                                                 mul_with_input)
        self.layer_name = layer_name
        self.channel_first = channel_first
        self.topK = topK
        self.attr_fn = KerasInflExp(model,
                                    channel_first=self.channel_first,
                                    verbose=False)
        self.experts = EU.Expert("Channel")
        self.top_idx = np.arange(self.topK)

    def set_doi(self, doi):
        self.attr_fn.set_doi_type(doi)

    def find_experts(self, X_test, from_layer='fc6'):
        raw_infl = self.attr_fn.internal_infl(X_test,
                                              from_layer=from_layer,
                                              wrt_layer=self.layer_name,
                                              interest_mask=self.qoi)
        self.experts(raw_infl,
                     heuristic='max',
                     channel_first=self.channel_first)
        self.tops = self.experts.get_tops()
        self.inter_unit_wts = self.experts.get_inter_unit_wts(self.top_idx)

    def __call__(self, X):
        return self.attr_fn.multi_unit_visualizaiton(
            X,
            self.layer_name,
            self.tops[:self.topK],
            inter_unit_wts=self.inter_unit_wts,
            infl_as_wts=True,
            interest_mask=self.qoi,
            multiply_with_input=self.mul_with_input)