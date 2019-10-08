from explainer.Influence import KerasInflExp
import explainer as E
import explainer.Influence as Infl
import explainer.QuantityInterests as Q
import explainer.ExpertUnit as EU
import keras.backend as K
import explainer.QuantityInterests as Q
from explainer.Attribution import KerasAttr
import numpy as np
from densenet import densenet121_model
COLOR_MEAN = [123.68, 116.779, 103.939]
LONGTERM = "/longterm/zifanw/Caltech/"


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    indices = np.unravel_index(indices, ary.shape)
    x, y = indices[0], indices[1]
    xx = x.reshape((1, x.shape[0]))
    yy = y.reshape((1, y.shape[0]))
    result = [xx, yy]
    result = np.vstack(result)
    return result


def load_model():
    img_rows, img_cols = 224, 224
    img_channels = 3
    nb_classes = 257

    model = densenet121_model(img_rows=img_rows,
                              img_cols=img_cols,
                              color_type=img_channels,
                              num_classes=nb_classes,
                              weight_path_prefix="../")
    print('=' * 50)
    print('Model is created')
    #     model.summary()

    model.load_weights(LONGTERM + '/weights/DenseNetCaltech.h5')

    return model


class AttributionWrapper:
    def __init__(self, model, qoi, batch_size, mul_with_input):
        self.batch_size = batch_size
        self.qoi = qoi
        self.mul_with_input = mul_with_input
        self.attr_fn = KerasAttr(model, qoi, verbose=False)

    def set_qoi(self, new_qoi):
        self.qoi = new_qoi
        self.attr_fn.set_qoi(self.qoi)

    def reset(self, new_qoi, **kwargs):
        self.set_qoi(new_qoi)

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
    def __init__(self,
                 model,
                 qoi,
                 batch_size=16,
                 mul_with_input=True,
                 imagenet_baseline=True):
        super(IntergratedGradWrapper, self).__init__(model, qoi, batch_size,
                                                     mul_with_input)
        self.imagenet_baseline = imagenet_baseline

    def __call__(self, X):
        baseline = np.zeros_like(X)
        if self.imagenet_baseline == True:
            baseline[:, :, :, 0] -= COLOR_MEAN[0]
            baseline[:, :, :, 1] -= COLOR_MEAN[1]
            baseline[:, :, :, 2] -= COLOR_MEAN[2]
        return self.attr_fn.integrated_grad(X,
                                            batch_size=self.batch_size,
                                            baseline=baseline,
                                            mul_with_input=self.mul_with_input)


class SmoothGradWrapper(AttributionWrapper):
    def __init__(self, model, qoi, batch_size=16, mul_with_input=True):
        super(SmoothGradWrapper, self).__init__(model, qoi, batch_size,
                                                mul_with_input)

    def __call__(self, X):
        return self.attr_fn.smooth_grad(X,
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

    def set_qoi(self, new_qoi):
        self.qoi = new_qoi

    def set_doi(self, doi):
        self.attr_fn.set_doi_type(doi)

    def reset(self, new_qoi, X_test=None):
        if self.qoi.get_class() != new_qoi.get_class():
            # model = load_model()
            # self.attr_fn = KerasInflExp(model,
            #                             channel_first=self.channel_first,
            #                             verbose=False)
            self.set_qoi(new_qoi)
            self.experts = EU.Expert("Neuron")
            if X_test is not None:
                self.find_experts(X_test)

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

    def set_qoi(self, new_qoi):
        self.qoi = new_qoi

    def set_doi(self, doi):
        self.attr_fn.set_doi_type(doi)

    def reset(self, new_qoi, X_test=None):
        if self.qoi.get_class() != new_qoi.get_class():
            # model = load_model()
            # self.attr_fn = KerasInflExp(model,
            #                             channel_first=self.channel_first,
            #                             verbose=False)
            self.set_qoi(new_qoi)
            self.experts = EU.Expert("Channel")
            if X_test is not None:
                self.find_experts(X_test)

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


class LocalOptimal(AttributionWrapper):
    def __init__(self,
                 infl_model,
                 qoi,
                 prob_layer='fc6',
                 search_space=1000,
                 max_step=1000):
        super(LocalOptimal, self).__init__(None, qoi, 1, False)
        self.infl_model = infl_model
        self.prob_layer = prob_layer
        self.search_space = search_space
        self.optimal_path = []
        self.max_step = self.max_step

    def find_optimal_path(self, X):
        self.optimal_path = []
        img = X[0].copy()
        img_width, img_height = img.shape[0], img.shape[1]
        for _ in range(self.max_step):
            candidate_space = largest_indices(
                np.random.randn(img_width, img_height), 2 * self.search_space)

            best_score = 1e10
            best_candidate = None

            for n, candidate in enumerate(candidate_space):
                if n > self.search_space:
                    break
                if img[candidate[0], candidate[1], :].mean() == 0:
                    continue
                perturbation = img.copy()
                perturbation[candidate[0], candidate[1], :] = 0
                score = self.infl_model.get_activation(perturbation[None, :],
                                                       self.prob_layer)
                score = score[0, self.qoi.get_class()]

                if score < best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate is not None:
                self.optimal_path.append(best_candidate)
                img[best_candidate[0], best_candidate[1], :] = 0

    def __call__(self, X):
        self.find_optimal_path(X)
        result = np.zeros_like(X)
        high_score = 1e5
        for point in self.optimal_path:
            result[:, point[0], point[1], 0] = high_score
            high_score -= 0.1
        return result


class RandomAttribution:
    def __call__(self, X):
        return np.random.randn(X.shape[0], X.shape[1], X.shape[2])
