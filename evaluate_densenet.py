import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import keras
import keras.backend as K
import explainer.QuantityInterests as Q
from explainer.Attribution import KerasAttr
from ruamel.yaml import YAML
from pathlib import Path
from densenet import densenet121_model
import scipy
from explainer.Influence import KerasInflExp
import explainer as E
import explainer.Influence as Infl
import explainer.QuantityInterests as Q
import explainer.ExpertUnit as EU
import explainer.Visualization as V
from tqdm import trange
from AttrWrapper import SaliencyMapWrapper, IntergratedGradWrapper
from AttrWrapper import SmoothGradWrapper, GradCAMWrapper
from AttrWrapper import NeuronInflWrapper, ChannelInflWrapper

from evaluate_metrics import K_Necessity, K_Sufficiency

TFconfig = tf.ConfigProto()
TFconfig.gpu_options.allow_growth = True
sess = tf.Session(config=TFconfig)
K.tensorflow_backend.set_session(sess)
LONGTERM = "/longterm/zifanw/Caltech/"
COLOR_MEAN = [123.68, 116.779, 103.939]

parser = argparse.ArgumentParser(
    description='Attribution Evaluations for Densenet on Caltech 256')
parser.add_argument('--method',
                    type=str,
                    default='saliencymap',
                    help='the mothod of attribution')
parser.add_argument('--K_value',
                    type=float,
                    default=0.1,
                    help='the score of necessity')


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


def load_dataset(max_instance=None):
    X_test = np.load(LONGTERM + "test_x.npy").astype('float')
    Y_test = np.load(LONGTERM + "test_y.npy")

    for i in range(3):
        X_test[:, :, :, i] -= COLOR_MEAN[i]

    Y_test = to_categorical(Y_test, num_classes=257)

    if max_instance is not None:
        X_test = X_test[:max_instance]
        Y_test = Y_test[:max_instance]

    return X_test, Y_test


def necessity_stat(infl_model,
                   target,
                   attr_fn,
                   blur=0,
                   max_num_pixel=1000,
                   step=1,
                   metric=K_Necessity,
                   K=0.1):
    # get the original attribution map
    attribution_map = attr_fn(target)
    attribution_map = V.point_cloud(attribution_map, threshold=0)[0]

    # find the prediction class to determine
    # which class to pick for the pre-softmax score
    prediction = attr_fn.qoi.get_class()

    if blur > 0:
        attribution_map = scipy.ndimage.gaussian_filter(attribution_map,
                                                        sigma=blur)

    # sort the pixels by its attribution score
    pos_attr_id = largest_indices(attribution_map, max_num_pixel)
    result = []
    for j in range(0, pos_attr_id.shape[1], step):
        img = target[0].copy()
        if j > 0:
            img[pos_attr_id[0][:j], pos_attr_id[1][:j], :] = 0
        result.append(img[None, :])
    removed = np.vstack(result)
    pre_softmax_scores = infl_model.get_activation(removed,
                                                   "fc6")[:, prediction]

    # Measure the necessity
    if metric is not None:
        return metric(pre_softmax_scores, resolution=step, K=K)
    else:
        return pre_softmax_scores


def main():
    args = parser.parse_args()

    model = load_model()
    X_test, _ = load_dataset(max_instance=2000)

    infl_model = KerasInflExp(model, channel_first=False, verbose=False)

    default_qoi = Q.ClassInterest(0)
    if args.method == 'saliencymap':
        attr_fn = SaliencyMapWrapper(model, default_qoi)

    elif args.method == 'integratedgrad':
        attr_fn = IntergratedGradWrapper(model, default_qoi)

    elif args.method == 'smoothgrad':
        attr_fn = SmoothGradWrapper(model, default_qoi)

    elif args.method == 'gradcam':
        attr_fn = GradCAMWrapper(model,
                                 default_qoi,
                                 "concatenate_58",
                                 channel_first=False)

    elif args.method == 'neuroninfl':
        attr_fn = NeuronInflWrapper(model, default_qoi, "concatenate_55")
        attr_fn.find_experts(X_test, 'fc6')

    elif args.method == 'channelinfl':
        attr_fn = ChannelInflWrapper(model, default_qoi, "concatenate_55")
        attr_fn.find_experts(X_test, 'fc6')

    else:
        raise ValueError("not a supported attribution methods")

    score = []
    for i in trange(len(X_test)):
        target = X_test[i:i + 1] if i < len(X_test) - 1 else X_test[i:]
        s = necessity_stat(infl_model,
                           target,
                           attr_fn,
                           blur=0,
                           max_num_pixel=1000,
                           step=20,
                           metric=K_Necessity,
                           K=args.K_value)
        score.append(s)
    score = np.array(score)
    np.save(args.method + "_" + str(args.K_value) + "-Necessity.npy", score)
    print("Attribution : {0:s}, {1:.2f}-Necessity: {2:f}".format(
        args.method, args.K_value, score.mean()))


if __name__ == "__main__":
    main()