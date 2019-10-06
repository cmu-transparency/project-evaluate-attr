# -*- coding: utf-8 -*-
from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D, concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
import numpy as np
from sklearn.metrics import log_loss
from keras.utils.np_utils import to_categorical
import argparse
from custom_layers.scale_layer import Scale
from keras.models import load_model
#from load_cifar10 import load_cifar10_data
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from functions import *
from densenet import densenet121_model
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from time import time
from ExplainToolKeras import *


parser = argparse.ArgumentParser(description='Pneumonia DenseNet')
parser.add_argument(
    '--mode',
    type=str,
    default='Train',
    help='Train or Test')
parser.add_argument(
    '--epoch',
    type=int,
    default=10,
    help='Training epoches')
parser.add_argument(
    '--use_pneumonia',
    type=bool,
    default=True,
    help='if this is the pneumonia dataset')
parser.add_argument(
    '--layer', type=str, default='relu2_1_x2', help='internal layer to explain')
parser.add_argument(
    '--weights_dir',
    type=str,
    default='weights/',
    help='pretrained weights directory')
parser.add_argument(
    '--create_explain_batch',
    type=bool,
    default=False,
    help='if True, sample from the test data and create a mini batch for explaination')
parser.add_argument(
    '--explain_batch_path',
    type=str,
    default='./explain_batch.npy',
    help='a pre-sampled batch from the test dataset')
parser.add_argument(
    '--lr', type=float, default=1e-6, help='initial learning rate')
parser.add_argument(
    '--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument(
    '--batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument(
    '--all_resource',
    type=int,
    default=0,
    help='if deploy all the GPU resources')
parser.add_argument(
    '--training_mode',
    type=str,
    default='Explain',
    help='Train, Test ,Explain_I and Explain_V mode')
parser.add_argument(
    '--save_influence',
    type=bool,
    default=False,
    help='If True, save the raw Influence to influence_raw_path')
parser.add_argument(
    '--influence_raw_path',
    type=str,
    default='influence_raw/',
    help='The path to save the raw influence results')
parser.add_argument(
    '--influence_path',
    type=str,
    default='influence/',
    help='The path to save the averaged influence results')
parser.add_argument(
    '--visualization_path',
    type=str,
    default='visualization/',
    help='The path to save the visualization results')
parser.add_argument(
    '--normalization',
    type=str,
    default='None',
    help=
    '\'None\': do not normlize the Influence. \'Max\': normalize the influence with max. \'Sum\': normlize the influence with sum'
)
parser.add_argument(
    '--influence_neuron_num',
    type=int,
    default=10,
    help='The num of significant neurons we want to add into the result')
parser.add_argument(
    '--visulization_class',
    type=int,
    default=0,
    help='The class to visualize the influence.')
parser.add_argument(
    '--visulization_index',
    type=str,
    default=None,
    help='The index of neurons to visualize')
parser.add_argument(
    '--visulization_direction',
    type=str,
    default='high',
    help='high: neurons with highest gradients value in the influence. low: neurons with lowest gradients value in the influence')
parser.add_argument(
    '--use_channel',
    type=bool,
    default=False,
    help='if True, use channel influence')
parser.add_argument(
    '--guided_bp',
    type=bool,
    default=False,
    help='If true, replace backpropagation with guided version')


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return  tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

class GuidedBackprop():
    GuidedReluRegistered = False
    def __init__(self, graph, session, y, x):
        super(GuidedBackprop, self).__init__(graph, session, y, x)
        self.x = x
        if GuidedBackprop.GuidedReluRegistered is False:
          #### Acknowledgement to Chris Olah ####
            @tf.RegisterGradient("GuidedRelu")
            def _GuidedReluGrad(op, grad):
                gate_g = tf.cast(grad > 0, "float32")
                gate_y = tf.cast(op.outputs[0] > 0, "float32")
                return gate_y * gate_g * grad
            GuidedBackprop.GuidedReluRegistered = True
            with graph.as_default():
                saver = tf.train.Saver()
                saver.save(session, '/tmp/guided_backprop_ckpt')

            graph_def = graph.as_graph_def()
            self.guided_graph = tf.Graph()
            with self.guided_graph.as_default():
                self.guided_sess = tf.Session(graph = self.guided_graph)
                with self.guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
                # Import the graph def, and all the variables.
                    tf.import_graph_def(graph_def, name='')
                    saver.restore(self.guided_sess, '/tmp/guided_backprop_ckpt')
                    imported_y = self.guided_graph.get_tensor_by_name(y.name)
                    imported_x = self.guided_graph.get_tensor_by_name(x.name)
                    self.guided_grads_node = tf.gradients(imported_y, imported_x)[0]

def compute_influence(model, num_class=2, layer_name='fc6'):
    class_tensor = model.get_layer("prob").output
    class_split = tf.split(class_tensor, num_class, axis=1)  # split class-wise
    layer_output = model.get_layer(layer_name).output
    #print (layer_output)
    #layer_influence = [K.gradients(i, layer_output) for i in class_split]
    layer_influence = [tf.gradients(xs=layer_output, ys=i) for i in class_split]
    return layer_influence 



def main(args):
    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 2
    batch_size = 16
    nb_epoch = args.epoch
    #the number of class for Pneunomia dataset is 2 and for OCT is 4
    class_num = 2 if args.use_pneumonia else 4

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    # X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    # print (Y_train[0])

    #Load our model
    model = densenet121_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
    #model.summary()

    if args.mode == 'Train':
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_Pneumonia_dataset()
        #X_train, Y_train, X_test, Y_test = load_Pneumonia_dataset('/longterm/zifanw/OCT_data5K/')
        #One-hot encoding

#         X_train = np.load('/longterm/zifanw/IDC_data/train_x.npy')
#         Y_train = np.load('/longterm/zifanw/IDC_data/train_y.npy')
#         X_test, X_train = X_train[:20000], X_train[20000:]
#         Y_test, Y_train = Y_train[:20000], Y_train[20000:]
       
        #X_train = np.vstack([X_train, X_valid])
     
        Y_train = to_categorical(Y_train, num_classes=num_classes)
        #Y_valid = to_categorical(Y_valid, num_classes=num_classes)
        Y_test = to_categorical(Y_test, num_classes=num_classes)
        #Y_train = np.vstack([Y_train, Y_valid])

        #index = np.random.permutation(X_train.shape[0])
        #X_train = X_train[index]
        #Y_train = Y_train[index]
        #X_test = np.vstack([X_test, X_train[:5000]])
        #Y_test = np.vstack([Y_test, Y_train[:5000]])
        #X_train = X_train[5000:]
        #Y_train = Y_train[5000:]	

        # Start Fine-tuning
        model.load_weights('weights/Pneumonia2.h5')
        weights_file = 'weights/Pneumonia3.h5'
        lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                    cooldown=0, patience=5, min_lr=1e-7)
        model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                  save_weights_only=True, verbose=1)

        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        callbacks = [tensorboard,lr_reducer, model_checkpoint]
        model.fit(X_train, Y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                shuffle=True,
                verbose=1,
                validation_data=(X_test, Y_test),
                callbacks=callbacks,
                )
        print("----------------Training is Complete------------------")
      

    elif args.mode == 'Test':
        X_test, Y_test = load_test_dataset('SPneumonia/Pneumonia_Apr8/')
        Y_test = to_categorical(Y_test, num_classes=num_classes)
        model.load_weights('SPneumonia/Pneumonia_Apr8/P_apr8.h5')
        print("----------------Model is restored from Pneumonia.h5---")
        #Make predictions
        print("---------Start to evaluate with the Test Dataset----")
        Y_pred = model.predict(X_test, batch_size=16)
        scores = model.evaluate(X_test, Y_test)
        cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
        np.save('confusion_matrix/confusion_matrix.npy',cm)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print('--------Confusion Matrix----------------------------')
        print(cm)
        print('--------Test Session is Complete--------------------')
        explain_index = np.load('./explain_batch.npy')
        print (explain_index)
        inputs = X_test[explain_index[-1:]]
        Y_pred = model.predict(X_test[explain_index], batch_size=4)
        print (Y_pred)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model.load_weights('weights/Pneumonia.h5')
            VaniliaGrad(model=model, sess=sess, inputs=inputs, vis=True, c=3)
            SmoothGrad(model=model, sess=sess, inputs=inputs, p=0.2, vis=True, c=3)
            IntegratedGrad(model=model, sess=sess, inputs=inputs, vis=True, c=3)
            GradCam(model=model, sess=sess, inputs=inputs, layer_name='conv5_blk_scale', c=3)

    elif args.mode == 'Explain_I':
        X_test, Y_test = load_test_dataset('SPneumonia/Pneumonia_Apr8/')
        Y_test = to_categorical(Y_test, num_classes=num_classes)
        model.load_weights('SPneumonia/Pneumonia_Apr8/P_apr8.h5')
        influence = compute_influence(model, num_class=num_classes, layer_name=args.layer)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        print (X_test.shape)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model.load_weights('SPneumonia/Pneumonia_Apr8/P_apr8.h5')
            print("----------------Model is restored from Pneumonia.h5---")
            print("---------Start to evaluate with Influence of the Test Dataset----")
            test_influence = []
            test_gen = my_generator(x=X_test, batch_size=args.batch_size)
            for _ in range(int(X_test.shape[0]/args.batch_size)+1):
                batch_test = next(test_gen)
                batch_influence = sess.run(influence, feed_dict={model.input: batch_test})
                test_influence.append(np.asarray(batch_influence))
        print("--------Complete computing the Influence----------------------")
        test_influence = np.concatenate(test_influence, axis=2)  # (Class_Num, 1, N, Neurons)
        test_influence = test_influence[:,0]  # get rid of redundant dimension
                                              #(Class_Num, N, Neurons) or (Class_Num, N, H, W, C)
        print("--------The shape of the raw Influence is----------------------")
        print(test_influence.shape)
        if args.save_influence:
            save_path = args.influence_raw_path + args.layer + '_' +'Influence_raw.npy'
            np.save(save_path, test_influence)
            print(
                "---> Complete Influence computing and the raw Influence result is saved to "
                + save_path)

        #if we do not normalize the influence, we just sort them to find the most significant neuron by its influence value.
        if args.normalization == 'None':
            #find the index of the most important neurons
            significant_neuron_ids = sort_by_influence(args, test_influence,
                                                       class_num)
          
            np.save(
                args.influence_path + args.layer + '_' +
                "significant_neuron_ids.npy", significant_neuron_ids)
            print("---> The averaged Influence for " + args.layer +
                  " without normalizations is saved to " + args.influence_path + args.layer + '_' +
                  "significant_neuron_ids.npy")
            return 

        #if we normalize the influence before sorting.
        # we do:
        # 1. devide the influence into the positive part and the negative part
        # 2. normalize with the max or sum of that specific sample
        # e.g. X = [[1,1], [2,1]]
        #      after normalizatin with max: X = [[1,1], [1, 0.5]]
        #      after normalizatin with sum: X = [[0.5, 0.5], [0.67, 0.33]]
        # 3. sort the positive influence and negative influence, respectively.
        
        elif args.normalization == 'Channel':
            significant_neuron_ids = sort_by_channel(test_influence)[:,:,:args.influence_neuron_num]
          
            np.save(
                args.influence_path + args.layer + '_' +
                "significant_channel_ids.npy", significant_neuron_ids)
            print("---> The averaged Influence for " + args.layer +
                  " with channel normalizations is saved to " + args.influence_path + args.layer + '_' +
                  "significant_channel_ids.npy")
            return 

            

        else:
            mask = np.zeros_like(test_influence)
            # devide the influence into the positive and the negative part
            test_influence_positive = np.maximum(mask, test_influence)
            test_influence_negative = np.minimum(mask, test_influence)
            for i in range(
                    test_influence_positive.shape[0]):  # class dimension
                #normalization
                for n in range(test_influence_positive.shape[1]):  # N dimension
                    influence_sum_positive = np.sum(
                        test_influence_positive[i, n])
                    influence_max_positive = np.amax(
                        test_influence_positive[i, n])
                    influence_sum_negative = np.sum(
                        test_influence_negative[i, n])
                    influence_max_negative = np.amin(
                        test_influence_negative[i, n])
                    if args.normalization == 'Max':
                        test_influence_positive[i, n] /= influence_max_positive
                        test_influence_negative[i,
                                                n] /= -influence_max_negative
                    elif args.normalization == 'Sum':
                        test_influence_positive[i, n] /= influence_sum_positive
                        test_influence_negative[i,
                                                n] /= -influence_sum_negative
            significant_neuron_ids_positive = sort_by_influence(
                args, test_influence_positive, class_num)
            significant_neuron_ids_negative = sort_by_influence(
                args, test_influence_negative, class_num)
            print (significant_neuron_ids_positive.shape)
            np.save(
                args.influence_path + args.layer + '_norm_' +
                "significant_neuron_ids_positive.npy",
                significant_neuron_ids_positive)
            print("---> The normalized positive Influence for " + args.layer +
                  " is saved to " + args.influence_path + args.layer + '_norm_' +
                  "significant_neuron_ids_positive.npy")
            np.save(
                args.influence_path + args.layer + '_norm_' +
                "significant_neuron_ids_negative.npy",
                significant_neuron_ids_negative)
            print("---> The normalized negative Influence for " + args.layer +
                  " is saved to " + args.influence_path + args.layer + '_norm_' +
                  "significant_neuron_ids_negative.npy")

    elif args.mode == 'Explain_V':
        X_test, Y_test = load_test_dataset('SPneumonia/Pneumonia_Apr8/')
        Y_test = to_categorical(Y_test, num_classes=num_classes)        

        # create explain batch
        if args.create_explain_batch:
            e_index = np.arange(X_test.shape[0])
            e_index = np.random.permutation(e_index)
            e_index = e_index[:4]
            np.save(args.explain_batch_path, e_index)
            print("---> Explain Batch indices are saved to " +
                  args.explain_batch_path)
        else:
            e_index = np.load(args.explain_batch_path)
            print("---> Explain Batch indices in the test data set are: ", e_index)

        explain_x = X_test[e_index]
        explain_y = Y_test[e_index]

        
        print ("The groudtruth for explain batches are")
        print (explain_y)
        
        model.load_weights('SPneumonia/Pneumonia_Apr8/P_apr8.h5')
        Y_pred = model.predict(explain_x, batch_size=2)
        print ("Prediction results")
        print(np.argmax(Y_pred, axis=1))
        
        input_tensor = model.get_layer('data').output
        output_tensor = model.get_layer('prob').output
        layer_output = model.get_layer(args.layer).output
        
        if args.use_channel:
            influence = tf.gradients(ys=output_tensor[:,args.visulization_class],
                                     xs=layer_output)[0]
            
        
        index = np.load(args.influence_path + args.visulization_index)

        index = index[args.visulization_class]
        if args.visulization_direction == 'high':
            index = index[0]
            if args.use_channel:
                influence = tf.maximum(tf.zeros_like(influence), influence)
        elif  args.visulization_direction == 'low':
            index = index[1]
            if args.use_channel:
                influence = tf.minimum(tf.zeros_like(influence), influence)
     

        visualization_result = []
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model.load_weights('SPneumonia/Pneumonia_Apr8/P_apr8.h5')
            if not args.guided_bp:
                if 'fc' in args.layer or 'prob' in args.layer:
                    visualization = compute_grad(x=input_tensor, 
                                                 y=layer_output, 
                                                 ids=index, 
                                                 type='fc')
                else:
                    if args.use_channel:
                        print (influence.shape)
                        visualization = compute_grad(x=input_tensor, 
                                                     y=layer_output, 
                                                     ids=index, 
                                                     type='conv',
                                                     initial_grad=influence)
                    else:
                        visualization = compute_grad(x=input_tensor, 
                                                     y=layer_output, 
                                                     ids=index, 
                                                     type='conv')
                print("----------------Model is restored---")
                print("---------Start to evaluate with Influence of the Test Dataset----")
                for i in range(len(visualization)):
                    value = compute_vis(sess, model, explain_x, visualization[i], grad_type='Smooth', m=50, p=0.2)
                    #value = sess.run(visualization[i], feed_dict={model.input: explain_x})
                    #value = np.asarray(value)
                    visualization_result.append(value[None,])
                visualization_result = np.vstack(visualization_result)
                if 'positive' in args.visulization_index:
                    path = args.visualization_path + 'positive_' + args.visulization_direction + '_visulization_result_' + args.layer + '.npy'
                elif 'negative' in args.visulization_index:
                    path = args.visualization_path +'negative_' + args.visulization_direction + '_visulization_result_' + args.layer + '.npy'
                else:
                    path = args.visualization_path + args.visulization_direction + '_visulization_result_' + args.layer + '.npy'
                np.save(path, visualization_result)
                print("---> The visulization is saved to "+path)


            else:
                args.layer += 'guided'
                g = tf.Graph()
                with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                    if ('fc' in args.layer) or ('prob' in args.layer):
                        visualization = compute_grad(x=input_tensor, y=layer_output, ids=index, type='fc')
                    else:
                        visualization = compute_grad(x=input_tensor, y=layer_output, ids=index, type='conv')
                    #model = K.tf.modify_model_backprop(model, 'guided')
                    print("----------------Model is restored from Pneumonia.h5---")
                    print("---------Start to evaluate with Influence of the Test Dataset----")
                    for i in range(len(visualization)):
                        value = sess.run(visualization[i], feed_dict={model.input: explain_x})
                        value = np.asarray(value)
                        visualization_result.append(value)
                    visualization_result = np.vstack(visualization_result)
                    if 'positive' in args.visulization_index:
                        path = args.visualization_path + 'positive_' + args.visulization_direction + '_visulization_result_' + args.layer + '.npy'
                    elif 'negative' in args.visulization_index:
                        path = args.visualization_path +'negative_' + args.visulization_direction + '_visulization_result_' + args.layer + '.npy'
                    else:
                        path = args.visualization_path + args.visulization_direction + '_visulization_result_' + args.layer + '.npy'
                    np.save(path, visualization_result)
                    print("---> The visulization is saved to "+path) 
    else:
        print ("Please input the correcnt mode name!!")

if __name__ == '__main__':
    args = parser.parse_args()
    if create_save_dir(args):
        main(args)
