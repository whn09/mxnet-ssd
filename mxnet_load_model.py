# load model and predicate
import mxnet as mx
import numpy as np

# define test data
batch_size = 1
num_batch = 1
filepath = 'frame-1.jpg'
DEFAULT_INPUT_SHAPE = 300


# load model
sym, arg_params, aux_params = mx.model.load_checkpoint("model/deploy_model_algo_1", 0) # load with net name and epoch num
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=["data"], label_names=["cls_prob"])
print('data_names:', mod.data_names)
print('output_names:', mod.output_names)
#print('data_shapes:', mod.data_shapes)
#print('label_shapes:', mod.label_shapes)
#print('output_shapes:', mod.output_shapes)

mod.bind(data_shapes=[("data", (1, 3, DEFAULT_INPUT_SHAPE, DEFAULT_INPUT_SHAPE))], for_training=False)
mod.set_params(arg_params, aux_params)  # , allow_missing=True

import cv2
img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
print(img.shape)
img = cv2.resize(img, (DEFAULT_INPUT_SHAPE, DEFAULT_INPUT_SHAPE))
img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)
img = img[np.newaxis, :]
print(img.shape)

# # predict
# eval_data = np.array([img])
# eval_label = np.zeros(len(eval_data)) # just need to be the same length, empty is ok
# eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)
# print('eval_iter.provide_data:', eval_iter.provide_data)
# print('eval_iter.provide_label:', eval_iter.provide_label)
# predict_stress = mod.predict(eval_iter, num_batch)
# print(predict_stress) # you can transfer to numpy array

# forward
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
mod.forward(Batch([mx.nd.array(img)]))
prob = mod.get_outputs()[0].asnumpy()
prob = np.squeeze(prob)
# Grab top result, convert to python list of lists and return
results = [prob[i].tolist() for i in range(4)]
print(results)
