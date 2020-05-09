import sys

sys.path.append('../')
import d2lzh as d2l
import mxnet
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn, rnn, loss as gloss
import gluoncv
from gluoncv import model_zoo
from gluoncv.data.transforms import video

ctx = mxnet.gpu(0)
input_size = 224
scale_ratios = [1.0, 0.875, 0.75, 0.66]
batch_size = 8
num_epochs = 200
num_classes = 101
num_depth = 8
num_hiddens = 256

train_list = '../data/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'
val_list = '../data/ucf101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt'
data_dir = '../data/ucf101/rawframes'

transform_train = video.VideoGroupTrainTransform(size=(input_size, input_size), scale_ratios=scale_ratios,
                                                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = gluoncv.data.UCF101(setting=train_list, root=data_dir, train=True,
                                    new_width=340, new_height=256, new_length=num_depth,
                                    target_width=224, target_height=224,
                                    num_segments=1, transform=transform_train)
val_dataset = gluoncv.data.UCF101(setting=val_list, root=data_dir, train=False,
                                  new_width=340, new_height=256, new_length=num_depth,
                                  target_width=224, target_height=224,
                                  num_segments=1, transform=transform_test)
num_workers = 0 if sys.platform.startswith('win32') else 4
train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   prefetch=num_workers * 2, last_batch='rollover')
val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 prefetch=num_workers * 2, last_batch='discard')


# nd.split()
def get_action_recognition_model(ctx):
    class LRCN_action_recognition_model(nn.HybridBlock):
        def __init__(self, ctx, **kwargs):
            super(LRCN_action_recognition_model, self).__init__(**kwargs)
            # self.feature_extractor = gluon.model_zoo.vision.get_model(name='alexnet', pretrained=True, ctx=ctx).features[
            #                          :10]
            self.feature_extractor = gluon.model_zoo.vision.get_model(name='vgg16', pretrained=True, ctx=ctx).features[
                                     :32]
            self.rnn_layer = rnn.LSTM(hidden_size=num_hiddens, prefix='LRCN_LSTM_')
            self.rnn_layer.initialize(ctx=ctx)
            self.begin_state()
            self.dense = nn.Dense(num_classes, prefix='LRCN_Dense_')
            self.dense.initialize(ctx=ctx)

        def begin_state(self, *args, **kwargs):
            self.init_state = self.rnn_layer.begin_state(batch_size=batch_size, ctx=ctx, *args, **kwargs)

        def hybrid_forward(self, F, X, *args, **kwargs):
            X = X.reshape((-1,) + X.shape[2:])  # X是五维NCDHW
            inputs = X.split(axis=2, num_outputs=X.shape[2], squeeze_axis=1)  # input变为D个NCHW组成的list
            xs = []
            state = self.init_state  #
            for x in inputs:
                x = F.flatten(self.feature_extractor(x))  # x变为NF的二维
                x = x.expand_dims(axis=0)  # x新增了一维，
                xs.append(x)
            X_ = nd.concat(*xs, dim=0)  # X_变为(T,N,F)
            output, state = self.rnn_layer(X_, state)  # 输出也成为TNF
            output = output.sum(axis=0, keepdims=False) / output.shape[0]  # 在时间步求平均，转换为NF
            Y = self.dense(output)  # Y输出(N,num_classes)
            return Y

    return LRCN_action_recognition_model(ctx=ctx)


net = get_action_recognition_model(ctx=ctx)
print(net)
loss = gloss.SoftmaxCrossEntropyLoss()

optimizer_params = {'learning_rate': 0.1, 'momentum': 0.9, 'wd': 5e-4}
trainer = gluon.Trainer(net.collect_params('LRCN'), 'sgd', optimizer_params)

d2l.train(train_iter=train_data, test_iter=val_data, net=net, loss=loss, trainer=trainer, ctx=ctx,
          num_epochs=num_epochs)
net.save_parameters(filename='./LRCN_AR_vgg16.params')
