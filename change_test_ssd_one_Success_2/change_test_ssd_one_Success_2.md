

```python
from mxnet import gluon
import mxnet as mx_net
import os
import numpy as np 
from mxnet import image
from mxnet import nd
from mxnet import init
from mxnet import cpu
import matplotlib as _plotlib
import matplotlib.pyplot as _pyplot
from mxnet.contrib.ndarray import MultiBoxPrior
from mxnet.gluon import nn
from mxnet.contrib.ndarray import MultiBoxDetection
ctx = mx_net.cpu()
#mean_rgb_value = nd.array([123, 117, 104])
mean_rgb_value = nd.array([123-55, 117-50, 104+60])
NumOfClass=1
NamesOfClass = ['pikachu'] #类名称
shape_dataset = 256

%matplotlib inline
_plotlib.rcParams['figure.dpi']= 120

def RectBox(Box_Rectangle, color, linewidth=3):
    #转换锚框成为rectangle
    Box_Rectangle = Box_Rectangle.asnumpy()
    return _pyplot.Rectangle(
        (Box_Rectangle[0], Box_Rectangle[1]), Box_Rectangle[2]-Box_Rectangle[0], Box_Rectangle[3]-Box_Rectangle[1],
        fill=False, edgecolor=color, linewidth=linewidth)

#预测物体的类别
def ClassPredictor(Anchors_Numbers, Classes_Numbers):
    #返回一个预测层
    return nn.Conv2D(Anchors_Numbers * (Classes_Numbers + 1), 3, padding=1)

#预测物体的边框
def YucheBox(Anchors_Numbers):
    #返回一个预测边框位置的网络
    return nn.Conv2D(Anchors_Numbers * 4, 3, padding=1)

#减半模块
def Reduce_Module(out_lays):
    #连接两个Conv-BatchNorm-Relu blocks和一个 pooling layer使得最后输出的特征减半
    outputs = nn.HybridSequential()
    for _ in range(2):
        outputs.add(nn.Conv2D(out_lays, 3, strides=1, padding=1))#输出　num_filters　个通道数
        outputs.add(nn.BatchNorm(in_channels=out_lays))#归一化
        outputs.add(nn.Activation('relu'))
    outputs.add(nn.MaxPool2D(2)) 
    return outputs

#将不同层的输出合并
def Fla_yuche(pred):
    return pred.transpose(axes=(0,2,3,1)).flatten()

def link_yuche(preds):
    return nd.concat(*preds, dim=1)

#主体网络
def main_body_net():
    outputs = nn.HybridSequential()
    for range_prediction in [16, 32, 64]:
        outputs.add(Reduce_Module(range_prediction))
    return outputs

#定义ssd模型
def SSD_Model(Anchors_Numbers, Classes_Numbers):
    ReduceSamplers = nn.Sequential()
    for _ in range(3):
        ReduceSamplers.add(Reduce_Module(128))
        
    ClassPred = nn.Sequential()
    Box_Pred = nn.Sequential()    
    for _ in range(5):
        ClassPred.add(ClassPredictor(Anchors_Numbers, Classes_Numbers))
        Box_Pred.add(YucheBox(Anchors_Numbers))

    All_Models = nn.Sequential()
    All_Models.add(main_body_net(), ReduceSamplers, ClassPred, Box_Pred)
    return All_Models

#计算预测
def SsdModelForward(x, All_Models, sizes, ratios, verbose=False):    
    main_body_net, ReduceSamplers, ClassPred, Box_Pred = All_Models
    output_anchors, output_class_preds, output_box_preds = [], [], []
    # feature extraction    
    x = main_body_net(x)#feature extraction完毕
    for i in range(5):
        # predict
        output_anchors.append(MultiBoxPrior(
            x, sizes=sizes[i], ratios=ratios[i]))
        output_class_preds.append(
            Fla_yuche(ClassPred[i](x)))
        output_box_preds.append(
            Fla_yuche(Box_Pred[i](x)))
        if verbose:
            print('Predict scale', i, x.shape, 'with', 
                  output_anchors[-1].shape[1], 'output_anchors')
        # down sample
        if i < 3:
            x = ReduceSamplers[i](x)
        elif i == 3:
            x = nd.Pooling(
                x, global_pool=True, pool_type='max', 
                kernel=(x.shape[2], x.shape[3]))
    # concat data
    return (link_yuche(output_anchors),
            link_yuche(output_class_preds),
            link_yuche(output_box_preds))

#完整的模型
class ToySSD(gluon.Block):
    def __init__(self, Classes_Numbers, verbose=False, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # anchor Box_Rectangle sizes and ratios for 5 feature scales
        self.sizes = [[.2,.272], [.37,.447], [.54,.619], 
                      [.71,.79], [.88,.961]]
        self.ratios = [[1,2,.5]]*5
        self.Classes_Numbers = Classes_Numbers
        self.verbose = verbose
        Anchors_Numbers = len(self.sizes[0]) + len(self.ratios[0]) - 1
        # use name_scope to guard the names
        with self.name_scope():
            self.All_Models = SSD_Model(Anchors_Numbers, Classes_Numbers)

    def forward(self, x):
        output_anchors, output_class_preds, output_box_preds = SsdModelForward(
            x, self.All_Models, self.sizes, self.ratios, 
            verbose=self.verbose)
        # it is better to have class predictions reshaped for softmax computation       
        output_class_preds = output_class_preds.reshape(shape=(0, -1, self.Classes_Numbers+1))
        return output_anchors, output_class_preds, output_box_preds
    
#预测初始化
os.makedirs('checkpoints',exist_ok=True)
filename = "checkpoints/testnet.params"
filename_2 = "checkpoints_2/ssd_net.params"
filename_3 = "checkpoints_3/ssd_net_3.params"
ctx = cpu(0)
#TrainData.reshape(label_shape=(3, 5))
#TrainData = TestData.sync_label_shape(TrainData)
net = ToySSD(NumOfClass)
net.load_params(filename_3, ctx=ctx)

#图像预处理
def img_Processor(file_name):
    with open(file_name, 'rb') as f:
        img = image.imdecode(f.read())
    # resize to shape_dataset
    data = image.imresize(img, shape_dataset, shape_dataset)
    # minus rgb mean
    data = data.astype('float32') - mean_rgb_value
    # convert to batch_test x channel x height xwidth
    return data.transpose((2,0,1)).expand_dims(axis=0), img

#定义预测函数
def predict(x):
    output_anchors, output_class_preds, output_box_preds = net(x.as_in_context(ctx))
    output_class_probs = nd.SoftmaxActivation(
        output_class_preds.transpose((0,2,1)), mode='channel')

    return MultiBoxDetection(output_class_probs, output_box_preds, output_anchors,force_suppress=True, clip=False)

#预测
path='../img/pikachu27.png'#threshold=0.51 pikachu6_2 pikachu16.png
path_2='../img/pikachu15.jpg'#threshold=0.45
path_3='../img/pikachu6_2.jpg'
x, img = img_Processor(path)
outputs = predict(x)
outputs.shape
print(outputs[0][0:25])

#显示输出
five_colors = ['blue', 'green', 'red', 'black', 'magenta']
_plotlib.rcParams['figure.figsize'] = (6,6)

def display_preds(img, outputs, threshold=0.5):    
    _pyplot.imshow(img.asnumpy())
    for rows in outputs:
        rows = rows.asnumpy()
        class_num_id, class_score = int(rows[0]), rows[1]
        if class_num_id < 0 or class_score < threshold:
            continue
        color = five_colors[class_num_id%len(five_colors)]#例如０％５＝０　１％５＝１　２％５＝２
        Box_Rectangle = rows[2:6] * np.array([img.shape[0],img.shape[1]]*2)
        rect = RectBox(nd.array(Box_Rectangle), color, 2)
        _pyplot.gca().add_patch(rect)
                        
        text = NamesOfClass[class_num_id]
        _pyplot.gca().text(Box_Rectangle[0], Box_Rectangle[1], 
                       '{:s} {:.2f}'.format(text, class_score),
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=10, color='white')
    _pyplot.show()

display_preds(img, outputs[0], threshold=0.72)

```

    
    [[ 0.          0.7258899   0.393848    0.51717395  0.571969    0.782157  ]
     [ 0.          0.7091782   0.3140946   0.4779768   0.6413722   0.81418645]
     [-1.          0.69349     0.39691657  0.5070703   0.57950413  0.7703539 ]
     [-1.          0.69337803  0.4266235   0.51767224  0.5820055   0.78598326]
     [-1.          0.6877676   0.36408052  0.50356776  0.5481125   0.7381697 ]
     [-1.          0.6754673   0.36862034  0.48785603  0.5577538   0.7844056 ]
     [-1.          0.6616515   0.32024372  0.47668248  0.5336816   0.75571054]
     [-1.          0.6407366   0.3891186   0.50150764  0.5632318   0.749519  ]
     [ 0.          0.6181189   0.60843873  0.7692001   0.8322693   1.2501876 ]
     [ 0.          0.61488074  0.34394407  0.4897967   0.5129043   0.7432627 ]
     [-1.          0.6056179   0.40233958  0.52348834  0.5770212   0.7863439 ]
     [ 0.          0.6037934   0.05302654  0.24143937  0.23644857  0.4928793 ]
     [-1.          0.60245305  0.3681883   0.50649595  0.5429158   0.7579088 ]
     [-1.          0.59721905  0.34790668  0.47824556  0.5741546   0.7665432 ]
     [-1.          0.59419423  0.38886622  0.43911642  0.5918128   0.75903255]
     [-1.          0.5909146   0.6483889   0.7578392   0.8963967   1.2466786 ]
     [ 0.          0.59037596  0.12913632  0.32400328  0.2806784   0.5710224 ]
     [ 0.          0.58786243 -0.02704573  0.2832108   0.13189736  0.49098945]
     [-1.          0.5865645   0.399627    0.5220457   0.5641701   0.75806826]
     [ 0.          0.5848427  -0.01195574  0.20531486  0.15569104  0.43504435]
     [-1.          0.583555    0.12551057  0.3025601   0.28098002  0.518989  ]
     [ 0.          0.58143604 -0.01469938  0.61406595  0.33237153  1.101212  ]
     [-1.          0.572242    0.3362449   0.50173724  0.5302497   0.72365177]
     [-1.          0.5696346  -0.02219776  0.23880126  0.14063737  0.45181012]
     [ 0.          0.5671981   0.47572172  0.7487014   0.71765995  1.2520254 ]]
    <NDArray 25x6 @cpu(0)>



![png](output_0_1.png)

