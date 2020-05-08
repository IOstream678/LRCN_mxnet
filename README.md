## LRCN_mxnet
我是研一小硕在读，如果理解和阐述有偏差，欢迎您批评指正！！  

这是
Long-term Recurrent Convolutional Networks for Visual Recognition and Description_2015_CVPR_paper 
在mxnet/Gluon上的复现。  
Gluon是mxnet的高级接口  
本代码运行需要用到d2lzh工具包，下载后和这个项目文件夹放置在同一个目录下。d2lzh包来源于李沐、阿斯顿张的《动手学深度学习》：https://zh.d2l.ai/，
（但我对于此工具包进行了~~魔改~~部分修改，源码还没传上来，之后补上）  

论文分为三个部分，计划依次实现：  
- [x]Activity(action) recognition
- []Image description
- []Video description  
###Action Recognition
论文使用ALEXNet+单层LSTM实现，将视频数据集的depth看作时间步，
每个时间步都将单帧图像输入CNN提取特征，
后接全连接层，输入LSTM，将各时间步LSTM的输出取平均。  

#### 数据集准备

论文中使用UCF101数据集，UCF101数据集的使用方法参考：https://gluon-cv.mxnet.io/build/examples_datasets/ucf101.html ，写的很明了。  
gluoncv是mxnet官方推出的a Deep Learning Toolkit for Computer Vision。  
目前代码存在严重过拟合，正在寻求改进。