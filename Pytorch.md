# Pytorch

+ [Start Locally | PyTorch](https://pytorch.org/get-started/locally/) 安装
+ You will need a total of 200 GB (at minimum) of free disk space on the system that will
    be used for training.



## torch modules

+ torch 模块提供了建模的常用网络层和其他架构的组件
    如全连接层 （full-connected layer） 卷积层（convolutional layer） 激活函数（activate function）损失函数（loss function）

```python
## 找出 out 中数值最大的 index
_, index = torch.max(out, 1)

# In[15]:
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()
# Out[15]:
('golden retriever', 96.29334259033203)
```

+ for tensor & randn
    argument : names= , refine_names=

    ```python
    # In[8]:
    img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
    batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
    print("img named:", img_named.shape, img_named.names)
    print("batch named:", batch_named.shape, batch_named.names)
    # Out[8]:
    img named: torch.Size([3, 5, 5]) ('channels', 'rows', 'columns')
    batch named: torch.Size([2, 3, 5, 5]) (None, 'channels', 'rows', 'columns')
    ```

+ `torch.cuda.is_available()`

+ 

## torch.utils.data

1. ```python
   from torch.utils.data import DataLoader
   
###
   CLASStorch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
   
   Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
   ###
   ```
   
2. 

## torch.optim

optimizer 优化器 即SGD Adam 等一些方法
autogrid 是pytorch的一大亮点

## 并行训练

+ torch.nn.parallel.DistributedDataParallel
+ torch.distributed

# 一些相关的 python 知识

## 类

+ ```python
    super(type[, object-or-type])
    ```

    ```python
    class fooChildren(fooParent):
        def __init__(self, **kwarg):
            super（fooChildren，self）. function
    ```

    调用 fooChildren 的父类 fooParent 的函数方法

    例子

    ```python
    class FooParent(object):
        def __init__(self):
            self.parent = 'I\'m the parent.'
            print ('Parent')
        
        def bar(self,message):
            print ("%s from Parent" % message)
     
    class FooChild(FooParent):
        def __init__(self):
            # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类 FooChild 的对象转换为类 FooParent 的对象
            super(FooChild,self).__init__()    
            print ('Child')
            
        def bar(self,message):
            super(FooChild, self).bar(message)
            print ('Child bar fuction')
            print (self.parent)
     
    if __name__ == '__main__':
        fooChild = FooChild()
        fooChild.bar('HelloWorld')
    ```

    结果

    ```python
    Parent
    Child
    HelloWorld from Parent
    Child bar fuction
    I'm the parent.
    ```
    
+ len(对象.属性)。 如 len ( self.value )

     ！！ *获取长度*

    ```python
    def __len__(self):
        return len(self.value)
    ```

+ 

## 模块

+ ```python
    from PIL import Image
    img = Image.open("../data/p1ch2/bobby.jpg")
    
    img.show()
    ```




## 函数

+ enumerate()
  enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，==同时列出数据和数据下标==，一般用在 for 循环当中。

# 名词

1. building blocks == layers
2. The process of running a trained model on new data is called ==inference== in deep learning circles
3. fine-tuning 微调
4. 

# 神经网络模型 （ nn ）

## 生成对抗神经网络 （ Generative adversarial network GAN ）

+ 一个 generator 与一个 discriminator 竞争，将对方的 output 作为参考优化自身的参数，最终目的是使得 generator 生成筛选器分辨不出 data 是从生成器产生的还是真实数据（ 人为的 input ）。

# Summary

+ 2.7

    ```tex
    A pretrained network is a model that has already been trained on a dataset.
    Such networks can typically produce useful results immediately after loading
    the network parameters.
     By knowing how to use a pretrained model, we can integrate a neural network
    into a project without having to design or train it.
     AlexNet and ResNet are two deep convolutional networks that set new benchmarks
    for image recognition in the years they were released.
     Generative adversarial networks (GANs) have two parts—the generator and the
    discriminator—that work together to produce output indistinguishable from
    authentic items.
     CycleGAN uses an architecture that supports converting back and forth
    between two different classes of images.
     NeuralTalk2 uses a hybrid model architecture to consume an image and produce
    a text description of the image.
     Torch Hub is a standardized way to load models and weights from any project
    with an appropriate hubconf.py file.
    ```

    