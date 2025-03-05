# 实用函数

- dir()
- help()

~~~python
import torch
dir(torch)
dir(torch.cuda)
#is_available没有括号！
help(torch.cuda.is_available)
~~~

# tensorboard

~~~python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
#文件名称可自定义
writer = SummaryWriter("logs")
#一系列writer操作
#添加图片
image_path ="F:\\DeepLearning\\Dataset\\train\\ants_image\\0013035.jpg"
img_PIL = Image.open(image_path)
#将图片转为需要的格式
img_array = np.array(img_PIL)
writer.add_image("test",img_array,1,dataformats='HWC')

#添加函数
for i in range(100):
    writer.add_scalar("y = 2x",2*i,i)

writer.close()
~~~

最后在终端实用命令行`tensorboard --logdir=logs`即可查看