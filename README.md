# XAIIC-classify-the-characteristics-of-Xi-an

> Xi'an was called Chang'an in the ancient times, and it is one of the birthplaces of the Chinese nation. The them of this competition is Xi'an tourism, using artificial intelligence technology to classify Xi'an popular attractions, food, special products, folk customs, crafts and other pictures.

![img_0](https://github.com/WWWangHan/XAIIC-classify-the-characteristics-of-Xi-an/blob/master/readme_img/img_0.jpg)

![img_1](https://github.com/WWWangHan/XAIIC-classify-the-characteristics-of-Xi-an/blob/master/readme_img/img_1.jpg)

![img_2](https://github.com/WWWangHan/XAIIC-classify-the-characteristics-of-Xi-an/blob/master/readme_img/img_2.jpg)

### training

> Running environment `ubuntu 16.04` with `pytorch 1.3.1` and `torchvision 0.4.2`.

> To start the training procedure, just run `python go.py`. Then you could see something like this

![start_0](https://github.com/WWWangHan/XAIIC-classify-the-characteristics-of-Xi-an/blob/master/readme_img/start_0.png)

![start_1](https://github.com/WWWangHan/XAIIC-classify-the-characteristics-of-Xi-an/blob/master/readme_img/start_1.png)

> After finishing training procedure, there will be some .pth files:

![all_pth](https://github.com/WWWangHan/XAIIC-classify-the-characteristics-of-Xi-an/blob/master/readme_img/all_pth.png)

### mention

> This is only the `preliminary version`.

> We download some pictures from internet using script to enlarge the train&val set, but there are still some problems(you do not need to take much care about this):

![m_1](https://github.com/WWWangHan/XAIIC-classify-the-characteristics-of-Xi-an/blob/master/readme_img/m_1.png)

> Another thing you should pay attention to is the plateform provided by HUAWEI which is called `ModelArts`, this version of our code can not be directly used on this plateform. We made some modificaions with the baseline using our developing environment and got a great improvement(but not the best, we do not pay much more time on finetuning). Meanwhile, we found that without our modification(just using more pictures), we can also get an excellent result.
