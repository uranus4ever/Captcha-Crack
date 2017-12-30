# Captcha验证码破解


![HACK][img6]

本文使用卷积神经网络（CNN）破解CAPTCHA验证码，主要基于参考资料中的两种方案进行了优化和整合，在此致谢!

## CAPTCHA验证码
CAPTCHA验证码是目前互联网上常见的一种图灵测试，用来区别“机器”和人类。当然，有了AI的加持，机器也就能像人类一样聪明地识（破）别（解）了。

![login pic][img1]

但是上图中的验证码白底黑字，间距分明，显得过于简单了，为了增加挑战性，采取了以下这种验证码。而且，在Python中，可以通过加载`capthcha.image`工具库就能轻松地生成。

![captcha example][img2]

## 使用方法：
### 工具
Python 3
keras 2.0 (2.0以前版本会有编译错误)
OpenCV
### 步骤
1. 运行`captcha_generator.py`生成并保存验证码图片，可根据需要修改训练集的数量。
2. 运行`model.py`进行卷积神经网络训练，模型和权重保存为`captcha_model.hdf5`。
3. 运行`predict_with_model.py`，加载权重文件，进行预测。

## 破解方法详解
参考资料1中的破解方法简单粗暴，构建了一个很深的神经网络，然后把整张图片喂给模型训练，对应4个字符标签。这个模型比较复杂，而且如果用CPU来计算时间就很长了。我想到，如果能够先把图片预处理一下，识别出原图中4个字符的位置，然后单独训练，不就能大大简化训练模型和计算量了嘛。于是，进行了图片预处理和采用了两层卷积模型。

### 图片预处理

在把图像转换为灰度图像后，用OpenCV中的`findContours`函数能帮助我们轻松找到轮廓边界。但是，问题来了，原图中的噪声点（线）会干扰边界判断。用`cv2.medianBlur(img, filter_size)`就能很好地解决这一类似胡椒面问题，把符合filter_size的噪点过滤掉，然后就基本准确地识别出每个字符的边界了。

![pipeline][img3]

如果两个或多个字符黏连，那长宽比肯定会异常，就进行一下切割。

![split][img4]

### 卷积神经网络模型

两层的网络就足够啦。

![model structure][img5]

15代训练后，单个字符的准确率为97.7%，CPU计算时间，大概5分钟不到。

![acc][img6]

### 预测和评估
使用训练完的权重进行预测，把四个字符标签都一一对应算作识别成功，准确率大概在50%。考虑到可以让程序不断尝试提交，在网速能保证的情况下基本满足要求。

![predict][gif]

## 尾声
所谓魔高一尺道高一丈，现在新型验证码层出不穷，就是为了防止不（聪）法（明）分（机）子（智）破解的。
比如下面这种需要滑动鼠标拼图的，原理是检测鼠标运动轨迹。网上也已经有破解攻略了。

![slide code][img7]

但是，遇到12306这种丧心病狂的，也只能尴尬地微笑了o(*￣︶￣*)o

![12306][img8]

参考资料：
1. [使用深度学习来破解 captcha 验证码](https://zhuanlan.zhihu.com/p/26078299)
2. [仅需15分钟，使用OpenCV+Keras轻松破解验证码](https://mp.weixin.qq.com/s/fHuHICI-_xAXIOkV0abpPg)
3. [滑块验证码（滑动验证码）相比图形验证码，破解难度如何？](https://www.zhihu.com/question/32209043)

[//]: # (Image References)
[img1]: https://github.com/uranus4ever/Captcha-Crack/blob/master/img/login.png
[img2]: https://github.com/uranus4ever/Captcha-Crack/blob/master/img/GA9L.png
[img3]: https://github.com/uranus4ever/Captcha-Crack/blob/master/img/pipeline.png
[img4]: https://github.com/uranus4ever/Captcha-Crack/blob/master/img/split.png
[img5]: https://github.com/uranus4ever/Captcha-Crack/blob/master/img/model.png
[img6]: https://github.com/uranus4ever/Captcha-Crack/blob/master/img/HACK.png
[gif]: https://github.com/uranus4ever/Captcha-Crack/blob/master/img/predict.gif
[img7]: https://github.com/uranus4ever/Captcha-Crack/blob/master/img/slide.png
[img8]: https://github.com/uranus4ever/Captcha-Crack/blob/master/img/12306.png
