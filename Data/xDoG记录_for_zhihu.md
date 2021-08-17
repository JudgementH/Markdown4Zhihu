# 1 基本信息

## 1.1 论文基本信息



**论文题目**：XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization

**论文地址**：https://www.sciencedirect.com/science/article/abs/pii/S009784931200043X

**发布时间**：2012年10月



# 2 问题提出

**问题起因**：DoG(difference-of-gaussians)是一个生成边缘图片的算法。作者发现很多论文为了提取边缘不仅仅使用DoG算法，还额外添加了很多复杂的过程。作者认为这些是没有必要的，这些复杂的过程都可以等效为一种DoG。因此作者对DoG进行拓展，发表了XDoG（eXtended difference-of-gaussians）。说白就是作者改进了DoG。



**作者思路**：作者对DoG的改进比较关键的一点是为了加宽边缘，采用阈值进行范围判断而不是取梯度为0之类的点。并且额外添加参数控制。



# 3 现有边缘提取方法与问题



## 3.1 现有方法及其问题

**(a)Canny算法**：使用基于梯度的方法进行边缘提取。问题是对噪声敏感，并且生成边缘图的边缘线宽度往往是只有1~2个像素。这很不合理一般的边缘图片不应该是这么细的。



**(b)LoG（Laplacian of Gaussian）**：对高斯卷积核使用拉普拉斯算子后再对图片进行卷积。问题是产生的边缘性宽度还是1个像素左右，不合理，并且计算效率比较低。

LoG用公式描述就是

<img src="https://www.zhihu.com/equation?tex=LoG=\nabla^2(G*I)=(\nabla^2G)*I
" alt="LoG=\nabla^2(G*I)=(\nabla^2G)*I
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 是图片， <img src="https://www.zhihu.com/equation?tex=G" alt="G" class="ee_img tr_noresize" eeimg="1"> 是高斯卷积核。



## 3.2 DoG算法



### 介绍

对于LoG中计算速度慢的问题，使用DoG可以解决。DoG使用两个图片的高斯差分图来进行边缘检测。

![image-20210818001928598](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/xDoG记录/image-20210818001928598.jpg)

上图是使用不同的方法对原图片(a)处理的结果

第一行是使用梯度的方法，选取图形梯度为0（或二阶导数）的点组成的边缘图片，可以发现是边缘为线状并且受到噪声影响比较大。



### 算法流程

1. 对于图片 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> ，设定高斯卷积核的方差 <img src="https://www.zhihu.com/equation?tex=\sigma" alt="\sigma" class="ee_img tr_noresize" eeimg="1"> 和放缩比率 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 。
2. 对 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> ，使用高斯滤波器 <img src="https://www.zhihu.com/equation?tex=G_\sigma" alt="G_\sigma" class="ee_img tr_noresize" eeimg="1"> 进行卷积，得到 <img src="https://www.zhihu.com/equation?tex=g_1" alt="g_1" class="ee_img tr_noresize" eeimg="1"> 
3. 对 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> ，使用高斯滤波器 <img src="https://www.zhihu.com/equation?tex=G_{k\sigma}" alt="G_{k\sigma}" class="ee_img tr_noresize" eeimg="1"> 进行卷积，得到 <img src="https://www.zhihu.com/equation?tex=g_2" alt="g_2" class="ee_img tr_noresize" eeimg="1"> 
4.  <img src="https://www.zhihu.com/equation?tex=g_1-g_2" alt="g_1-g_2" class="ee_img tr_noresize" eeimg="1"> 得到高斯差分图
5. 取差分图中为0的地方得到边缘图片







**DoG相对于LoG加速原理**

假设图像为 <img src="https://www.zhihu.com/equation?tex=I(x)" alt="I(x)" class="ee_img tr_noresize" eeimg="1"> ，表示在坐标 <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 处，灰度值是 <img src="https://www.zhihu.com/equation?tex=I(x)" alt="I(x)" class="ee_img tr_noresize" eeimg="1"> 。使用两个不同 <img src="https://www.zhihu.com/equation?tex=\sigma" alt="\sigma" class="ee_img tr_noresize" eeimg="1"> 的高斯卷积核，对图片 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 进行卷积操作。有

<img src="https://www.zhihu.com/equation?tex=g_1(x)=G(x,\sigma_1)*I(x)\\
g_2(x)=G(x,\sigma_2)*I(x)
" alt="g_1(x)=G(x,\sigma_1)*I(x)\\
g_2(x)=G(x,\sigma_2)*I(x)
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=*" alt="*" class="ee_img tr_noresize" eeimg="1"> 代表卷积。

那么DoG就是两个不同高斯核卷积卷积结果的差

<img src="https://www.zhihu.com/equation?tex=DoG=g_1 - g_2 = G(x,\sigma_1) * I(x) - G(x,\sigma_2) * I(x)
" alt="DoG=g_1 - g_2 = G(x,\sigma_1) * I(x) - G(x,\sigma_2) * I(x)
" class="ee_img tr_noresize" eeimg="1">
而对于导数定义有

<img src="https://www.zhihu.com/equation?tex=\frac{\part G}{\part \sigma} = \lim_{\Delta\sigma \rightarrow0 }\frac{G(x,\sigma +\Delta\sigma)-G(x,\sigma)}{\Delta\sigma}
" alt="\frac{\part G}{\part \sigma} = \lim_{\Delta\sigma \rightarrow0 }\frac{G(x,\sigma +\Delta\sigma)-G(x,\sigma)}{\Delta\sigma}
" class="ee_img tr_noresize" eeimg="1">
若设 <img src="https://www.zhihu.com/equation?tex=k\sigma = \sigma + \Delta \sigma" alt="k\sigma = \sigma + \Delta \sigma" class="ee_img tr_noresize" eeimg="1"> ，当 <img src="https://www.zhihu.com/equation?tex=k\rightarrow 1" alt="k\rightarrow 1" class="ee_img tr_noresize" eeimg="1"> 时，近似有

<img src="https://www.zhihu.com/equation?tex=\frac{\part G}{\part \sigma} \approx \frac{G(x,k\sigma)-G(x,\sigma)}{(k-1)\sigma}
" alt="\frac{\part G}{\part \sigma} \approx \frac{G(x,k\sigma)-G(x,\sigma)}{(k-1)\sigma}
" class="ee_img tr_noresize" eeimg="1">
代入DoG的公式，也就是令 <img src="https://www.zhihu.com/equation?tex=\sigma_1=k\sigma,\sigma_2 = \sigma" alt="\sigma_1=k\sigma,\sigma_2 = \sigma" class="ee_img tr_noresize" eeimg="1"> 

<img src="https://www.zhihu.com/equation?tex=DoG = G(x,\sigma_1)-G(x,\sigma_2)\approx (k-1)\sigma_2 \frac{\part G}{\part \sigma_2}
" alt="DoG = G(x,\sigma_1)-G(x,\sigma_2)\approx (k-1)\sigma_2 \frac{\part G}{\part \sigma_2}
" class="ee_img tr_noresize" eeimg="1">
而对于高斯函数对 <img src="https://www.zhihu.com/equation?tex=\sigma" alt="\sigma" class="ee_img tr_noresize" eeimg="1"> 求偏导有

<img src="https://www.zhihu.com/equation?tex=\frac{\part G}{\part \sigma}=\frac{-2\sigma^2+x^2}{2\pi\sigma^5}exp(-\frac{x^2}{2\sigma^2})=\sigma \nabla^2G
" alt="\frac{\part G}{\part \sigma}=\frac{-2\sigma^2+x^2}{2\pi\sigma^5}exp(-\frac{x^2}{2\sigma^2})=\sigma \nabla^2G
" class="ee_img tr_noresize" eeimg="1">
所以

<img src="https://www.zhihu.com/equation?tex=DoG\approx(k-1)\sigma^2\nabla^2G
" alt="DoG\approx(k-1)\sigma^2\nabla^2G
" class="ee_img tr_noresize" eeimg="1">
与LoG很相似，所以用DoG来近似代替LoG，进而加速计算。







# 4 XDoG算法



### 改进之处

xDoG对DoG中差分图和阈值添加新的参数来控制。

![image-20210818001928598](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/xDoG记录/image-20210818001928598.jpg)

如上图，为了修正边缘处只有一两个像素的问题，使用了阈值的方法去做边缘判定

<img src="https://www.zhihu.com/equation?tex=T_{\epsilon}(u) =
\left\{\begin{array}{l} 
  1 &u\ge\epsilon\\  
  0 & u<\epsilon
\end{array}\right.
" alt="T_{\epsilon}(u) =
\left\{\begin{array}{l} 
  1 &u\ge\epsilon\\  
  0 & u<\epsilon
\end{array}\right.
" class="ee_img tr_noresize" eeimg="1">
第二行的(e)是应用阈值处理后的图片，可以发现效果明显比(d)好多了。同时为了更强的适应性，可以把阈值函数进行拓展，如下。



使用改进的阈值函数

<img src="https://www.zhihu.com/equation?tex=T_{\epsilon,\varphi}(u) =
\left\{\begin{array}{l} 
  1 &u\ge\epsilon\\  
  1 + tanh(\varphi \cdot (u - \epsilon)) & u<\epsilon
\end{array}\right.
" alt="T_{\epsilon,\varphi}(u) =
\left\{\begin{array}{l} 
  1 &u\ge\epsilon\\  
  1 + tanh(\varphi \cdot (u - \epsilon)) & u<\epsilon
\end{array}\right.
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\epsilon，\varphi" alt="\epsilon，\varphi" class="ee_img tr_noresize" eeimg="1"> 是参数， <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> 是高斯差分图中每个像素点的灰度值，1是白色，0是黑色。这样当 <img src="https://www.zhihu.com/equation?tex=\varphi\rightarrow \infin" alt="\varphi\rightarrow \infin" class="ee_img tr_noresize" eeimg="1"> 时，也可以近似看作阶梯的二元函数。





扩展高斯差分，在高斯差分图中新加参数 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 来控制高斯差分图

<img src="https://www.zhihu.com/equation?tex=D_{\sigma,k,\gamma}=G_{\sigma}(x)-\gamma G_{k\sigma}(x)
" alt="D_{\sigma,k,\gamma}=G_{\sigma}(x)-\gamma G_{k\sigma}(x)
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\sigma,k,\gamma" alt="\sigma,k,\gamma" class="ee_img tr_noresize" eeimg="1"> 是控制的参数



论文中还写为了方便参数控制，高斯差分还可以写为

<img src="https://www.zhihu.com/equation?tex=S_{\sigma,k,p}=\frac{D_{\sigma,k,\gamma}(x)}{\gamma-1}=(1+p)G_{\sigma}(x)-p G_{k\sigma}(x)
" alt="S_{\sigma,k,p}=\frac{D_{\sigma,k,\gamma}(x)}{\gamma-1}=(1+p)G_{\sigma}(x)-p G_{k\sigma}(x)
" class="ee_img tr_noresize" eeimg="1">
总结：在xDoG中参数共有 <img src="https://www.zhihu.com/equation?tex=\sigma,k,\gamma,\epsilon,\varphi" alt="\sigma,k,\gamma,\epsilon,\varphi" class="ee_img tr_noresize" eeimg="1"> ，根据不同的差分形式，也有可能没有 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 而是有 <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1"> 



参数说明

+  <img src="https://www.zhihu.com/equation?tex=\epsilon" alt="\epsilon" class="ee_img tr_noresize" eeimg="1"> 一般选取图片中间灰度值
+ p，接近20的值效果不错
+  <img src="https://www.zhihu.com/equation?tex=\varphi" alt="\varphi" class="ee_img tr_noresize" eeimg="1"> ，较小时对细节比较敏感。
+  <img src="https://www.zhihu.com/equation?tex=\sigma" alt="\sigma" class="ee_img tr_noresize" eeimg="1"> ，较大时会忽略细节，边缘变粗



### 算法流程

1. 对于图片 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> ，设定高斯卷积核的方差 <img src="https://www.zhihu.com/equation?tex=\sigma" alt="\sigma" class="ee_img tr_noresize" eeimg="1"> 和放缩比率 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 。
2. 对 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> ，使用高斯滤波器 <img src="https://www.zhihu.com/equation?tex=G_\sigma" alt="G_\sigma" class="ee_img tr_noresize" eeimg="1"> 进行卷积，得到 <img src="https://www.zhihu.com/equation?tex=g_1" alt="g_1" class="ee_img tr_noresize" eeimg="1"> 
3. 对 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> ，使用高斯滤波器 <img src="https://www.zhihu.com/equation?tex=G_{k\sigma}" alt="G_{k\sigma}" class="ee_img tr_noresize" eeimg="1"> 进行卷积，得到 <img src="https://www.zhihu.com/equation?tex=g_2" alt="g_2" class="ee_img tr_noresize" eeimg="1"> 
4.  <img src="https://www.zhihu.com/equation?tex=g_1-\gamma g_2" alt="g_1-\gamma g_2" class="ee_img tr_noresize" eeimg="1"> 得到高斯差分图
5. 根据阈值函数得到边缘图片



### 实验

在DoG算法中，有参数 <img src="https://www.zhihu.com/equation?tex=\sigma,k,\gamma,\epsilon" alt="\sigma,k,\gamma,\epsilon" class="ee_img tr_noresize" eeimg="1"> 下面对这几个参数分别进行测试



**1）对比σ对xDoG的影响**

![image-20210818010229786](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/xDoG记录/image-20210818010229786.png)

其余参数 <img src="https://www.zhihu.com/equation?tex=k=1.6，\gamma=0.98，\epsilon=0.1,\varphi=100" alt="k=1.6，\gamma=0.98，\epsilon=0.1,\varphi=100" class="ee_img tr_noresize" eeimg="1"> 。

可以发现σ越大得到的边缘越粗，结果越抽象



**2)对比k对xDoG的影响**

![image-20210818010242707](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/xDoG记录/image-20210818010242707.png)

其余 <img src="https://www.zhihu.com/equation?tex=\sigma=3，\gamma=0.98，\epsilon=0.1, \varphi=100" alt="\sigma=3，\gamma=0.98，\epsilon=0.1, \varphi=100" class="ee_img tr_noresize" eeimg="1"> 。

当 <img src="https://www.zhihu.com/equation?tex=k\rightarrow0" alt="k\rightarrow0" class="ee_img tr_noresize" eeimg="1"> 时，得到图片会尽可能保留细节，并且边缘较细。

当 <img src="https://www.zhihu.com/equation?tex=k\rightarrow 1" alt="k\rightarrow 1" class="ee_img tr_noresize" eeimg="1"> 时，两张高斯图片比较接近，对于一些不太明显的边缘就被忽略了。

当 <img src="https://www.zhihu.com/equation?tex=k\rightarrow \infin" alt="k\rightarrow \infin" class="ee_img tr_noresize" eeimg="1"> 时，图片会忽略细节，并且边缘交粗。

作者在文中说 <img src="https://www.zhihu.com/equation?tex=k=1.6" alt="k=1.6" class="ee_img tr_noresize" eeimg="1"> 时通常效果比较好



**3)对比γ对xDoG的影响**

![image-20210818014324719](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/xDoG记录/image-20210818014324719.png)

其余 <img src="https://www.zhihu.com/equation?tex=\sigma=3，k=1.6，\epsilon=0.1，\varphi=100" alt="\sigma=3，k=1.6，\epsilon=0.1，\varphi=100" class="ee_img tr_noresize" eeimg="1"> 

发现 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 这个参数比较敏感，不能在个位数级别进行调整。

当 <img src="https://www.zhihu.com/equation?tex=\gamma<1" alt="\gamma<1" class="ee_img tr_noresize" eeimg="1"> 时，边缘为黑色。

当 <img src="https://www.zhihu.com/equation?tex=\gamma>1" alt="\gamma>1" class="ee_img tr_noresize" eeimg="1"> 时，边缘为白色。



![image-20210818014729751](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/xDoG记录/image-20210818014729751.png)

又可以发现当 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 越接近1，边缘越粗，同时也会更受到噪声影响。







**4)对比ε对xDoG的影响**

论文中描述 <img src="https://www.zhihu.com/equation?tex=\epsilon" alt="\epsilon" class="ee_img tr_noresize" eeimg="1"> 一般是取中间值。由于阈值判断直接和 <img src="https://www.zhihu.com/equation?tex=\epsilon" alt="\epsilon" class="ee_img tr_noresize" eeimg="1"> 相关联， <img src="https://www.zhihu.com/equation?tex=\epsilon" alt="\epsilon" class="ee_img tr_noresize" eeimg="1"> 是在高斯差分图中的阈值。

![image-20210818015532030](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/xDoG记录/image-20210818015532030.png)

其余 <img src="https://www.zhihu.com/equation?tex=\sigma=3，k=1.6，\gamma=0.98，\varphi=100" alt="\sigma=3，k=1.6，\gamma=0.98，\varphi=100" class="ee_img tr_noresize" eeimg="1"> 

同时本张图片的高斯差分图的统计数据为

| min    | max   | mean |

| ------ | ----- | ---- |

| -41.18 | 43.60 | 2.05 |


这个主要影响到的是取内外边缘，和对噪声的过滤能力。

 <img src="https://www.zhihu.com/equation?tex=\epsilon" alt="\epsilon" class="ee_img tr_noresize" eeimg="1"> 越大，也就是通过阈值的条件越苛刻，图中黑色越多。并且 <img src="https://www.zhihu.com/equation?tex=\epsilon" alt="\epsilon" class="ee_img tr_noresize" eeimg="1"> 越接近最大值或最小值，边缘细节越少。



**5)对比φ对xDoG的影响**

![image-20210818021452662](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/xDoG记录/image-20210818021452662.png)

其余 <img src="https://www.zhihu.com/equation?tex=\sigma=3，k=1.6，\gamma=0.98，\epsilon=0.1" alt="\sigma=3，k=1.6，\gamma=0.98，\epsilon=0.1" class="ee_img tr_noresize" eeimg="1"> 

当 <img src="https://www.zhihu.com/equation?tex=\varphi" alt="\varphi" class="ee_img tr_noresize" eeimg="1"> 较小时，有渐变的过程，当 <img src="https://www.zhihu.com/equation?tex=\varphi" alt="\varphi" class="ee_img tr_noresize" eeimg="1"> 较大时，基本为二值图片。





边缘提取，对于人物来说要尽量避免受到阴影的影响，并且要忽视背景。经过我的尝试当参数为 <img src="https://www.zhihu.com/equation?tex=\sigma=1,k=1.6,\gamma=0.98,\epsilon=-1.3,\varphi=5" alt="\sigma=1,k=1.6,\gamma=0.98,\epsilon=-1.3,\varphi=5" class="ee_img tr_noresize" eeimg="1"> 时效果比较不错

![image-20210818022505847](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/xDoG记录/image-20210818022505847.png)

