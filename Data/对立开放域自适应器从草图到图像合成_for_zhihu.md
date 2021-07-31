# 1 基本信息

## 1.1 论文基本信息



**论文标题**：Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis

**论文地址**：[[2104.05703\] Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis (arxiv.org)](https://arxiv.org/abs/2104.05703)

**发表时间**：2021年4月

**官方代码**：只有图片生成草图部分的代码[Mukosame/Anime2Sketch: A sketch extractor for anime/illustration. (github.com)](https://github.com/Mukosame/Anime2Sketch)，全部的代码作者说“The source code and pre-trained models will be released soon.”



# 2 针对问题：由草图生成图片



**问题起因**：作者在进行草图到图像生成工作时，发现主要有两大难点。1）缺少监督训练的数据集，也就苏大部分图像没有与之对应的草图。2）草图到图像域之间有着几何变形（geometry distortion）。针对第一个难点，一般的做法是使用假草图来替代真实草图进行训练。但假草图和真实草图之间也存在显著差距。本文就是为了解决这个差距而提出新的方法。



**作者思路**：作者提出了开放域的采样和优化策略来混淆生成器，让生成器把假草图当作真实草图

（这里介绍一下开放域 open-domain。在A上训练，在A上用是in-domain，在B上用是out-domain，在AB上用是open-domain）



# 3 现有方法的思路与问题



## 3.1 基于草图的图像合成



**（a）使用配对数据的监督训练方法**

**SketchyGAN**：使用草图和扩充的边缘图片，训练出GAN网络，来生成图片



**ContextualGAN**：把图片生成（image generation）问题转化图像补全（image completion）问题。网络学习到草图到图像对的联合分布，并且通过迭代的通过流形空间来获得结果。



**iSketchNFill**：使用轮廓信息（outlines）来代替草图，利用两阶段的方式，使用配对数据进行生成。



**EdgeGAN**：使用两个生成器来生成前景和背景。并且使用GAN网络来把边缘和相关图片编码为同一个隐空间中。



**（b）不使用配对数据的方法**

**Unsupervised sketchto-photo synthesis**：使用两阶段，类似cycleGAN的方法，可以结合参考图片风格进行单类别的草图到图像生成。



**（c）作者的目标**

目标是从一个不完整并且高度不均衡的数据集中，不使用配对数据，不用监督训练的方法，学习到一个多类别生成器。



## 3.2 使用边缘代替草图的策略

以上提及了很多方法是使用边缘（edge）来代替草图（sketch）进行数据增广，进行训练。针对这种方法，作者使用边缘提取器进行了测试。

![image-20210730174042212](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/对立开放域自适应器从草图到图像合成/image-20210730174042212.png)

作者使用SketchyCOCO数据集，使用Edge当作Sketch进行训练。



左侧两列，是模型使用边缘图片及其输出结果，可以看到产生的结果十分不错。

右侧两列，是模型使用真正的Sketch作为输入，可以看到生成效果不好，更像是给草图上纹理和色彩，而不是生成了一张真实照片级别的图片。



**结论**：使用边缘当作草图进行训练，最终效果不佳。

**原因**：Edge与Sketch之间存在巨大的domain gap，导致了在Edge上训练的模型在Sketch上的泛化性能不佳。



## 3.3 使用合成草图的策略

除去使用Edge代替Sketch的方法，还有使用合成的草图进行训练的方法。在数据集Scibble和QMUL-SKetch上预训练出从图片到草图的生成器。使用in-domain类别的数据集训练出图片到草图生成器。然后使用生成的草图作为真实草图，训练出草图到图像生成模型。

![image-20210730180555074](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/对立开放域自适应器从草图到图像合成/image-20210730180555074.png)

左侧两列是由合成的草图生成出真实图片，效果很不错。

右侧两列是由真实的草图生成出真实图片，发现生成器无法正常生成合理的真实图片。



**结论**：使用合成草图代替真实草图进行训练，效果不佳。

**原因**：虽然合成的草图和真实的草图，由人看来极其相似，但是对于生成模型来说，他们仍是不可辨别的。所以简单使用合成草图填补确实草图的位置仍然不能保存生成真实照片级别的图片。





# 4 解决方案基本思路



**主要思想**：对于生成草图策略中效果不理想问题，本文目标是学习到图片到草图和草图到图片的联合转化。借此减小合成草图和真实草图的domain gap





# 5 模型整体框架



## 5.1 模型图示

![image-20210730220121058](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/对立开放域自适应器从草图到图像合成/image-20210730220121058.png)

文中提出模型如上所示。



## 5.2 组件及特点

**主要组成**

+ 一个图片到草图的生成器 <img src="https://www.zhihu.com/equation?tex=G_A" alt="G_A" class="ee_img tr_noresize" eeimg="1"> （输入：图片；输出：草图）
+ 草图判别器 <img src="https://www.zhihu.com/equation?tex=D_A" alt="D_A" class="ee_img tr_noresize" eeimg="1"> （输入：草图；输出：草图的真假）
+ 一个多类别草图到图像生成器 <img src="https://www.zhihu.com/equation?tex=G_B" alt="G_B" class="ee_img tr_noresize" eeimg="1"> （输入：草图，标签；输出：图片）
+ 图片判别器 <img src="https://www.zhihu.com/equation?tex=D_B" alt="D_B" class="ee_img tr_noresize" eeimg="1"> （输入：图片；输出：图片的真假）
+ 分类器R（输入：图片，输出：图片的类别；保证输出受到标签的影响，产生正确的类别）



**模型特点**

+ 可以使用不配对草图与图像数据进行训练



## 5.3 模型过程



**流程输入**

+ 训练数据集 <img src="https://www.zhihu.com/equation?tex=D" alt="D" class="ee_img tr_noresize" eeimg="1"> 
+ 图片数据集 <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1"> 
+ 草图数据集 <img src="https://www.zhihu.com/equation?tex=s" alt="s" class="ee_img tr_noresize" eeimg="1"> 
+ 图片数据集对应的类别标签 <img src="https://www.zhihu.com/equation?tex=\eta_p" alt="\eta_p" class="ee_img tr_noresize" eeimg="1"> 
+ 草图数据集对应类别标签 <img src="https://www.zhihu.com/equation?tex=\eta_s" alt="\eta_s" class="ee_img tr_noresize" eeimg="1"> 

（**注意**，图片数据集和草图数据集不要求成对数据，但草图/图像和类别标签是对应的）



**训练流程**

下面伪代码中一些符号的含义如下。

+ pool：是一个队列，保存了最新生成的n个s_fake。query是取出最近的生成草图与对应标签
+ t：是提前设定好的一个阈值
+ t < u~U(0,1)：u是临时生成的服从U(0,1)的一个随机数，当随机数大于设定阈值时，本次使用的真草图更改为生成的假草图。（在真草图中时不时掺一些假的草图）
+  <img src="https://www.zhihu.com/equation?tex=L_{GAN}=\lambda_A L_{G_A}(G_A,D_A,p)+\lambda_B L_{G_B}(G_B,D_B,s,\eta_s)+\lambda_{pix}L_{pix}(G_A,G_B,p,\eta_p)+\lambda_{\eta}L_{\eta}(R,G_B,s,\eta_s)" alt="L_{GAN}=\lambda_A L_{G_A}(G_A,D_A,p)+\lambda_B L_{G_B}(G_B,D_B,s,\eta_s)+\lambda_{pix}L_{pix}(G_A,G_B,p,\eta_p)+\lambda_{\eta}L_{\eta}(R,G_B,s,\eta_s)" class="ee_img tr_noresize" eeimg="1"> 
  代表含义分别为 <img src="https://www.zhihu.com/equation?tex=G_A" alt="G_A" class="ee_img tr_noresize" eeimg="1"> 的GAN Loss， <img src="https://www.zhihu.com/equation?tex=G_B" alt="G_B" class="ee_img tr_noresize" eeimg="1"> 的GAN Loss， <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1"> 和生成 <img src="https://www.zhihu.com/equation?tex=p_{fake}" alt="p_{fake}" class="ee_img tr_noresize" eeimg="1"> 的像素损失，生成图像的分类损失



```
for p,eta_p,s,eta_s in iter(D):
	s_fake = G_A(p)
	s_c = s
	eta_c = eta_s
	if t < u~U(0,1) then # t是预先设定好的概率，U(0,1)是在区间0到1的均匀分布
		s_c,eta_c = pool.query(s_fake, eta_p)
	p_rec = G_B(s_fake, eta_p)
	p_fake = G_B(s, eta_s)
	使用p，s_c, p_rec, eta_c计算L_GAN，并更新G_A和G_B的参数
	使用s, s_fake更新D_A的参数；使用p和p_fake更新D_B的参数
	使用p，p_fake, eta_p, eta_s计算L_R，更新分类器

```

（**注意**，s和s_fake不要求是同一类的，p和p_fake也不要求是同一类）



# 6 训练相关

在论文中，作者对训练描写的比较细致，描述了使用的数据集，进行的评估指标与结果，损失函数的细节，训练采用的设备，关键的超参数，鲁棒性，消融实验证明了训练策略的有效性，t-SNE可视化不同训练策略导致结果的不同，确实草图数据集的影响。

这里我简单记录一下比较关键的几项。



## 6.1 数据集

作者使用了两个数据集Scribble和SketchyCOCO进行训练



**Scribble**：是由10类数据组成，其中大多都是空白的背景和物品的简单轮廓。其中有6类物体都是有相似的轮廓，都是圆的。作者只用了4类数据进行训练，菠萝（151张），cookie（147张），橘子（146张），西瓜（146张）。其他6类数据作为open domain，即不使用对应的草图。图片都被设置为256*256大小。



**SketchyCOCO**：由14类不同物体组成，作者把绵羊和长颈鹿作为open domain，其余12类图片拿去进行训练。



## 6.2 评估指标



定量评估指标

+ FID
+ 分类准确率



定性评估指标

+ 找测试用户进行打分



## 6.3 训练设备与实践



具体的训练实践作者没有提及，作者使用pytorch进行实现，使用了一块NVIDIA V100 GPU进行训练。训练中的batch size设置为1，学习率设置为2e-4。对于Scribble，QMUL-Sketch，SketchyCOCO数据集，作者分别训练了200，400，100个epochs。在epochs过一半时，学习率调整为原来的一半



# 7 网络和训练策略有效的思考

这里是本人对文章的一些胡乱思考，没有经过实验去证明，也就图一乐。

本文作者也说，本文的两个核心思想是

1. 使用Sketch-to-Photo和Photo-to-Sketch联合训练的方法，生成缺失的草图数据，来达到对open domain无监督学习
2. 使用了特殊的训练策略，减少了domain gap。具体说就是混合使用真实草图与生成草图，真中掺假进行训练。



对于第一个思想，很容易理解，也可以方便借鉴，对于缺少训练数据的训练工作来说可以采用生成假数据的方法进行训练学习。



对于第二个思想，真中掺假联合训练为什么比独立分阶段训练要好。我认为独立分阶段训练，其中第一阶段从图片生成草图，这个阶段的损失函数是我们人为确定的，优化生成器，也就是降低损失函数。我们的目的是使用生成出假图片来代替真图片，训练第二个生成器。第二个生成器会根据输入草图的一些特征进行生成，但这些特征是什么，其实我们并不知道。这些特征必然是在图中的，但这个图其实并不是真实的，而是我们通过另一个手段创造的，所以创作的图中到底有没有第二个生成器所看重的特征，我们也不知道。

假设一下，第一个生成器产生了完美的，和人绘制草图完全一致的图片，怎么可能达不到目标要求呢？既然实际的结果确实是没有达到预期生成效果，那么证明第一个生成器在人所看不到的地方，损失函数顾及不到的地方产生了一些和真实图片微妙的区别。联想一下图片攻击，我们人看可能两张图片一样 ，采用损失函数评估，也相差不大，但就是分类器失灵了。

![image-20210731031705366](https://raw.githubusercontent.com/miracleyoo/Markdown4Zhihu/master/Data/对立开放域自适应器从草图到图像合成/image-20210731031705366.png)

同样在本次的草图生成中，可能在第一个生成器中，产生了一些不利于第二个生成器判断的特征，这些特征我们人是看不出来的，通过简单设置Loss可能也不好消除，所以造成第二阶段不利的问题。归根结底，我认为是第一阶段和第二阶段的生成目的不一致，而中间又没有联系手段。过于分裂的两阶段，造成了最终产出效果的不佳。而真中混假的训练方式，联系起第一个生成器和第二个生成器。两者统一起来，输出结果也就更好。



最后总结一下这片文章对我的启发

+ 利用神经网络生成数据的时候，可以把两个阶段连结起来，联合训练。
+ 在证明自己方法有效时，可以使用t-SNE可视化生成结果







# 其他趣事

2. 在论文4.1.1中，作者说我们训练我们的模型在三个数据集上：Scribble数据集，SketchyCOCO数据集。。。”We train our model on three datasets: Scribble(10 classes), and SketchyCOCO(14 classes of objects).“。是我数错了吗？