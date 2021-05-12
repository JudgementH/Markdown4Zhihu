# 0 Gamma函数的由来

首先看看阶乘的定义

<img src="https://www.zhihu.com/equation?tex=n! := n(n-1)(n-2)\cdots2\times1
" alt="n! := n(n-1)(n-2)\cdots2\times1
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=n \in N^+" alt="n \in N^+" class="ee_img tr_noresize" eeimg="1"> 。



把阶乘的定义拓展到复数域，阶乘就有了新的叫法 <img src="https://www.zhihu.com/equation?tex=\Gamma" alt="\Gamma" class="ee_img tr_noresize" eeimg="1"> （Gamma）函数。

<img src="https://www.zhihu.com/equation?tex=\Gamma (z) = \int_0^{+\infty}x^{z -1} e^{-x}dx
" alt="\Gamma (z) = \int_0^{+\infty}x^{z -1} e^{-x}dx
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=z \in \C" alt="z \in \C" class="ee_img tr_noresize" eeimg="1"> 且 <img src="https://www.zhihu.com/equation?tex=Re(z)>0" alt="Re(z)>0" class="ee_img tr_noresize" eeimg="1"> 。



>我并不去理解Gamma函数，我只去习惯Gamma函数



Gamma函数更方便的写为以下的形式

<img src="https://www.zhihu.com/equation?tex=\Gamma(z+1)=\int_0^{+\infty}x^ze^{-x}dx
" alt="\Gamma(z+1)=\int_0^{+\infty}x^ze^{-x}dx
" class="ee_img tr_noresize" eeimg="1">
也就是把积分内指数项的 <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 的参数改为 <img src="https://www.zhihu.com/equation?tex=z" alt="z" class="ee_img tr_noresize" eeimg="1"> 



根据公式，Gamma函数也可以写为

<img src="https://www.zhihu.com/equation?tex=\Gamma(\frac{z+1}{2})=2\int_0^{+\infty}x^{2z-1}e^{x^2}dx
" alt="\Gamma(\frac{z+1}{2})=2\int_0^{+\infty}x^{2z-1}e^{x^2}dx
" class="ee_img tr_noresize" eeimg="1">




# 1 Gamma函数的性质



## 递推公式

Gamma函数满足以下的递推关系

<img src="https://www.zhihu.com/equation?tex=\Gamma(z + 1) = z \Gamma(z)
" alt="\Gamma(z + 1) = z \Gamma(z)
" class="ee_img tr_noresize" eeimg="1">
其正确性证明如下，运用分部积分即可

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\Gamma(z+1)&=\int_0^{+\infty}x^ze^{-x}dx\\
&=-x^ze^{-x}|_0^{+\infty}+z\int_0^{+\infty}x^{z-1}e^{-x}dx\\
&=0 +z\Gamma(z)\\
&=z\Gamma(z)
\end{aligned}
" alt="\begin{aligned}
\Gamma(z+1)&=\int_0^{+\infty}x^ze^{-x}dx\\
&=-x^ze^{-x}|_0^{+\infty}+z\int_0^{+\infty}x^{z-1}e^{-x}dx\\
&=0 +z\Gamma(z)\\
&=z\Gamma(z)
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">


根据这个递推公式，当 <img src="https://www.zhihu.com/equation?tex=z\in \N" alt="z\in \N" class="ee_img tr_noresize" eeimg="1"> 时，有以下公式成立

<img src="https://www.zhihu.com/equation?tex=\Gamma(z+1)=z!,z\in N
" alt="\Gamma(z+1)=z!,z\in N
" class="ee_img tr_noresize" eeimg="1">
也就是说，运用Gamma函数的递推公式可以很方便的把积分化为阶乘公式



运用这个递推公式，我们最终会把一个Gamma函数化为连乘，最后会归结为一些特殊点的Gamma函数值

+  <img src="https://www.zhihu.com/equation?tex=\Gamma(1)=1" alt="\Gamma(1)=1" class="ee_img tr_noresize" eeimg="1"> 
+  <img src="https://www.zhihu.com/equation?tex=\Gamma(\frac{1}{2})=\sqrt{\pi}" alt="\Gamma(\frac{1}{2})=\sqrt{\pi}" class="ee_img tr_noresize" eeimg="1"> 
+  <img src="https://www.zhihu.com/equation?tex=\Gamma(-\frac{1}{2})=-2\sqrt{\pi}" alt="\Gamma(-\frac{1}{2})=-2\sqrt{\pi}" class="ee_img tr_noresize" eeimg="1"> 



## 余元公式



<img src="https://www.zhihu.com/equation?tex=\Gamma(z)\Gamma(1-z)=\frac{z}{sin\pi z}
" alt="\Gamma(z)\Gamma(1-z)=\frac{z}{sin\pi z}
" class="ee_img tr_noresize" eeimg="1">




