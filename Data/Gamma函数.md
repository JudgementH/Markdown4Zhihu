# 0 Gamma函数的由来

首先看看阶乘的定义
$$
n! := n(n-1)(n-2)\cdots2\times1
$$
其中$n \in N^+$。



把阶乘的定义拓展到复数域，阶乘就有了新的叫法$\Gamma$（Gamma）函数。
$$
\Gamma (z) = \int_0^{+\infty}x^{z -1} e^{-x}dx
$$
其中$z \in \C$且$Re(z)>0$。



>我并不去理解Gamma函数，我只去习惯Gamma函数



Gamma函数更方便的写为以下的形式
$$
\Gamma(z+1)=\int_0^{+\infty}x^ze^{-x}dx
$$
也就是把积分内指数项的$x$的参数改为$z$



根据公式，Gamma函数也可以写为
$$
\Gamma(\frac{z+1}{2})=2\int_0^{+\infty}x^{2z-1}e^{x^2}dx
$$




# 1 Gamma函数的性质



## 递推公式

Gamma函数满足以下的递推关系
$$
\Gamma(z + 1) = z \Gamma(z)
$$
其正确性证明如下，运用分部积分即可
$$
\begin{aligned}
\Gamma(z+1)&=\int_0^{+\infty}x^ze^{-x}dx\\
&=-x^ze^{-x}|_0^{+\infty}+z\int_0^{+\infty}x^{z-1}e^{-x}dx\\
&=0 +z\Gamma(z)\\
&=z\Gamma(z)
\end{aligned}
$$


根据这个递推公式，当$z\in \N$时，有以下公式成立
$$
\Gamma(z+1)=z!,z\in N
$$
也就是说，运用Gamma函数的递推公式可以很方便的把积分化为阶乘公式



运用这个递推公式，我们最终会把一个Gamma函数化为连乘，最后会归结为一些特殊点的Gamma函数值

+ $\Gamma(1)=1$
+ $\Gamma(\frac{1}{2})=\sqrt{\pi}$
+ $\Gamma(-\frac{1}{2})=-2\sqrt{\pi}$



## 余元公式


$$
\Gamma(z)\Gamma(1-z)=\frac{z}{sin\pi z}
$$




