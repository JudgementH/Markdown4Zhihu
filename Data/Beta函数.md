# 0 Beta函数的定义



定义Beta函数为
$$
B(x,y)=\int_0^1t^{x-1}(1-t)^{y-1}dt
$$
其中$x,y \in C$并且$Re(x) > 0,Re(y)>0$



Beta函数也经常写作
$$
\Beta(x+1,y+1)=\int_0^1t^x(1-t)^ydt
$$




# 1 Beta函数性质



## 对称性

$$
\Beta(x,y)=\Beta(y,x)
$$

正确性证明如下，关键点是$s=1-t$
$$
\begin{aligned}
\Beta(x,y)&=\int_0^1t^{x-1}(1-t)^{y-1}dt\\
&\overset{s=1-t}{=}\int_0^1(1-s)^{x-1}s^{y-1}ds\\
&=\Beta(y,x)

\end{aligned}
$$




## 与Gamma函数联系

Beta函数可以用Gamma函数来表示。
$$
\Beta(x,y)=\frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}
$$




