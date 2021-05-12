# 0 Beta函数的定义



定义Beta函数为

<img src="https://www.zhihu.com/equation?tex=B(x,y)=\int_0^1t^{x-1}(1-t)^{y-1}dt
%5C%5C+" alt="B(x,y)=\int_0^1t^{x-1}(1-t)^{y-1}dt
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=x,y \in C" alt="x,y \in C" class="ee_img tr_noresize" eeimg="1"> 并且 <img src="https://www.zhihu.com/equation?tex=Re(x) > 0,Re(y)>0" alt="Re(x) > 0,Re(y)>0" class="ee_img tr_noresize" eeimg="1"> 



Beta函数也经常写作

<img src="https://www.zhihu.com/equation?tex=\Beta(x+1,y+1)=\int_0^1t^x(1-t)^ydt
%5C%5C+" alt="\Beta(x+1,y+1)=\int_0^1t^x(1-t)^ydt
" class="ee_img tr_noresize" eeimg="1">




# 1 Beta函数性质



## 对称性


<img src="https://www.zhihu.com/equation?tex=\Beta(x,y)=\Beta(y,x)
%5C%5C+" alt="\Beta(x,y)=\Beta(y,x)
" class="ee_img tr_noresize" eeimg="1">

正确性证明如下，关键点是 <img src="https://www.zhihu.com/equation?tex=s=1-t" alt="s=1-t" class="ee_img tr_noresize" eeimg="1"> 

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\Beta(x,y)&=\int_0^1t^{x-1}(1-t)^{y-1}dt\\
&\overset{s=1-t}{=}\int_0^1(1-s)^{x-1}s^{y-1}ds\\
&=\Beta(y,x)

\end{aligned}
%5C%5C+" alt="\begin{aligned}
\Beta(x,y)&=\int_0^1t^{x-1}(1-t)^{y-1}dt\\
&\overset{s=1-t}{=}\int_0^1(1-s)^{x-1}s^{y-1}ds\\
&=\Beta(y,x)

\end{aligned}
" class="ee_img tr_noresize" eeimg="1">




## 与Gamma函数联系

Beta函数可以用Gamma函数来表示。

<img src="https://www.zhihu.com/equation?tex=\Beta(x,y)=\frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}
%5C%5C+" alt="\Beta(x,y)=\frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}
" class="ee_img tr_noresize" eeimg="1">




