# AnomalyDetection
## LSTM对时间序列的数据做预测，通过偏离值的大小达到异常检测的目的
### 1.思路
&#8195;&#8195;用一个网格7x24个小时的数据预测该网格第7x24+1个小时的in out，如果这个预测in out与真实值的差距大于一个阈值（该阈值最终取3,见Fig3），就把这小时对应的网格当做一个异常的网格，可以达到实时检测的效果
<div align=center><img width="600" height="300" src="https://github.com/DQ0408/AnomalyDetection/blob/master/imgs/Fig1.png"/></div>
<div align=center> Fig1.某网格7*24小时（一周）的时序图 </div>

<div align=center><img width="600" height="300" src="https://github.com/DQ0408/AnomalyDetection/blob/master/imgs/Fig2.png"/></div>
<div align=center>Fig2.某网格相邻四周的时序图</div>

<div align=center><img width="600" height="300" src="https://github.com/DQ0408/AnomalyDetection/blob/master/imgs/Fig3.jpg"/></div>
<div align=center>Fig3_1.训练loss与epoch的箱型图</div>

<div align=center><img width="600" height="300" src="https://github.com/DQ0408/AnomalyDetection/blob/master/imgs/Fig6.jpg"/></div>
<div align=center>Fig3_2.最后一个epoch的loss箱型图</div>

<div align=center>（异常的阈值取3因为最后一个epoch的均方差loss基本小于9）</div>

<div align=center><img width="600" height="300" src="https://github.com/DQ0408/AnomalyDetection/blob/master/imgs/Fig4.jpg"/></div>
<div align=center>Fig4.训练loss与每个step的曲线图</div>

<div align=center><img width="600   " height="300" src="https://github.com/DQ0408/AnomalyDetection/blob/master/imgs/Fig5.png"/></div>
<div align=center>Fig5.某网格第五周的预测图</div>

#### Fig5中红线是真实值±3，代表异常的阈值曲线，绿线是预测值曲线，蓝线是真实值的曲线，绿线超出红线即视为异常

### 2.流程

>1)过滤掉稀疏的网格，一个网格用一个向量存储2088个小时的in/out，如果2088个值里边有超过三分之一的值为0，则过滤掉，不用该网格的数据进行模型训练。原因：一天24小时，可以有三分之一的时间比如0a.m.-8a.m. 没有in/out，其他时间段有in/out，比较符合该数据合理的直觉，共1317个符合条件的向量，原始数据有104000*2个向量

>2）最终训练数据生成：将符合条件的向量用一个24*7的窗口划分

>>A.为了减少模型过拟合，窗口之间不重合

>>B.为了增加模型的适用性，划分窗口时加一个offset，使得模型从一周的随便一个小时起开始预测都可以，
最终有大约1317*(2088/(24*7))=16368个训练样本
