# 课程笔记 cuda
### 课程信息说明
课程网站：https://www.bilibili.com/video/BV1Qc411J7DB?p=45&vd_source=60ffcefe479d27f859009dd3a47fcea1

这门课主要讲解了cuda中的常用函数与算法，并做了相关实验

### 课程代码说明

#### reduce 

       reduce文件夹下面的代码展示了邻域求和  间域求和的方法，以及减少warp divergence的方法，不过gpu的用时一直在波动，啥子情况？计时函数有问题？   

       cuda会自动优化，所以减少了warp divergence以后的测试时间和不减少差不多。(咱也不清楚怎么关自动优化)
       
       间域并行比领域并行快了很多，具体原因是啥？ 可能是因为邻域并行的warp里有一般的线程不工作？
       
       关于循环展开，展开2次比不展开提升很多，展开次数越多，效果越不明显，感觉展开4次或者8次就差不多了

       ![Image text](https://github.com/caixiao-0725/cuda-course/blob/main/pictures/reduce.png)