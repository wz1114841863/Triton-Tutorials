# 针对 RTX 3060 (Ampere 架构)
export TORCH_CUDA_ARCH_LIST="8.6"

矩阵转置：

需要考虑合并访问和写入冲突问题。

这里先注释掉了对于cute的调用。
```
 python ./mat_transpose.py
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.30605388ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.29381537ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.32569385ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.29659820ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.29612279ms
                  out_f32_diagnonal: [0.0, 1024.0, 1.0], validate True , time:0.29708552ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.33749938ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.33225060ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.29667497ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.30128074ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.29960251ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.29999399ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.30027413ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.30104303ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.30114222ms
Traceback (most recent call last):
  File "/home/wz/AI/Triton-Tutorials/kernels/mat_transpose/./mat_transpose.py", line 138, in <module>
    lib.mat_transpose_cute_col2row_reg,
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'mat_transpose_lib' has no attribute 'mat_transpose_cute_col2row_reg'. Did you mean: 'mat_transpose_f32_col2row'?
(triton) wz@DESKTOP-953MNSG:~/AI/Triton-Tutorials/kernels/mat_transpose$ python ./mat_transpose.py
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.30644774ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.29511499ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.31159019ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.29896498ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.29800558ms
                  out_f32_diagnonal: [0.0, 1024.0, 1.0], validate True , time:0.29752731ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.30886078ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.34469056ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.29917336ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.30220175ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.29969025ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.30042553ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.30019617ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.30243039ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.30182457ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:0.60540891ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.30390286ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.63000679ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.63958597ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.62580109ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.60386014ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.59613585ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.63557816ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.60301495ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.58986115ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.59997296ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.60092115ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.59687948ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.61130738ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.68283653ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:1.01148295ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:1.34692454ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:0.70401621ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:1.27122855ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:1.35092854ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:1.26876402ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:1.17416954ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:1.17300344ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:1.34879494ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:1.20954227ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:1.18021655ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:1.19528270ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:1.19824481ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:1.18359351ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:1.19786644ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:1.19325542ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:1.20086741ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:2.44777322ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:1.82945967ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:2.52051473ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:3.20525837ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:2.52734303ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:2.38142562ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:2.37925982ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:3.50675511ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:2.50125527ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:2.42868996ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:2.38417220ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:2.41674352ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:2.36532712ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:2.46023679ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:2.38317442ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:2.39892387ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:4.93052769ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:2.68608141ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.62781000ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.72359562ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.60955048ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.59249282ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.59168482ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.68433642ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.64897299ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.59571147ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.60525465ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.60481930ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.59887552ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.60481954ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.60432220ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.60976386ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:1.22934127ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:0.59914017ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:1.26519895ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:1.35224366ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:1.25092149ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:1.17655611ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:1.17908049ms
                  out_f32_diagnonal: [0.0, 2048.0, 1.0], validate True , time:1.16395783ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:1.32419896ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:1.20159864ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:1.18508387ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:1.20485711ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:1.21512437ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:1.19977212ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:1.21467662ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:1.20609307ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:1.21775484ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:2.46846151ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:1.19686627ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:2.52308035ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:2.80544877ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:2.54002523ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:2.33282542ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:2.34843278ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:2.74675417ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:2.41923952ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:2.37613916ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:2.39561343ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:2.40702629ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:2.37521768ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:2.41295648ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:2.39251351ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:2.41180325ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:4.95471311ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:2.38223934ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:5.03277659ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:6.18046641ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:5.11728907ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:4.76540232ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:4.76209474ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:6.75721931ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:5.03708673ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:4.88705993ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:4.80459857ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:4.84465694ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:4.73209429ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:4.89230061ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:4.78451777ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:4.88613939ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:9.95812368ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:4.90518188ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:1.28766084ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:1.32130051ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:1.25054002ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:1.18504953ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:1.18559265ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:1.29726934ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:1.20785809ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:1.21353364ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:1.23226738ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:1.23093390ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:1.20527196ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:1.21802449ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:1.21520448ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:1.22204852ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:2.46320200ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:1.20581365ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:2.54407930ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:3.47669077ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:3.17731357ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:2.40582323ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:2.35215855ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:2.91982579ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:2.66509032ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:2.39350557ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:2.40827775ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:2.72562766ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:2.42328739ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:3.12157536ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:2.42231393ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:2.43596339ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:5.07852411ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:2.40589571ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:5.04376388ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:5.58186197ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:5.25021625ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:4.66272378ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:4.68186712ms
                  out_f32_diagnonal: [0.0, 4096.0, 1.0], validate True , time:4.58946586ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:5.41052961ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:4.95420766ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:4.73999333ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:5.39169407ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:4.82553363ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:4.77422357ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:4.82378101ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:4.79372787ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:4.82199693ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:10.12292409ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:4.77500391ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:10.08255196ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:13.31123757ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:10.80378294ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:10.03971171ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:9.54809141ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:15.59674835ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:11.23782754ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:10.13791561ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:10.01866341ms
       out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:9.89237261ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:9.76302218ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:9.85268283ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:9.61885166ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:9.69179368ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:20.29574180ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:9.54456282ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:2.52842832ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:2.74649930ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:2.58493328ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:2.34976530ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:2.33890557ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:2.67848325ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:2.57396531ms
              out_f32x4_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:2.36200142ms
              out_f32x4_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:2.41355252ms
       out_f32x4_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:2.41962743ms
       out_f32x4_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:2.38445497ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:2.42243361ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:2.40268230ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:2.42702746ms
                         out_f32_th: [0.0, 1024.0, 1.0], validate True , time:4.95002151ms
                out_f32_th_compiled: [0.0, 1024.0, 1.0], validate True , time:2.39831066ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:5.03267527ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:5.49397349ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:5.63167620ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:4.69368577ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:4.69209313ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:5.38238072ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:5.47860599ms
              out_f32x4_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:4.72581840ms
              out_f32x4_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:4.81065750ms
       out_f32x4_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:4.83510447ms
       out_f32x4_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:4.75194645ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:4.84176779ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:4.79272127ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:4.83959389ms
                         out_f32_th: [0.0, 2048.0, 1.0], validate True , time:10.26485491ms
                out_f32_th_compiled: [0.0, 2048.0, 1.0], validate True , time:4.72816658ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:10.05532742ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:10.93936276ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:11.06762338ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:9.41446495ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:9.41558146ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:11.07248735ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:10.82570982ms
              out_f32x4_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:9.50288224ms
              out_f32x4_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:9.60016418ms
       out_f32x4_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:9.66719508ms
       out_f32x4_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:9.53549361ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:9.66438365ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:9.57110810ms
out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:9.65128779ms
                         out_f32_th: [0.0, 4096.0, 1.0], validate True , time:22.20534515ms
                out_f32_th_compiled: [0.0, 4096.0, 1.0], validate True , time:22.52406764ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:29.28501105ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:32.79897714ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:32.67244840ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:21.08730960ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:19.43807459ms
                  out_f32_diagnonal: [0.0, 8192.0, 1.0], validate True , time:18.80629182ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:29.01643109ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:51.57150292ms
              out_f32x4_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:22.54033422ms
              out_f32x4_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:23.25784564ms
        out_f32x4_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:252.24811721ms
       out_f32x4_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:22.29365301ms
   out_f32x4_shared_bcf_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:34.39414716ms
   out_f32x4_shared_bcf_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:22.36395812ms
  out_f32x4_shared_bcf_merge_write_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:25.79358101ms
                         out_f32_th: [0.0, 8192.0, 1.0], validate True , time:58.54820800ms
                out_f32_th_compiled: [0.0, 8192.0, 1.0], validate True , time:27.80035496ms
----------------------------------------------------------------------------------------------------------------------------------
```
