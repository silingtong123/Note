# Dockefile

## Dockerfile的基本结构

Dockerfile 一般分为四部分：基础镜像信息、维护者信息、镜像操作指令和容器启动时执行指令，’#’ 为 Dockerfile 中的注释

## Dockerfile文件说明

### 常用指令
- FROM：指定基础镜像，必须为第一个命令
- MAINTAINER： 维护者信息(可为空)
  - 格式：MAINTAINER <name>
  - 示例：MAINTAINER Jasper Xu
- RUN: 构建镜像时执行的命令
  - shell执行：RUN <command>
  - exec执行： RUN ["executable", "param1", "param2"]
- ADD：将本地文件添加到容器中，tar类型文件会自动解压(网络压缩资源不会被解压)，可以访问网络资源，类似wget
  - 格式：ADD <src>... <dest>
  - 示例：ADD hom* /mydir/  # 添加所有以"hom"开头的文件
- COPY: 功能类似ADD，但是是不会自动解压文件，也不能访问网络资源
- CMD: 构建容器后调用，也就是在容器启动时才进行调用
  - CMD ["executable","param1","param2"] (执行可执行文件，优先)
  - CMD ["param1","param2"] (设置了ENTRYPOINT，则直接调用ENTRYPOINT添加参数)
  - CMD command param1 param2 (执行shell内部命令)
- ENTRYPOINT: 配置容器，使其可执行化。配合CMD可省去"application"，只使用参数。
  - ENTRYPOINT ["executable", "param1", "param2"] (可执行文件, 优先)
  - ENTRYPOINT command param1 param2 (shell内部命令)
- ENV：设置环境变量， docker run 生效
- EXPOSE：指定于外界交互的端口
- VOLUME：用于指定持久化目录
- WORKDIR：工作目录，类似于cd命令
- ARG：用于指定传递给构建Image运行时的变量, DOCKER BUILD 命令时生效

注：
- CMD不同于RUN，CMD用于指定在容器启动时所要执行的命令，而RUN用于指定镜像构建时所要执行的命令
- Dockerfile中的cmd容器被启动参数覆盖，如：CMD ["Hello World"] docker run -it test:1  bash 不会打印Hello World，被bash命令覆盖
- 如果一个Dockerfile中有多个CMD或ENTRYPOINT，只有最后一个会生效，前面其他的都会被覆盖
-  如果有多个cmd，建议写成shell脚本，cmd改为执行脚本CMD["/bin/bash", "start.sh"]
-  ARG 存在于 docker build 命令执行期间。默认值写在 Dockerfile 里。如果需要修改，可以通过 docker build 命令里的 --build-arg 参数来指定
-  ENV 存在于 docker run 命令执行期间。默认值写在 Dockerfile 里。如果要修改，可以通过 docker run 命令的 --env 参数来指定
-  如果要把 ARG 的值保存到 container 运行起来之后仍然可以可用，则需要在 ARG 之后写一个 ENV

### dockerfile中可被覆盖的指令
- ENTRYPOINT     docker run --entrypoint /bin/bash mysql:latest 
- CMD       docker run ... <New_Command>
- EXPOSE  docker run --expose="port_number:port_number"
- ENV    docker run -e "key=value" ...
- VOLUME    docker run -v ...
- USER docker run -u="" ...
- WORKDIR  docker run -w="" ...


### docker的常用命令
- 登录镜像仓库：docker login --username=aliyun-registry@1884010027272209 metaapp-registry-vpc.cn-beijing.cr.aliyuncs.com
- 构建镜像（需要dockerfile）:docker build  -t metaapp-registry-vpc.cn-beijing.cr.aliyuncs.com/metaops/deeprec-serving:deepfm-0324-pd-v1 .
- 推送镜像到仓库：docker push   metaapp-registry-vpc.cn-beijing.cr.aliyuncs.com/metaops/deeprec-ctr-test:v1
- 拉取镜像：docker pull metaapp-registry-vpc.cn-beijing.cr.aliyuncs.com/metaops/deeprec-ctr-test:v1
- 启动容器：docker run --name serving-cpu-processor  --net=host  -it -v $PWD:/Newdeeprec  -w /Newdeeprec b92167a4cab6 /bin/bash
- 查看容器：docker ps -a | grep
- 进入容器：docker exec -it continer_name /bin/bash
- 启动容器：docker start continer_name
- 删除容器：docker rm continer_name
- 查看镜像： docker images | grep
- 删除镜像：docker rmi image_name
- 重命名镜像：docker tag 1417b43a3ff5 faster-rcnn-3d:v1
- 查看容器启动命令：docker ps -a --no-trunc | grep container_name   # 通过docker --no-trunc参数来详细展示容器运行命令
- 查看容器磁盘： docker system df
- overlay2过大问题排查：
  - cd /var/lib/docker/overlay2
  - du -h --max-depth=1 ，假设最大的为6b6572a745dccd415280c6b4eacdfb531c61c5f3618be64dc6e20f00b72d6951
  - 查看占用空间的pid，以及对应的容器名称： docker ps -q | xargs docker inspect --format '{{.State.Pid}}, {{.Name}}, {{.GraphDriver.Data.WorkDir}}' | grep "6b6572a745dccd415280c6b4eacdfb531c61c5f3618be64dc6e20f00b72d6951"


- 指定显卡：--runtime=nvidia  -e NVIDIA_VISIBLE_DEVICES=1,2, 此时内部只能看到两张卡，但是指定 CUDA_VISIBLE_DEVICES=i 会直接使用外部的第i张卡   
- export NVIDIA_VISIBLE_DEVICES=
- 指定使用cpu_allocator: export TF_DISABLE_EV_ALLOCATOR=1
- 用户设置 CUDA_VISIBLE_DEVICES=3,2,1,0，那么deeprec中看到的编号0,1,2,3对应的物理GPU是3,2,1,0。
- 172.16.11.2
- 查看GPU信息
  - Nvida公司：GPU Cuda/AMD公司：GPU  ROCm
  - watch -n 2 -d nvidia-smi
  - 查看gpu通信通道：nvidia-smi -L 
  - 查看gpu互联：nvidia-smi topo -m
  - lspci
  - gpu debug信息： NCCL_DEBUG=INFO 和 NCCL_DEBUG_SUBSYS=ALL
  - 设置网卡 NCCL_SOCKET_IFNAME=eth0  export NCCL_IB_DISABLE=1
  
### 单机多卡测试
- 启动容器: 
```shell
#启动容器，不限制cpu和gpu
docker run --name slt-gpu   --privileged=true --pid=host \
--ipc=host --net=host \
--cap-add=SYS_ADMIN  \
--cap-add=SYS_PTRACE \
-it -v /data/slt/:/Newdeeprec  -w /Newdeeprec \
--runtime=nvidia 155b56c7d050  /bin/bash

#启动容器，限制cpu 0-7和gpu 2
docker run --name slt-robin-262-8c-cpuset \
--privileged=true --pid=host \
--ipc=host --net=host --cap-add=SYS_ADMIN  \
-it -v /home/slt/:/Newdeeprec --cpuset-cpus=0-7 --memory=57g -w /
Newdeeprec -e NVIDIA_VISIBLE_DEVICES=2 --runtime=nvidia  \ metaapp-registry-vpc.cn-beijing.cr.aliyuncs.com/meta-rec/robin-hb:v262  /bin/bash

#启动大模型
docker run -it  --name=slt-llm-0626 \
--volume /data/slt:/workdir -v /oss:/oss -w /workdir \
--ipc=host --network host -P  \
--device=/dev/infiniband/rdma_cm \
--device=/dev/infiniband/uverbs0 \
--device=/dev/infiniband/uverbs1 \
--ulimit memlock=-1 \
metaapp-registry-vpc.cn-beijing.cr.aliyuncs.com/metaops/llm:1.1 /bin/bash
```
- 参数解释：
  - --privileged=true: 使容器root权限变成真正的root权限，否则就只能等于外部的普调用户权限，可以看到很多host上的设备比如设置privileged=true能看到NVIDIA_VISIBLE_DEVICES指定外的显卡
  - --pid=host 容器的PID命名空间
  - --ipc=host  docker中的进程要与宿主机使用共享内存通信,否则docker容易出现shm不够，默认是64M
  - --net=host
  - -P :是容器内部端口随机映射到主机的端口, -p ip:hostPort:containerPort容器内部端口绑定到指定的主机端口
- 设置代理：
  - export http_proxy=http://10.0.24.95:8888
  - export http_proxy=http://10.0.24.95:8888
- 取消代理(单机多卡测试，需要取消代理)：
  - unset http_proxy
  - unset https_proxy
- 设置环境变量：export PYTHONPATH=${PYTHONPATH}:./
- libcuda.so大小为0，需要拷贝过去：cp -r *470.82.01  /usr/lib/x86_64-linux-gnu/
- ```python
  hb.data.ParquetDataset(
        filenames,
        batch_size=batch_size,
        num_parallel_reads=len(filenames),
        num_parallel_parser_calls=self._args.num_parsers,
        drop_remainder=True，
        partition_count =hb.context.world_size，
        partition_index =hb.context.rank
  )

        if flags.is_hb_training:
            hooks=[]
            hooks.append(tf.train.StopAtStepHook(last_step=100))
            hooks.append(tf.train.ProfilerHook(save_steps=100, output_dir=self.__hb_training))
            return tf.train.MonitoredTrainingSession(master="", hooks= hooks, save_checkpoint_steps=1000000, checkpoint_dir=self.__hb_training, config=sess_config)  
  ```
- 启动命令：CUDA_VISIBLE_DEVICES=0,1 python -m hybridbackend.run python ranking/taobao/train.py /Newdeeprec/day_0.parquet
- /usr/local/lib/python3.6/dist-packages/hybridbackend/tensorflow/plugins/deeprec/ev.py
- 查看gpu利用率：watch -n 2 -d nvidia-smi
- https://cdn.233xyx.com/1671162381712_317.zip
- http://easyrec.oss-cn-beijing.aliyuncs.com/data/taobao/day_0.parquet
- http://easyrec.oss-cn-beijing.aliyuncs.com/data/taobao/day_1.parquet 
- http://easyrec.oss-cn-beijing.aliyuncs.com/data/taobao/day_2.parquet 

### 磁盘预测测试
- export TF_SSDHASH_ASYNC_COMPACTION=0
- export LD_LIBRARY_PATH=/go_serving/app/controller/serving/tf_model/lib/:$LD_LIBRARY_PATH

### NVIDIA docker 容器中devel runtime base三种文件的区别
- base版本：该版本是从cuda9.0开始，包含了部署预构建cuda应用程序的最低限度（libcudart）。
如果用户需要自己安装自己需要的cuda包，可以选择使用这个image版本，但如果想省事儿，则不建议使用该image，会多出许多麻烦。
-runtime版本：该版本通过添加cuda工具包中的所有共享库开扩展基本image。如果使用多个cuda库的预构建应用程序，可使用此image。但是如果想借助cuda中的头文件对自己的工程进行编译，则会出现找不到文件的错误。
- devel版本：通过添加编译器工具链，测试工具，头文件和静态库来扩展运行的image，使用此图像可以从源代码编译cuda应用程序

### docker绑核
- docker inspect dockerid|grep -i pid 查看容器核数
- taskset -pc 0-31  pid

### A100卡安装出错
- failed call to cuInit: CUDA_ERROR_SYSTEM_NOT_READY: system not yet initialized
- 缺少安装nvidia-fabricmanager：https://forums.developer.nvidia.com/t/error-802-system-not-yet-initialized-cuda-11-3/234955/3
- 安装参考：https://www.volcengine.com/docs/6419/73634
- 启动：
  - systemctl start nvidia-fabricmanager 开启服务
  - systemctl status nvidia-fabricmanager 状态查看
  - systemctl enable nvidia-fabricmanager 开机启动
### nvidia driver安装
- https://www.alibabacloud.com/help/zh/elastic-gpu-service/latest/install-a-gpu-driver-on-a-linux-gpu-accelerated-compute-optimized-instance?spm=a2c63.p38356.0.0.67786b206kLbGI#concept-ecy-qrz-wgb