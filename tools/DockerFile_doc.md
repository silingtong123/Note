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

### 单机多卡测试
- 启动容器: docker run --name slt-gpu   --privileged=true --pid=host --ipc=host --net=host --cap-add=SYS_ADMIN  --cap-add=SYS_PTRACE -it -v /data/slt/:/Newdeeprec  -w /Newdeeprec --runtime=nvidia 155b56c7d050  /bin/bash
- 设置代理：
  - export http_proxy=http://10.0.24.120:7895
  - export https_proxy=http://10.0.24.120:7895
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
  ```
- 启动命令：CUDA_VISIBLE_DEVICES=0,1 python -m hybridbackend.run python ranking/taobao/train.py /Newdeeprec/day_0.parquet
- 查看gpu利用率：watch -n 2 -d nvidia-smi
- https://cdn.233xyx.com/1671162381712_317.zip
- http://easyrec.oss-cn-beijing.aliyuncs.com/data/taobao/day_0.parquet
- http://easyrec.oss-cn-beijing.aliyuncs.com/data/taobao/day_1.parquet 
- http://easyrec.oss-cn-beijing.aliyuncs.com/data/taobao/day_2.parquet 

### 磁盘预测测试
- export TF_SSDHASH_ASYNC_COMPACTION=0
- export LD_LIBRARY_PATH=/go_serving/app/controller/serving/tf_model/lib/:$LD_LIBRARY_PATH