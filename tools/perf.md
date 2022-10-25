#### 介绍
- perf可以看到函数的调用耗时，定位到性能瓶颈
- docker容器内的perf需要源码编译安装


#### 安装
- 内核源码准备,使用阿里云资源，不用翻墙
```
uname -r #查看系统内核版本
https://mirrors.aliyun.com/linux-kernel
wget -c *.tar.xz
tar -Jxf *.tar.xz
```
- 安装依赖
```
apt install flex bison libelf-dev libdw-dev binutils-dev libiberty-dev libslang2-dev libaudit-dev
```
- 编译和使用

```
make && make install prefix="/usr"
ps -ef 获取进程id
pref record -g -p 进程号 --sleep 60 收集60s统计信息，perf report查看性能统计信息
```