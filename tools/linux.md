### 常见工具和命令
#### vim
- 容器内鼠标右键无效出现insert(VISUAL):set mouse-=a
- 复制保留格式：set paste
- 批量注释:ctrl+v 进行块选择，加上大写的I 写注释符号#或\\
- 取消批量注释：ctrl+v 进行块选择，选择符号+d
- 多行缩进：可使用ctrl+v进入visual模式，然后用光标移动或者上下键方法选中要缩进的多行代码，shift+‘>’ 向左缩进，‘<’向右缩进


#### ldd
- ldd *.so : 可以列出一个程序所需要得动态链接库（so）,左边为库名，右边为库的文件路径

#### GDB
- 开启： ulimit -c unlimited

#### linux快速搭建FTP
1. - SimpleHTTPServer使用方法
   - 进入待分享的目录 
   - 执行命令python -m SimpleHTTPServer 端口号 注意：不填端口号则默认使用8000端口。 
2. Python 3的用法与在Python 2的用法相似(python3 -m http.server  或者 python3 -m http.server 端口号)

3. 浏览器访问该主机的地址：http://IP:端口号/
#### 常用命令
- 查看系统信息：uname -a 
- 确定系统是centos还是ubuntu,前者是yum，后者是apt-get
- 查看系统版本
  - centos版本：rpm -q centos-release
  - uabanu版本：cat /proc/version
  - cpu型号： cat /proc/cpuinfo
  - 查看cpu详细信息：lscpu
- 查看进程文件描述符号：
  - ps -ef  |  grep 4379  找进程id,如模型_4379 ， 得到进程27511
  - ls -l /proc/27511/fd | wc -l 统计文件描述符多少  
- 查看端口：Linux 查看端口占用情况可以使用 lsof 和 netstat 命令
  - netstat -ntlp   | grep 1095 可用得到pid
  - 根据pid查看进程信息  ps -ef | | grep 1095
  - lsof -i:8898
  - 查看端口的所有连接： netstat -ntap | grep 4973
- 查看内存：free -h 转换为Mb或者Gb
- Find命令
  - 递归查看：find . -name “*.txt”
  - 非递归查看： find . -name “*.txt” -maxdepth 1
  - 递归删除： find . -name "111"  | xargs rm -rf
  - 常用环境变量：
    - PATH：可执行文件路径
    - LIBRARY_PATH：程序编译期间查找动态链接库时指定查找共享库的路径
    - LD_LIBRARY_PATH：程序加载运行期间查找动态链接库时指定除了系统默认路径之外的其他路径
- 建立链接文件：
  - 软连接： ln -s a a_softlink 删除链接文件，不影响原文件
  - 硬链接： ln a a_hardlink 删除链接文件，会将原文件一起删除
- du 显示文件或目录所占用的磁盘空间
  - 查看目录大小：du -sh /silingtong/tmp
  - 递归显示目录所有文件大小：du  -h 
  - 非递归显示目录大小：du -h --max-depth=1
- 常用路径：
  - /usr/bin：系统预装的一些可执行程序，随系统升级会改变
  - /usr/local/bin：用户安装的可执行程序，不受系统升级影响，用户编译安装软件时，一般放到/usr/local目录下
  - make install：安装在 /usr/local/lib和/usr/local/include
  - 其他库路径：/usr/local/lib64 /usr/lib /usr/lib64
- Ubuntu之ld搜索路径顺序：
  - 静态库链接时搜索路径顺序
    - ld会去找GCC命令中的参数-L
    - 再找gcc的环境变量LIBRARY_PATH
    - 再找内定目录 /lib /usr/lib /usr/local/lib 这是当初compile gcc时写在程序内的
  - 动态链接时、执行时搜索路径顺序
    -  编译目标代码时指定的动态库搜索路径
    -  环境变量LD_LIBRARY_PATH指定的动态库搜索路径
    -  配置文件/etc/ld.so.conf中指定的动态库搜索路径
    -  默认的动态库搜索路径/lib 
    -  默认的动态库搜索路径/usr/lib
 - ubantu永久修改环境变量
   - vim ~/.bashrc
   - export PATH="$PATH:$HOME/bin"
   - source ~/.bashrc
 - tar命令
   - 压缩文件：tar  -zcvf   压缩文件名.tar.gz   被压缩文件名
   - 解压缩：tar  -zxvf   压缩文件名.tar.gz
 - telnet 
   - ping ip: ping www.baidu.com
   - 端口测试：telnet 114.80.67.193 8080
 - 添加用户
   - useradd username
   - passwd 123456
- GPU命令
  - Nvida GPU Cuda/AMD GPU  ROCm
  - watch -n 2 -d nvidia-smi
  - 查看gpu通信通道：nvidia-smi -L 
  - 查看gpu互联：nvidia-smi topo -m
  - lspci

#### 同步，异步，并行和并发

- 同步：是指一个进程在执行某个请求的时候,如果该请求需要一段时间才能返回信息,那么这个进程会一直等待下去,直到收到返回信息才继续执行下去
- 异步:异步是指进程不需要一直等待下去,而是继续执行下面的操作,不管其他进程的状态,当有信息返回的时候会通知进程进行处理,这样就可以提高执行的效率了
- 并发：把任务在不同的时间点交给处理器进行处理。在同一时间点,任务并不会同时运行
- 并行：把每一个任务分配给每一个处理器独立完成。在同一时间点,任务一定是同时运行。
- 当程序中写下多进程或多线程代码时，这意味着的是并发而不是并行， 并行与否程序员无法控制，只能让操作系统决定。 一定吗？
