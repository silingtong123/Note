- 典型的cuda执行流程如下：
  - 将输入数据从host转移到device
  - 在device上执行kernel
  - 将结果从device上转移回host
  - 释放device和host上分配的内存
- CUDA中内存的申请和释放：
  - 线性内存通常使用 cudaMalloc() 分配并使用 cudaFree() 释放，主机内存和设备内存之间的数据传输通常使用 cudaMemcpy() 完成；申请2D或3D数组时cudaMallocPitch() 和 cudaMalloc3D()性能更佳
  - 流的创建和销毁cudaStreamCreate，cudaStreamDestroy()
- 函数类型：
  - \__global__: 在device上执行，从host中调用（一些特定的GPU也可以从device上调用），返回类型必须是void，不支持可变参数参数，不能成为类成员函数。注意用__global__定义的kernel是异步的，这意味着host不会等待kernel执行完就执行下一步
  - \__device__：在device上执行，单仅可以从device中调用，不可以和__global__同时用
  - \__host__：在host上执行，仅可以从host上调用，一般省略不写，不可以和__global__同时用，但可和__device__，此时函数会在device和host都编译
- 所有的cuda操作（包括kernel执行和数据传输）都显式或隐式的运行在stream中，stream也就两种类型:隐式声明stream（NULL stream）, 显示声明stream（non-NULL stream）

### CUDA结构

- 流失多处理器（SM，Streaming Multiprocessor）：SM : 线程块=1:N, 一个kernel的各个线程块被分配多个SM,grid是逻辑层，SM是物理层
- SM是SIMT(单指令多线程，执行单元是线程束，一个线程束有32个线程)：一个SM接受一个线程块时，会放入多个线程束（所以block的大小是32的倍数），但是一个SM同时并发的线程束数时有限的
```c++
dim3 grid(3, 2); // grid中有3*2个block，排列顺序按列存储
dim3 block(5, 3); // block中有5* 3个线程，排列顺序按列存储
// dim3 grid(3, 4, 2);
// dim3 block(5, 3, 2);
kernel_fun<<< grid, block >>>(prams...);

```
- dim3： 包含3个无符号数，在定义时缺省值初始化为1， 所以grid和block可以灵活定义为1-dim,2-dim,3-dim，执行配置中两个参数分别为grid和block
- 线程坐标thead（blockIdx, thredIdx）唯一标识, 而blockIdx.x =1, blockIdx.y=1, threadIdx.x=1, threadIdx.y=1,表示时为grid中下表为(1,1)的block中下标为(1,1)的线程
- 线程，block，grid 都按列储存，所以对于大小为(Dx, Dy)的block，索引为(x, y)的线程的线程ID为(x + y*Dx)； 对于大小为 (Dx, Dy, Dz) 的block，索引为 (x, y, z) 的线程的线程 ID 为 (x + y*Dx + z*Dx*Dy)
- grid中的每个block都可以由一个一维、二维或三维的唯一索引标识，通过blockIdx变量访问
- grid的维度可以通过内置的gridDim变量在内核中访问，block的维度可以通过内置的blockDim变量在内核中访问,都是dim3参数
- blocks的个数记为网格大小grid_size;每个线程块中含有同样数目的线程，该数目称为线程块大小block_size，所以核函数中的总的线程就等与网格大小乘以线程块大小，即<<<grid_size，block_size>>>。
- int tid = threadIdx.z * blockDim.x * blockDim.y +threadIdx.y * blockDim.x + threadIdx.x;
  - index = blockDim.x * blockIdx.x + threadIdx.x: blockIdx.x表示第几个块，blockDim.x表示块的大小，一般为32的倍数，threadIdx.x表示第几个线程
  - stride = blockDim.x * gridDim.x

### 样例

- cuda支持的编程语言：c/c++/python/fortran/java
- 添加环境变量，vim ~/.bashrc

```shell
export PATH="/usr/local/cuda-10.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH"
```
- nvcc -V
- 支持C++语言，直接编译 nvcc hello.cu, 运行./a.out
```C++
#include<stdio.h>
int main()
{

        printf("helloword\n");
        return 0;
}

```
- cuda 中的核函数与c++中的函数是类似的，cuda的核函数必须被限定词__global__修饰,核函数的返回类型必须是空类型，即void
- 

```C++
#include<stdio.h>

__global__ void hello_from_gpu()
{
   printf("hello word from the gpu!\n");
}

int main()
{

   hello_from_gpu<<<1,1>>>();// 三括号中的第一个数时线程块的个数，第二个数可以看作每个线程块中的线程数
   cudaDeviceSynchronize(); // 缓存区不会核会自动刷新, 执行同步操作时缓存区才会刷新
   printf("helloword\n");
   return 0;
}
  
```

- 核函数中的printf函数的使用方法和C++库中的printf函数的使用方法基本上是一样的，而在核函数中使用printf函数时也需要包含头文件<stdio.h>,核函数中不支持C++的iostream
```C++
#include<stdio.h>
__global__ void hello_from_gpu()
{
    //blockIdx 和 threadIdx 有x,y,z的三个成员
   const int bid = blockIdx.x; // 表示当前在第几个bolck上运行,该变量指定一个线程在一个网格中的线程块指标。其取值范围是从0到gridDim.x-1
   const int tid = threadIdx.x; //表示当前在第几个thread上运行, 该变量指定一个线程在一个线程块中的线程指标，其取值范围是从0到blockDim.x-1
   printf("hello word from block %d and thread %d\n",bid,tid);
   //分配释放内存
   cudaMallocManaged(&x, N*sizeof(float));
   cudaFree(x);
}

int main()
{
    //gridDim.x ：该变量的数值等与执行配置中变量grid_size的数值, 本例为2
    //blockDim.x: 表示每个block的大小,该变量的数值等与执行配置中变量block_size的数值,本例为4
   hello_from_gpu<<<2,4>>>();
   cudaDeviceSynchronize(); //CPU需要等待cuda上的代码运行完毕，才能对数据进行读取
   printf("helloword\n");
   return 0;
}

```