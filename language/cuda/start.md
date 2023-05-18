- 典型的cuda编程模式我们已经熟知了
  - 将输入数据从host转移到device
  - 在device上执行kernel
  - 将结果从device上转移回host
  - 线性内存通常使用 cudaMalloc() 分配并使用 cudaFree() 释放，主机内存和设备内存之间的数据传输通常使用 cudaMemcpy() 完成；申请2D或3D数组时cudaMallocPitch() 和 cudaMalloc3D()性能更佳
  - 流的创建和销毁cudaStreamCreate，cudaStreamDestroy()
- 所有的cuda操作（包括kernel执行和数据传输）都显式或隐式的运行在stream中，stream也就两种类型:隐式声明stream（NULL stream）, 显示声明stream（non-NULL stream）

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
- 线程，线程块，grid 都按列储存，所以对于大小为(Dx, Dy)的二维块，索引为(x, y)的线程的线程ID为(x + y*Dx)； 对于大小为 (Dx, Dy, Dz) 的三维块，索引为 (x, y, z) 的线程的线程 ID 为 (x + y*Dx + z*Dx*Dy)
- 网格中的每个块都可以由一个一维、二维或三维的唯一索引标识，通过blockIdx变量访问
- 网格的维度可以通过内置的gridDim变量在内核中访问，线程块的维度可以通过内置的blockDim变量在内核中访问
- 线程块的个数记为网格大小grid_size;每个线程块中含有同样数目的线程，该数目称为线程块大小block_size，所以核函数中的总的线程就等与网格大小乘以线程块大小，即<<<grid_size，block_size>>>。
- int tid = threadIdx.z * blockDim.x * blockDim.y +threadIdx.y * blockDim.x + threadIdx.x;
  - index = blockDim.x * blockIdx.x + threadIdx.x: blockIdx.x表示第几个块，表示块的大小，一般为32的倍数，threadIdx.x表示第几个线程
  - stride = blockDim.x * gridDim.x
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