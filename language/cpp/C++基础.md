# C++ 基础

- [C++ 基础](#c-基础)
    - [前向声明](#前向声明)
    - [getline](#getline)
    - [srand](#srand)
    - [map](#map)
    - [new operator和operator new](#new-operator和operator-new)
    - [placement new](#placement-new)
    - [模板的分离式编译](#模板的分离式编译)
    - [内联函数](#内联函数)
    - [结构体对齐规则](#结构体对齐规则)
    - [typedef](#typedef)
    - [volatile](#volatile)
    - [无锁编程](#无锁编程)
    - [抽象类方法](#抽象类方法)
    - [类型转换](#类型转换)
    - [enum VS enum class](#enum-vs-enum-class)
    - [回调函数](#回调函数)
    - [线程安全注解（clang独有）](#线程安全注解clang独有)
    - [memset](#memset)
    - [memcpy](#memcpy)
    - [vector](#vector)
    - [print](#print)
    - [常用宏定义](#常用宏定义)
    - [std::bitset](#stdbitset)
    - [do while](#do-while)
    - [虚析构](#虚析构)
    - [构造一个只能在堆或者栈上生成的对象](#构造一个只能在堆或者栈上生成的对象)
    - [任何一个含virtual函数的基类，都应该申明为virtual 析构函数](#任何一个含virtual函数的基类都应该申明为virtual-析构函数)
    - [非虚基类不应该考虑继承，而应该考虑组合](#非虚基类不应该考虑继承而应该考虑组合)
    - [\__attribute_\_](#_attribute_)
    - [访问权限](#访问权限)
    - [性能分析工具callgrind，gperftools](#性能分析工具callgrindgperftools)
    - [文件和流](#文件和流)

### 前向声明
-  前向声明的结构体只能定义为指针，否则会报错 field '*' has incomplete type（Linux）,前向声明时不能定义对象(申请内存)和使用对象的方法
-  前向声明能解决循环include的问题， 在对应cpp文件中需要实例化时，可再include，避免在头文件include后引起循环include的问题

### getline
- 从管道中读取数据，在头文件<istream>中，是istream类的成员函数。  istream& getline (char* s, streamsize n );
- 在头文件<string>中，是普通函数。istream& getline (istream&& is, string& str);

### srand
- 用来设置rand()产生随机数时的随机数种子

### map
- insert方法会忽略重复key，而不是替换pair

### new operator和operator new
-  new operator就是new操作符，不能被重载， 在C++中常用的创建对象操作符， 实际上执行如下3个过程
   -  调用operator new分配内存，operator new (sizeof(A))
   -  调用构造函数生成类对象
   -  返回相应指针
-  operator new 是函数，分为三种形式
   -  void* operator new (std::size_t size) throw (std::bad_alloc);   A* a = new A; //调用第一种
   -  void* operator new (std::size_t size, const std::nothrow_t& nothrow_constant) throw(); A* a = new(std::nothrow) A; //调用第二种
   -  void* operator new (std::size_t size, void* ptr) throw();  new (p)A(); //调用第三种 ，调用placement new之后，还会在p上调用A::A()，这里的p可以是堆中动态分配的内存，也可以是栈中缓冲
   -   第一、第二个版本可以被用户重载，且都分配内存，定义自己的版本，第三种placement new不可重载。 第三种可为内存池

###  placement new
- 它本质上是对operator new的重载，定义于#include <new>中

### 模板的分离式编译
- C++模板大多时候不使用分离式编译,所以一般都是声明定义一起放入头文件中，但模板其实能够分离式编译
  - 当编译器只看到模板的声明的时候，他不能实例化该模板，只能创建一个具有外部连接的符号并期待连接器能够将符号的地址查找出来
  - 然而当实现该模板的.cpp文件中没有用到模板的实例时，编译器懒得去实例化，所以，整个工程的.obj中就找不到一行模板实例的二进制代码，就会出现报错

### 内联函数
- 目的是为了提高函数的执行效率,会造成代码膨胀
- 与宏定义相比的优点：
  -  内联可调试
  -  可进行类型安全检查或自动类型转换
  -  可访问成员变量

### 结构体对齐规则
- 所有数据成员的偏移量必须是自身长度的整数倍，若不满足编译器会在成员之间加填充字节（0是所有长度的整数倍）
  - 变量存放的起始地址相对于结构的起始地址的偏移量必须为当前数据结构的倍数
- 结构体的总大小为结构体最宽基本类型成员大小的整数倍，如有需要编译器会在最末一个成员之后加上填充字节
  - 如结构体大小为8bytes,  alignment 为8, 则对应的块数为1， 对齐后的字节数为8bytes；如果大小为10 bytes,alignment 为8, 则对应的块数为2， 对齐后的字节数为16bytes

### typedef
- 作用域和函数变量类似：在类中定义的typedef就必须使用class_name::, 且受到private的权限限制（经过测试在类中受定义位置的影响，在前面的无法使用后面定义的typedef）
- 在typedef也具有继承性，子类可用使用父类定义的别名

### volatile
- 声明某个变量的值是随时可能被改变的，每次读取次变量都从内存地址中直接读取
- 为了防止编译器的优化而从寄存器中读取数据，而导致多线程时数据不一致
- 但是volatile仅仅是针对编译器的，对CPU无影响，因此再多核环境下没有任何作用 Really???

### 无锁编程
- 其核心为CAS，即compare & swap, 原子操作，保证原子性
- std::atomic类模板的成员函数compare_exchange_strong
- obj.compare_exchange_strong(expected &, desired&, memory_order)
- obj.compare_exchange_weak(expected &, desired&, memory_order)
- 伪代码如下：
```C++
    if (*obj = expected) { //其他线程未修改obj的值，直接将obj的值修改为新的值desired
          *obj = desired;
          return ture;
    } else { //其他线程修改了obj的值，所以修改expeted的值，下次将obj的值修改为新的值desired
          expected = *obj; 
          return false;
    }
```
### 抽象类方法
- 抽象类(含有纯虚函数(func_A)的类)中的成员函数(func_B)可调用纯虚函数(func_A)方法；由于抽象类不能实例化,所以成员函数必定由子类对象调用，而子类必须实现纯虚函数接口

### 类型转换
- const_cast: 去掉符合类型种的const或者volatile属性,只能用在指针和引用上面(特有)，将const指针转为非const指针， 将const引用转为非const引用
- static_cast:(无法处理const)可以用在指针引用(类似dynamic_cast)，基础数据和对象(独一无二)
- dynamic_cast:需要继承关系基于具有多态性质的类的对象的指针和引用转换，普通类的指针和引用会转换失败
- reinterpret_cast：运算符是用来处理无关类型之间的转换；它会产生一个新的值，这个值会有与原始参数（expressoin）有完全相同的比特位；错误的使用reinterpret_cast很容易导致程序的不安全，只有将转换后的类型值转换回到其原始类型，这样才是正确使用reinterpret_cast方式

### enum VS enum class
- enum 枚举类型: 不是类型安全的，被视为整数，不同的枚举类型间可比较。 枚举的全名称暴露在一般范围中，即不能由相同的枚举名
-  enum class 强枚举类型(C++11)：类型安全，不被视作整数，不可与整数值比较，限定了作用域指定enum的大小优化内存

### 回调函数
- C++的回调函数其实为函数指针的使用
- 函数指针：相同的函数返回类型，参数类型，参数个数。函数名不关心
- 虚函数：相同的函数返回类型，参数类型，参数个数，函数名
- 重载：不同的参数个数或参数类型，相同的函数名。返回类型不关心
- 隐藏：父类子类函数不能重载，只要函数名相同，子类就会隐藏父类同名函数，可显示调用父类函数

### 线程安全注解（clang独有）
- GUARDED_BY(c)： 是一个应用在数据成员上的属性，它声明了数据成员被给定的监护权保护。对于数据的读操作需要共享的访问权限，而写操作需要独占的访问权限
- PT_GUARDED_BY：与之类似，只不过它是为指针和智能指针准备的。对数据成员（指针）本身没有任何限制，它保护的是指针指向的数据
- EXCLUDES：EXCLUDES是函数或方法的属性，该属性声明调用方不拥有给定的功能。该注释用于防止死锁。许多互斥量实现都不是可重入的，因此，如果函数第二次获取互斥量，则可能发生死锁

### memset
- void *memset(void *s, int v, size_t n)
- 用它对一片内存空间逐字节进行初始化
- 由于是逐字节进行初始化，所以，在memset使用时要千万小心，在给char以外的数组赋值时，只能初始化为0或者-1

### memcpy
- void *memcpy(void*dest, const void *src, size_t n)
- 由src指向地址为起始地址的连续n个字节的数据复制到以destin指向地址为起始地址的空间内
- dest和src都不一定是数组，任意的可读写的空间均可

### vector
- emplace_back: 调用构造函数,
  - 在实现时，则是直接在容器尾部创建这个元素，省去了拷贝或移动元素的过程
  - emplace_pack仅在通过 使用构造参数传入的时候更高效
- push_back: 调用构造函数， 调用移动构造函数（优先）/ 拷贝构造函数

### print
- int printf(const char *format, ...); //输出到标准输出
- int fprintf(FILE *stream, const char *format, ...); //输出到文件
- int sprintf(char *str, const char *format, ...); //输出到字符串str中
- int snprintf(char *str, size_t size, const char *format, ...); //按size大小输出到字符串str中
- int printf(const char *format, ...);//线程安全

### 常用宏定义
- \__FILE__，\__LINE__,\__FUNCTION__: 表示文件名、行数和函数名，用于程序运行期异常的跟踪
- \__DATE__,\__TIME__: 用于得到最后一次编译的日期和时间
- \__TIMESTAMP__: 和__TIME__的格式相同。同于得到本文件最后一次被修改的时间
- \__GNUC__、\__GNUC_MINOR__、\__GNUC_MINOR__、\__GNUC_PATCHLEVEL__:  用于得到GNU版本
- \__COUNTER__: 自身计数器，用于记录以前编译过程中出现的__COUNTER__的次数，从0开始计数。常用于构造一系列的变量名称，函数名称等
- C、C++宏体中出现的#，#@，##:
  - #的功能是将其后面的宏参数进行字符串化操作
  - ##被称为连接符（concatenator），用来将两个Token连接为一个Token

### std::bitset
```C++
  std::bitset<16> foo;    // 0000000000000000
  std::bitset<16> bar (0xfa2);  // 0000111110100010
  std::bitset<16> baz (std::string("0101111001")); // 0000000101111001
```
- bitset::test: 返回pos位置的元素是否被设置，或者是否为1。返回值为true，或false
- bitset::any: 判断是否任何一个元素被设置，或者判断是否至少有一个元素为1
- bitset::none: 判断一个bitset是否没被set。如果一个bitset中有元素为1，则返回false，否则返回true
- string对象和bitset对象之间是反向转化的: string对象的最右边字符（即下标最大的那个字符）用来初始化bitset对象的低阶位（即下标为0的位）

### do while
```C++
do{
}while(0) 
```
-  为了宏展开的时候不会出错
-  使用break，达到使用类似goto的作用

### 虚析构
-  普通继承析构顺序：先父再子，与构造相反
-  使用父类指针或引用的时候， 析构为非虚函数时，只会调用父类的析构，不会调用子类的析构，可能导致内存泄漏

### 构造一个只能在堆或者栈上生成的对象
- 只能在堆上，创建静态函数返回指针 = new A();
-  只能在栈上，创建静态函数返回对象=A() 或者屏蔽new操作符，即设置为私有

### 任何一个含virtual函数的基类，都应该申明为virtual 析构函数
- 虚函数表指针随对象走，它发生在对象运行期，当对象创建的时候，虚函数表表指针位于该对象所在内存的最前面。 使用虚函数时，虚函数表指针指向虚函数表中的函数地址即可实现多态
- 虚函数表是在编译期间就已经确定，且虚函数表存放虚函数的地址也是在创建时被确定
- 虚函数表属于类，类的所有对象共享这个类的虚函数表
- 虚函数表由编译器在编译时生成，保存在exe的(常量区).rdata只读数据段

### 非虚基类不应该考虑继承，而应该考虑组合
- 原因是非虚基类继承时，无法使用父类指针析构子类对象

### \__attribute__
- 关键字主要是用来在函数或数据声明中设置其属性。给函数赋给属性的主要目的在于让编译器进行优化
- \__attribute__((unused)   此属性附加到函数上，表示该函数可能未被使用。 GCC不会对此功能发出警告
- \__attribute__ ((noreturn))  这个属性告诉编译器函数不会返回。当遇到函数需要返回值却还没运行到返回值处就已退出来的情况 GCC不会对此功能发出警告
- \__attribute__((cold))   这样在分支预测机制里就不会对该函数进行预取，或说是将它和其他同样冷门(cold)的函数放到一块，这样它就很可能不会被放到缓存中来，而让更热门的指令放到缓存中
- \__attribute__((hot))   函数前面使用这个扩展，表示该函数会被经常调用到，在编译链接时要对其优化，或说是将它和其他同样热(hot)的函数放到一块，这样有利于缓存的存取
- \__attribute__((packed))   的作用就是告诉编译器，取消结构在编译过程中的优化对齐，按照实际占用字节数进行对齐，是GCC特有的语法
- \__attribute__((weak))  我们不确定外部模块是否提供一个函数func，但是我们不得不用这个函数。将本模块的func转成弱符号类型，如果遇到强符号类型（即外部模块定义了func），那么我们在本模块执行的func将会是外部模块定义的func。类似于c++中的重载 weak属性只会在静态库(.o .a )中生效，动态库(.so)中不会生效

### 访问权限
- 派生类的成员可以直接访问基类的保护成员，但不能直接访问基类的私有成员

### 性能分析工具callgrind，gperftools
- callgrind是valgrind工具套件中用于分析程序性能的一个工具
- gperftools是Google提供的一套工具

### 文件和流
- ifstream 输入文件流，用于读取文件，将文件内容输入到内存的ifstream中，遇到空格或者换行停止，可以使用read(char *, int size)函数
- ofstream 输出文件流，用于写文件，将文件流输出到文件中