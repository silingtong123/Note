## 数据类型

### 数组和切片
- 1个数组变量即表示整个数组，它并不是隐式的指向第一个元素的指针,大数组在值传递时有较大开销，建议传递数组指针，但数组指针不是数组，却可以按照数组的方法使用
-  长度为 0 的数组在内存中并不占用空间。空数组虽然很少直接使用，但是可以用于强调某种特有类型的操作时避免分配额外的内存空间，比如用于管道的同步操作
- 和数组不同的是，字符串的元素不可修改，是一个只读的字节数组.字符串结构由两个信息组成：第一个是字符串指向的底层字节数组，第二个是字符串的字节的长度。字符串其实是一个结构体，因此字符串的赋值操作也就是 reflect.StringHeader 结构体的复制过程，<font color="#00dd00">**并不会涉及底层字节数组的复制**</font>，可以将字符串数组看作一个结构体数组
- <font color="#00dd00">****</font>
- 动态数组，即为切片：容量必须大于或等于切片的长度，对于类型，和数组的最大不同是，切片的类型和长度信息无关，只要是相同类型元素构成的切片均对应相同的切片类型
  - 在容量不足的情况下，append 的操作会导致重新分配内存，可能导致巨大的内存分配和复制数据代价。即使容量足够，依然需要用 append 函数的返回值来更新切片本身，因为新切片的长度已经发生了变化
  - 头部追加元素，在开头一般都会导致内存的重新分配，而且会导致已有的元素全部复制 1 次。因此，从切片的开头添加元素的性能一般要比从尾部追加元素的性能差很多
  - 根据要删除元素的位置有三种情况：从开头位置删除，从中间位置删除，从尾部删除。其中删除切片尾部的元素最快
  - 切片高效操作的要点是要降低内存分配的次数，尽量保证 append 操作不会超出 cap 的容量，降低触发内存分配的次数和每次分配内存大小
  - 切片操作并不会复制底层的数据。底层的数组会被保存在内存中，直到它不再被引用。但是有时候可能会因为一个小的内存引用而导致底层整个数组处于被使用的状态，这会延迟自动内存回收器对底层数组的回收
- 
```golang
var (
    a []int               // nil 切片, 和 nil 相等, 一般用来表示一个不存在的切片
    b = []int{}           // 空切片, 和 nil 不相等, 一般用来表示一个空的集合
    c = []int{1, 2, 3}    // 有 3 个元素的切片, len 和 cap 都为 3
    d = c[:2]             // 有 2 个元素的切片, len 为 2, cap 为 3
    e = c[0:2:cap(c)]     // 有 2 个元素的切片, len 为 2, cap 为 3
    f = c[:0]             // 有 0 个元素的切片, len 为 0, cap 为 3
    g = make([]int, 3)    // 有 3 个元素的切片, len 和 cap 都为 3
    h = make([]int, 2, 3) // 有 2 个元素的切片, len 为 2, cap 为 3
    i = make([]int, 0, 3) // 有 0 个元素的切片, len 为 0, cap 为 3
)

var a []int
a = append(a, 1)               // 追加 1 个元素
a = append(a, 1, 2, 3)         // 追加多个元素, 手写解包方式
a = append(a, []int{1,2,3}...) // 追加 1 个切片, 切片需要解包

var a = []int{1,2,3}
a = append([]int{0}, a...)        // 在开头添加 1 个元素
a = append([]int{-3,-2,-1}, a...) // 在开头添加 1 个切片
```


### map
- make(map[string]int)  map是make创建的数据结构的引用，必须先创建实例，才能使用
```golang
 var map1  make(map[string]int)
 map1["test"] = 1 // 报错，不能正常添加元素，map必须先初始化

 map1 = make(map[string]int)
 map1["test"] = 1 //已初始化，正确添加元素

 delete(map1, "test") //删除元素
```

### channel
- 无缓冲 Channel（同步管道）： 发送操作后会阻塞，直到管道被另一个携程读取。(channel为空时，接受携程阻塞)，发送携程的阻塞状态才会变化
- 缓冲管道： 对于 Channel 的第 K 个接收完成操作发生在第 K+C 个发送操作完成之前，其中 C 是 Channel 的缓存大小
  - 对于这种要等待 N 个线程完成后再进行下一步的同步操作有一个简单的做法，除了使用带缓冲的通道，就是使用 sync.WaitGroup
  - buffer没满时，写入携程不会阻塞，buffer没空时接收携程不会阻塞

- 匿名管道：我们并不关心管道中传输数据的真实类型，其中管道接收和发送操作只是用于消息的同步, 一般更倾向于用无类型的匿名结构体代替
  
```golang
    ch :=make(chan  int) //只写管道声明
    ch <- x //写入管道
    x = <-ch //读取管道数据，也可以使用<-ch 读取管道但是丢弃数据
    x, ok = <-ch //ok为ture时表示正常接受数据，否则接受操作在一个关闭并且读完的通道上
    for x : =range ch{//也可使用循环读取管道数据，ok为false时会自动结束循环，即管道不关闭会死循环

    }
    close(ch)// 关闭管道

    c0 := make(chan int) //无缓冲管道声明
    c1 := make(chan int, size) //缓冲管道声明
    c2 := make(chan struct{}) //匿名管道声明
    //管道作为形参时，会被有意限制为只能接受或者发送，可读写管道能被隐式转换为只能接受或者发送管道
    make(<-chan int) //只读管道声明
    make(chan<- int) //只写管道声明
    go func() {
        fmt.Println("c2")
        c2 <- struct{}{} // struct{} 部分是类型, {} 表示对应的结构体值
    }()
    <-c2

    tick := time.Tick(1 * time.Second) //计数器通道

    select {} //永远等待
```
### 函数，变量声明
- 可变数量的参数必须是最后出现的参数，可变数量的参数其实是一个切片类型的参数
- 可在声明时初始化，未初始化时会被隐式初始化，string会初始化为空串，数字初始化为0
- 短变量声明： i:=0
- <font color="#00dd00">**interface{} 为参数时，表示可以接受任意数据类型的参数**</font>
- 
```golang
    type ( //同时给多种类型起别名
        subscriber chan interface{}
        topicFunc  func(v interface{}) bool
    )

    var str1, str2 string //变量声明

    func (p * StructA )Log(arg1, arg2, more ...) errorcode {
    }
    //StructA 表示Log为成员函数，errorcode为返回值
```

### 语法
- 只支持前缀 i++, ++i为不合法， j=i++同样不合法
- 空标识符 "_"  ：程序逻辑不需要，但语法需要该变量名
- 字符串的 "+=":通过给旧内容追加新的部分，会生成新的字符串， 旧字符串将不再使用，会被例行垃圾回收。 如果 有大量数据需要处理，这样代价比较大。<font color="#00dd00">可换为使用string中的jion方法</font><br />
- for, if用法
```golang
   for k, v : = range map1 {}//获取k,v

   for k:= range map1 {} //获取k

   for -, v:= range map1 {} //获取 v
   
   for i:= range 5 {} // i= 0,1,2,3,4

   if n > 1 {}

   for i:= range chan1 {} //直接range只会读取值，缓冲区无数据会读取默认零值，chan1被关闭，则退出range

   //T * t = new (T)
   wg sync.WaitGroup //主携程
   wg.Add(1) //worker携程
   wg.Down() //每个worker携程完成就调用，一般用defer调用
```
- err ==nil时为正常情况, error 的值描述错误原因
- 继承
  - 子类结构体可以通过对父类结构体以组合的方式匿名继承，即可继承父类的方法
  - 通过对接口类似的以组合的方式匿名继承，重写该方法即可
- select：通过select可以监听channel上的数据流动。当 select 有多个分支时，会随机选择一个可用的管道分支，如果没有可用的管道分支则选择 default 分支，否则会一直保存阻塞状态
  -  Go 语言并没有提供在一个直接终止 Goroutine 的方法，这样会导致 Goroutine 之间的共享变量处在未定义的状态上 
  -  如果没有加入默认分支，那么一旦所有的case表达式都没有满足求值条件，那么select语句就会被阻塞。直到至少有一个case表达式满足条件为止，如果select语句发现同时有多个候选分支满足选择条件，那么它就会用一种伪随机的算法在这些分支中选择一个并执行
  -   但是管道的发送操作和接收操作是一一对应的，如果要停止多个 Goroutine 那么可能需要创建同样数量的管道，这个代价太大了。其实我们可以通过 close 关闭一个管道来实现广播的效果，所有从关闭管道接收的操作均会收到一个零值和一个可选的失败标志
  -  switch：类似C++ switch 但是没有不需要break
- context :cancel() 来通知后台 Goroutine 退出，这样就避免了 Goroutine 的泄漏
```golang
ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)

//其他携程监控通道
select{
    case  ctx.Done()
}

//不同携程的ctx的Done调用顺序确定吗？
```
- recover:  recover：捕获异常，将异常转换为错误，常在defer中使用，必须在 defer 函数中直接调用 recover。如果 defer 中调用的是 recover 函数的包装函数的话，异常的捕获工作将失败，必须要和有异常的栈帧只隔一个栈帧，recover 函数才能正常捕获异常。换言之，recover 函数捕获的是祖父一级调用函数栈帧的异常
```golang
func foo() (err error) {
    defer func() {
        if r := recover(); r != nil {
            //...
        }
    }()

    panic("TODO")
}
```

