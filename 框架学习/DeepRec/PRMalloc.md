### **DeepRec PRMalloc**

- AllocStats： 记录每次申请内存时的状态：begin,end,size
- AllocBlock:  
  - 主要成员std::vector<AllocStats*> stats_;
  - CanInsert(AllocStats* alloc_stats) -> 插入的stat和已有的stats_成员无交集
- VirtualAllocBlock：
  - 主要成员AllocBlock* internal_block_;
  - 该block的所有者为为外界所有，本类只是持有其地址（即指针）

```C++
class LifetimeBin{
  ...
  std::vector<AllocStats*> stats_;
  std::vector<AllocBlock*> blocks_;
  std::vector<VirtualAllocBlock*> virtual_blocks_;  
}

```

- TrackAllocate() -> max_alignment_= std::max<int64_t>(max_alignment_, alignment); 修改max_alignment_ 
- TrackDeallocate() -> stats_.emplace_back(stats); 将记录的stats收集起来放入stats_
- FindBlock() -> 遍历blocks_，返回第一个满足block->CanInsert(stats)== true 的block
- BestFit(LifetimePolicy* policy)  -> 遍历stats_并调用FindBlock，若找到block，则调用block->insert, 将该stat插入AllocBlock的stats_成员； 若未找到block，则调用LifetimePolicy->FindBlock, 若找到则插入stat，并根据找到的block创建新的VirtualAllocBlock， push到virtual_blocks_中; 若仍未找到，则创建AllocBlock，并在该block中插入stat，并将block加入blocks_

- - -

```C++
class LifetimePolicy{
  ...
  std::vector<LifetimeBin*> bins_;
  std::map<size_t, LifetimeBin*> large_bins_;
  mutable spin_lock large_bin_lock_;  
}

```

- TrackDeallocate() -> GetBin(index)->TrackDeallocate(alloc_stats);找到对应bin调用TrackDeallocate
- TrackAllocate() ->GetBin(index)->TrackAllocate(alignment); 找到对应bin调用TrackAllocate
- FindBlock(AllocStats* stats, size_t bindex) ->
  - 遍历bins_， 对每个bin调用FindBlock
  - 遍历large_bins_，  对每个bin调用FindBlock

- BestFit()
  - 反向遍历large_bins_， 对每个bin调用BestFit, 传递this指针
  - 反向遍历large_bins_， 对每个bin调用BestFit, 传递this指针

- LifetimePolicy() -> 构造函数， 初始化bins_，size = large_bin_index_
                                bins_[0] = 36KB， bins_[1] = 340KB bins_[i] = 32KB + ( i+1)*4KB
- GetBin(size_t index) -> 当index < large_bin_index_时，返回bins_[index] ,当index > large_bin_index_时，在large_bins_查找，找到则返回，否则创建LifetimeBin，push到large_bins_中

- - -

```C++
class MemoryPlanner {
  ...
  std::vector<LifetimePolicy*> lifetime_stats_polices_;
  TensorPoolAllocator* allocator_;
  thread::ThreadPool* thread_pool_;
}

```

- MemoryPlanner()  --> InitPolicy(),InitStepInfo(), is_stats_(false)
- InitPolicy() --> lifetime_stats_polices_添加3个元素
  - new LifetimePolicy(_4KB,_4KB_OFFSET, _32KB));
  - new LifetimePolicy(_8KB,_8KB_OFFSET, _32KB));
  - new LifetimePolicy(_16KB,_16KB_OFFSET, _32KB));
- InitStepInfo() -->获取START_STATISTIC_STEP，  STOP_STATISTIC_STEP
- StartCollect() 开始收集， 其实也做了停止收集的工作 ->CollectDone()
  - [START_STATISTIC_STEP，  STOP_STATISTIC_STEP) is_stats_ = true.
  - [STOP_STATISTIC_STEP, ]  is_stats_ = false， 并调用CollectDone
- CollectDone() -> Schedule（） 传递了一个lamda函数, 函数中调用allocator_->Init();
- Schedule() -> thread_pool_->Schedule()
- BestLifetimePolicy() -> 遍历lifetime_stats_polices_并调用BestFit(), 选择policy->TotalMem()最小的policy返回
- StopCollect () 停止收集，函数内部代码为空。
- SetAllocator() 设置TensorPoolAllocator成员
- SetThreadPool() 设置ThreadPool成员
- Cleanup() 调用lifetime_stats_polices_中所有元素的clear_up() -> 遍历lifetimepolicy中的bins_  和large_bins_元素的clear_up()--->清理所有blocks_，virtual_blocks_
- TrackAllocate() :is_stats_==true时，遍历lifetime_stats_polices_，对每个元素调用TrackAllocate()， is_stats_= false时直接返回
- TrackDeallocate()::is_stats_==true时，遍历lifetime_stats_polices_，对每个元素调用TrackDeallocate()，  is_stats_= false时直接返回

- - -

```C++
//RAII特性：
class ScopedMemoryCollector {
  ScopedMemoryCollector(){
  MemoryPlannerFactory::GetMemoryPlanner()->StartCollect();
  }

  ~ScopedMemoryCollector(){
    MemoryPlannerFactory::GetMemoryPlanner()->StopCollect();
  }
}
```

MemoryPlannerFactory::GetMemoryPlanner()为单例模式，能够根据参数得到MemoryPlanner或NullableMemoryPlanner

- - -

```C++
class TensorPoolAllocator {
  ...
  std::unique_ptr<SubAllocator> sub_allocator_;
  MemoryPlannerBase* mem_planner_;
  size_t large_bin_index_;
  std::vector<Bin*> lifetime_bins_;
  std::map<size_t, Bin*> large_lifetime_bins_;

  TensorPoolAllocator():sub_allocator_(new DefaultCPUSubAllocator),
      mem_planner_(MemoryPlannerFactory::GetMemoryPlanner()),  
  {
   .....
   mem_planner_->SetAllocator(this);
  }

  init(){
    .....
    auto  lifetime_policy = mem_planner_->BestLifetimePolicy();
  }

  AllocateRaw(size_t alignment, size_t num_bytes)(){
    ......
    SmallAlloc();
    ...
    BigAllocate(alignment, num_bytes);
    ....
    BigAllocateStatistic(alignment, num_bytes);
  }
  SmallAlloc(){
    .....
    sub_allocator_->Alloc(alignment, total) 
  }
  BigAllocate(alignment, num_bytes){
    return SetDefaultHeader();
  }
  BigAllocateStatistic(alignment, num_bytes) {
    return SetDefaultHeader();
  }

  DeallocateRaw(){
    ...
    BigDeallocate();
    ...
  }
}


```

- BestLifetimePolicy: 遍历lifetime_policy->GetLargeBins(); 创建元素Bin, push进入large_lifetime_bins_, 遍历lifetime_policy->GetBins(); 创建元素Bin, push进入lifetime_bins_
- BigAllocate:
  - inited_==false时，调用mem_planner_->TrackAllocate(alignment, total);且后续都调用sub_allocator_->Alloc(alignment, total)
  - inited_== true时，能得到Bin就 调用Bin->Allocate, ***init()后inited_== true，mem_planner_->CollectDone会调用allocator_->Init()***;
- BigAllocateStatistic: 同上，只是额外添加了一些统计信息null_bin_counter_，hit_counter_，missed_counter_
- SetDefaultHeader: 使用了placement new， 返回rar_ptr+sizeof(head_size)为起点的user_ptr
- BigDeallocate:
  - 如果inited_==false, mem_planner_->TrackDeallocate(header); 接着调用sub_allocator_->Free(ptr, num_bytes);
  - 如果inited_==true且，header->bin不为空,调用Bin->Deallocate 

- - -

```C++
class Bin{
  ... 
  Buffer buffer_;
  VirtualBuffer virtual_buffer_;
  SubAllocator* sub_allocator_;

  Allocate(size_t total, size_t header_size) {
    ...
    buffer_.Allocate();
  }

  Deallocate(){
    ...
    buffer_.Deallocate(header->raw_ptr); 
  } 

  //初始化buffer_，virtual_buffer_，sub_allocator_
  Allocate() {...} 
}

class Buffer {
  ...
  std::stack<void*> buffer_;
  void* begin_;
  void* end_;

  //申请内存空间并且将指针保存下来
  Buffer(){...}
  
  //不真正申请，而是使用以前申请好保存下来的
  Allocate(){
    ...
     ptr = buffer_.top(); 
  }

  //不真正释放，而是存放指针
  Deallocate(){
    ...
    buffer_.emplace(p);
  }
}

```

```C++
class {
  static const int64 kFullExtent;

  // TODO(yangke): switch to Eigen once it supports variable size arrays.
  // A value of
  gtl::InlinedVector<int64, 4> starts_;
  gtl::InlinedVector<int64, 4> lengths_;
}
```