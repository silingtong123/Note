# DeepRec 源码学习

## DeepRec 内存管理
1. 如何管理数据的存储，特别是EV，EV针对不同的存储介质如何做出适配？
   - 首先会注册内存存在的一些allocator,具体调用如下：
      - REGISTER_MEM_ALLOCATOR -》REGISTER_MEM_ALLOCATOR_UNIQ_HELPER-》REGISTER_MEM_ALLOCATOR_UNIQ -》static AllocatorFactoryRegistration allocator_factory_reg_##ctr
      - AllocatorFactoryRegistration -》AllocatorFactoryRegistry::singleton()->Register   
   - 使用举例：
      - REGISTER_MEM_ALLOCATOR("TensorPoolAllocator", 300, TensorPoolAllocatorFactory);
      - REGISTER_MEM_ALLOCATOR("EVAllocator", 20, EVAllocatorFactory); 存储EV
2. EVAllocatorFactory 和 EVAllocator具体是什么？如何工作？
   - EVAllocatorFactory::CreateAllocator --> new EVAllocator 
     - EVAllocator::AllocateRawAlignedMalloc -> posix_memalign 使用linux底层接口申请内存
3. AllocatorStats  是什么？有什么作用？
  1. 分配器收集的运行时统计信息。与 stream_executor::AllocatorStats 完全相同，但独立定义以保持 StreamExecutor 和 TensorFlow 的相互独立性
  2. 数据成员：
    - num_allocs 分配内存的次数
    - bytes_in_use 分配的字节数
    - peak_bytes_in_use 分配的字节数峰值
    - largest_alloc_size 最大的单次申请大小
    - bytes_reserved 保留的字节数
    - peak_bytes_reserved 保留的字节数峰值
4. DBValuePtr 管理DB上EV的value
  1. DBValuePtr:: GetOrAllocate 若存在，则从level_db_->Get， 否则使用默认值，并调用level_db_->Put
  2. 调用commit也会调用level_db_->Put
  3. COMPUTE_FTRL --> commit, 还有一些op的compute函数也会调用该函数

## EV存储
### EmbeddingVar数据读写

```C++
class  EmbeddingVar{
  std::string name_;
  bool is_initialized_ = false;

  mutex mu_;

  V* default_value_;
  int64 value_len_;
  Allocator* alloc_;
  embedding::StorageManager<K, V>* storage_manager_;
  EmbeddingConfig emb_config_;
  EmbeddingFilter<K, V, EmbeddingVar<K, V>>* filter_;
  std::function<void(ValuePtr<V>*, int, int64)> add_freq_fn_;
  std::function<void(ValuePtr<V>*, int64)> update_version_fn_;
}

//isfilter
void LookupOrCreate(...) {
    ...
    filter_->LookupOrCreate()
    // ev_->LookupOrCreateKey 
    // ev_->LookupOrCreateEmb 
}

Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr) {
    ...
    storage_manager_->GetOrCreate
}

V* LookupOrCreateEmb(ValuePtr<V>* value_ptr, const V* default_v){
    ...
    value_ptr->GetOrAllocate
}
```

EV通过调用storage_manager_来进行数据的读写

```C++
class StorageManager {
  int32 hash_table_count_;
  std::string name_;
  std::vector<std::pair<KVInterface<K, V>*, Allocator*>> kvs_;
  std::vector<ValuePtr<V>*> value_ptr_out_of_date_;
  std::function<ValuePtr<V>*(Allocator*, size_t)> new_value_ptr_fn_;
  StorageConfig sc_;
  bool is_multi_level_;

  int64 alloc_len_;
  int64 total_dims_;

  std::unique_ptr<thread::ThreadPool> thread_pool_;
  Thread* eviction_thread_;
  BatchCache<K>* cache_;
  int64 cache_capacity_;
  mutex mu_;
  volatile bool shutdown_ GUARDED_BY(mu_) = false;

  volatile bool done_ = false;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
}
```

---

#### **写数据**


EmbeddingVar::Commit -->  storage_manager_->Commit -->kvs_[0].first -->Commit

   kvs_[0]为 LocklessHashMap，其commit函数为空，未实际实现，真正插入数据EmbeddingVar::LookupOrCreateKey--》storage_manager_->GetOrCreate -》kvs_[0]first->Insert

StorageManager中当储存介质大于2时，会存在一个LRU的cache_，eviction_thread_ 后台线程调用BatchEviction,将过期的ids从LocklessHashMap移除，在第二种介质(比如levelDB,SSD)中提交

BatchEviction -> kvs_[0].first->Remove(evic_ids[i])
                 kvs_[1].first->Commit(evic_ids[i], value_ptr) 
当kvs_[1]为levelDB时，kvs_[1].first->Commit会向levelDB中写数据，当kvs_[1]为SSD时，kvs_[1].first->Commit会向SSD中写数据

EmbeddingVar::BatchCommit --》 storage_manager_->BatchCommit --》 kv.first->BatchCommit(keys, value_ptrs)

- - - -
#### **读数据**

LookupOrCreateKey --》 storage_manager_->GetOrCreate  --》kvs_[level].first->Lookup  会去多个地方找，内存找不到就到第二种介质中找

---

### DRAM_SSDHASH

kv存储方式:
使用kv mmap方式将硬件映射带内存，使用mmap取消映射
```C++
class SSDHashKV : public KVInterface<K, V>
```

该方式会在后台启动一个EV_Eviction线程调用BatchEviction，定时清理过期的ids
```C++
    hash_table_count_ = kvs_.size();
    //只有DRAM_PMEM，DRAM_LEVELDB，DRAM_SSDHASH方式时，会满足下面if条件
    if (hash_table_count_ > 1) { 
      cache_ = new LRUCache<K>();
      eviction_thread_ = Env::Default()->StartThread(ThreadOptions(), "EV_Eviction",
                                                     [this]() { BatchEviction(); });
      thread_pool_.reset(new thread::ThreadPool(Env::Default(), ThreadOptions(),
                                               "MultiLevel_Embedding_Cache", 2,
                                               /*low_latency_hint=*/false));
    }
```

BatchEviction 中当cache_count大于cache_capacity_，会删除部分过期的ids
```C++
      int cache_count = cache_->size();
      if (cache_count > cache_capacity_) {
        // eviction
        int k_size = cache_count - cache_capacity_;
        k_size = std::min(k_size, EvictionSize); //EvictionSize默认10000
        size_t true_size = cache_->get_evic_ids(evic_ids, k_size);
        ValuePtr<V>* value_ptr;
        //驱除真正的value
        for (int64 i = 0; i < true_size; ++i) {
          if (kvs_[0].first->Lookup(evic_ids[i], &value_ptr).ok()) {
            TF_CHECK_OK(kvs_[1].first->Commit(evic_ids[i], value_ptr));
            TF_CHECK_OK(kvs_[0].first->Remove(evic_ids[i]));
            value_ptr_out_of_date_.emplace_back(value_ptr);
          } else {
            // bypass
          }
        }
      }
```
---
#### google/dense_hash_map

用法 [https://github.com/sparsehash/sparsehash](https://github.com/sparsehash/sparsehash)：
- 在插入数据之前，需要先调用set_empty_key()设置一个空Key，Key的值可以为任意符合类型的。但请注意之后插入的Key不能和空Key相同，否则会abort。这个空Key的目的是为了防止死循环，它需要这样一个标志来判断查找是否该结束了
- 在使用erase()时，前面必须调用set_deleted_key（）函数，set_deleted_key与set_empty_key的参数不应该相同。

**API**

sparse_hash_map、dense_hash_map、sparse_hash_set 和 dense_hash_set 的 API 是 SGI 的 hash_map 类 API 的超集。
有关 API 的更多信息，请参阅 doc/sparse_hash_map.html 等。

这些类的用法不同于 SGI 的 hash_map 和其他
hashtable的实现，主要有以下几种方式：
1. dense_hash_map 需要预留一个键值作为'空桶'值，通过set_empty_key()方法设置。 在使用 dense_hash_map 之前，*必须*调用此方法。 将任何元素插入其键等于空键的 dense_hash_map 是非法的。

2. 对于dense_hash_map和sparse_hash_map，如果你想从哈希表中删除元素，你必须预留一个键值作为'deleted bucket'值，通过set_deleted_key()方法设置。 如果您的哈希映射是仅插入的，则无需调用此方法。 如果调用 set_deleted_key()，则将任何元素插入键等于删除键的 dense_hash_map 或 sparse_hash_map 是非法的。

3. 这些哈希映射实现支持 I/O。 见下文。

还有一些较小的差异：

1. 构造函数采用一个可选参数，该参数指定您希望插入到哈希表中的元素数。 这与 SGI 的 hash_map 实现不同，它采用可选数量的桶。

2. erase() 不会立即回收内存。 因此，erase() 不会使任何迭代器失效，从而使循环正确：
      for (it = ht.begin(); it != ht.end(); ++it)
        if (...) ht.erase(it);
    另一个结果是，一系列的 erase() 调用会使您的哈希表使用比它需要的更多的内存。 哈希表将在下次调用 insert() 时自动压缩，但要手动压缩哈希表，您可以调用 ht.resize(0)

**IO**

除了正常的哈希映射操作之外，sparse_hash_map 还可以将哈希表读写到磁盘。 （dense_hash_map也有API，但是还没实现，写总会失败。）

在最简单的情况下，编写哈希表就像在哈希表上调用两个方法一样简单：
    ht.write_metadata(fp);
    ht.write_nopointer_data(fp);

读取这些数据同样简单：
    google::sparse_hash_map<...> ht;
    ht.read_metadata(fp);
    ht.read_nopointer_data(fp);

如果键和值不包含任何指针，以上内容就足够了：它们是基本 C 类型或基本 C 类型的集合。 如果键和/或值确实包含指针，您仍然可以通过将 write_nopointer_data() 替换为自定义写入例程来存储哈希表。 请参见 sparse_hash_map.html 等。 想要查询更多的信息。

**可分割**

除了 hash-map 和 hash-set 类之外，这个包还提供了 sparsetable.h，这是一个数组实现，它使用与数组中元素数量成比例的空间，而不是最大元素索引。 它使用的空间开销非常小：每个条目 2 到 5 bit。 有关 API，请参阅 doc/sparsetable.html。

**资源使用**

* 假设典型的平均占用率为 50%，sparse_hash_map 的每个哈希映射条目的内存开销约为 4 到 10 bit。
* dense_hash_map 有 2-3 倍的内存开销：如果你的哈希表数据占用 X 字节，dense_hash_map 将使用 3X-4X 内存总量。

调整大小时，哈希表的大小往往会加倍，从而产生额外的 50% 空间开销。 dense_hash_map 实际上确实有一个显着的“高水位线”内存使用要求，它是调整大小时表中散列条目大小的 6 倍（当达到 50% 的占用率时，表会调整到以前大小的两倍，而旧表 (2x) 被复制到新表 (4x))。

然而，sparse_hash_map 被编写为在调整大小时只需要很少的空间开销：每个哈希表条目只有几bit。

- - - 

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