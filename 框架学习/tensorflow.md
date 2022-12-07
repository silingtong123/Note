#### 常见结构
TF会给每一个核心概念设计一个构建类


- NodeDef
  - NodeDef只有repeated string input = 3，格式形如node:src_output;没有output
  - 前端
  - 后端：NodeProperties 联系opDef, Nodedef
- OpDef


#### 如何查看一个op

- 查看前端
- 查看后端
  - 查看op注册：REGISTER_OP("Select")
  - 查看kernel注册：REGISTER_XLA_OP(Name("Select"), SelectOp);

#### 常见结构
- LocalExecutorParams
  - "params" 为excecutor提供了一组上下文。我们期望不同的上下文会提供不同的实现。
- Executor 
  - Executor运行一个图的计算


#### 切片
- 高维矩阵理解：
  - 从数组理解[4,6,3] -> 4个[6,3]的数组
  - 从图片理解[4,6,4] ->HWC, [4,6]的图片有3个通道

####  shape推理
```
class Dimension {
  ...
  const int64 value_;
}

class DimensionHandle{
  ...
  const Dimension* ptr_ = nullptr;
}

Class shape {
  ...
  const int32 rank_;
  const std::vector<DimensionHandle> dims_;  
}

class ShapeHandle{
  ...
  const Shape* ptr_ = nullptr;
}

class DimensionOrConstant{
  ...
  //dim takes precedence. If dim != nullptr, val is ignored.
  DimensionHandle dim;
  int64 val;
}

struct ShapeAndType{
  ...
  ShapeHandle shape;
  DataType dtype = DT_INVALID;  
}

```

```
  ShapeManager shape_manager_;

  // inputs_, outputs_, and input_tensors_as_shapes_ 引用的来自
  // `shape_manager_`的值.
  std::vector<ShapeHandle> inputs_;
  std::vector<const Tensor*> input_tensors_;
  std::vector<bool> requested_input_tensor_; //标记改tensor会被使用
  std::vector<ShapeHandle> outputs_;

  // 下面2个vector大小可以比inputs_小.
  std::vector<ShapeHandle> input_tensors_as_shapes_;
  std::vector<bool> requested_input_tensor_as_partial_shape_;

// input_handle_shapes_and_types_[i] 是通过沿节点的输入 i 传递的资源句柄可用的形状/类型对的列表，其值可能为空
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
      input_handle_shapes_and_types_;

//output_handle_shapes_and_types_[i] 是通过沿节点的输出 i 传递的资源句柄可用的形状/类型对的列表。值可能为 NULL
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
      output_handle_shapes_and_types_;

  const int graph_def_version_;
  const NodeDef* node_def_;
  NameRangeMap input_name_map_; //根据名字返回输入shapehandle
  NameRangeMap output_name_map_; //根据名字返回输出shapehandle

  // An error set during construction. TODO(cwhipkey): remove when test
  // constructor is removed.
  Status construction_status_;

  // 多组等效的shape或者dim的形状表示，每个pair中至少有一个未知的handle，这里不记录已知的
  std::vector<std::pair<ShapeHandle, ShapeHandle>> merged_shapes_;
  std::vector<std::pair<DimensionHandle, DimensionHandle>> merged_dims_;
```

#### OpKernel
```
class  OpKernelConstruction{
  OpKernelConstruction(DeviceType device_type, DeviceBase* device,
                       Allocator* allocator, const NodeDef* node_def,
                       const OpDef* op_def, FunctionLibraryRuntime* flib,
                       const DataTypeSlice& input_types,
                       const MemoryTypeSlice& input_memory_types,
                       const DataTypeSlice& output_types,
                       const MemoryTypeSlice& output_memory_types,
                       int graph_def_version, Status* status);

  const DeviceType device_type_;
  DeviceBase* const device_;
  Allocator* allocator_;
  const NodeDef* def_;
  const OpDef* op_def_;
  FunctionLibraryRuntime* flib_;
  DataTypeSlice input_types_;
  MemoryTypeSlice input_memory_types_;
  DataTypeSlice output_types_;
  MemoryTypeSlice output_memory_types_;
  const int graph_def_version_;
  Status* status_;
  ......
  allocate_temp //申请临时tensoor
  allocate_persistent //申请永久tensor  
}

typedef gtl::FlatMap<StringPiece, std::pair<int, int>, hash<StringPiece>>
    NameRangeMap;

explicit OpKernel(OpKernelConstruction* context);

const std::unique_ptr<const NodeDef> def_;
const DataTypeVector input_types_;
const MemoryTypeVector input_memory_types_;
const DataTypeVector output_types_;
const MemoryTypeVector output_memory_types_;
NameRangeMap input_name_map_;
NameRangeMap output_name_map_;
const int graph_def_version_;
bool expensive_;
std::atomic_uint_fast64_t cost_estimate_;
```
#### OpKernelContext
```
class Params {

}
  Status status_;
  friend class CollectiveExecutor;  // for access to params_
  Params* params_;                  // not owned
  mutable mutex mu_;  // mutable so const accessors can acquire the lock
  gtl::InlinedVector<WrappedAllocator, 4> wrapped_allocators_ GUARDED_BY(mu_);
  gtl::InlinedVector<TensorValue, 4> outputs_;

  // Keep track of calls to ScopedAllocator.
  // TODO(ayushd): change to absl::flat_hash_set.
  std::unique_ptr<std::unordered_set<int32>> allocated_scope_ids_;
```

#### Tensor
```
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>  Tensor;

//struct TTypes::tensor 其实为egien::tensor的别名
//DenseIndex -> std::ptrdiff_t ->貌似为long int
//Eigen::Tensor第三个参数为0表示ColMajor， 为1表示RowMajor
```

#### 并行加速
```

```