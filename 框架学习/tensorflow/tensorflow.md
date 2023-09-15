#### 常见结构
TF会给每一个核心概念设计一个构建类


- NodeDef
  - NodeDef只有repeated string input = 3，格式形如node:src_output;没有output
  - 前端
  - 后端：NodeProperties 联系opDef, Nodedef
- OpDef


#### 如何查看一个op

- 查看前端
- 查看后端：op和kernel的注册都是通过全局static变量来实现的
  - 查看op注册：REGISTER_OP("Select"),  REGISTER_OP是注册OpDef中的静态信息和SetShapeFn
    - static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr = OpDefBuilderWrapper：OpDefBuilderReceiver使用OpDefBuilderWrapper进行初始化，其初始化函数中调用OpRegistry::Global()->Register进行注册, 注册结构为vector deferred_，该vector元素为std::function，该函数主要调用wrapper.builder().Finalize
    - OpDefBuilderWrapper 主要成员为 OpDefBuilder builder_， 主要函数为Attr，input, output, SetShapeFn等，这些函数底层调用OpDefBuilder的对应函数
    - OpDefBuilder 起主要成员为OpRegistrationData op_reg_data_，Finalize函数负责最终初始化op_reg_data->op_def， 而OpRegistrationData的主要成员OpDef op_def; OpShapeInferenceFn shape_inference_fn;
  - 查看kernel注册, REGISTER_KERNEL_BUILDER, 其中SelectOp 注册真正运算函数compute函数：
  
    ```c++
    REGISTER_KERNEL_BUILDER(                                           \
      Name("Select").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      SelectOp<CPUDevice, type>);
    
    //register_kernel命名空间下
    class Name : public KernelDefBuilder{}

    //主要成员KernelDef* kernel_def_;
    class KernelDefBuilder{

    }
    ```

    - KernelDefBuilder大多成员函数为设置kernel_def_， 其中Build成员函数返回kernel_def_指针并移交所有权
    - static ::tensorflow::kernel_factory::OpKernelRegistrar ,其初始化函数参数为获得所有权的KernelDef *，class name, OpKernelFactory * factory指针(factory函数指针会创建一个该Op的对象，如上创建一个SelectOp<CPUDevice, type>，并返回指针 )，该函数会调用InitInternal注册kernel
    - InitInternal 中会将factory注册到map中，其中key为kernel name和设备名以及label的concat，value为KernelRegistration(其用KernelDef，class name, OpKernelFactory * factory进行初始化，并初始化对应成员)，
  
#### 常见结构
- LocalExecutorParams
  - "params" 为excecutor提供了一组上下文。我们期望不同的上下文会提供不同的实现。
- Executor 
  - Executor运行一个图的计算

### 优化器

```python
grads_and_vars = _optimizer.compute_gradients(loss, var_list=var_list)
_optimizer.apply_gradients(grads_and_vars, global_step=global_step)

#上面两步等价于 _optimizer.minimize(loss, global_step=global_step, var_list=var_list) 
```

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