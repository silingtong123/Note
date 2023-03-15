### Save Restore

- tf.train.Saver()/saver.restore()：Saver中可传入list或dict，用来指定保存或恢复时候的变量。默认是所有变量。恢复的变量应当是保存时候变量的一个子集。要恢复的变量必须存在
  - 如果保存和恢复的模型名不一样，应该使用map进行映射
  - checkpoint：保存并维护着模型列表，以及最新模型的文件名
  - model.ckpt-7867424.data-00000-of-00001: 代表device信息，比如有一个GPU，且在第0个上
  - model.ckpt-7867424.index
  - model.ckpt-7867424.meta
- saver.export_meta_graph()/ tf.train.import_meta_graph：graph.util.convert_variables_to_constants 转variable为常量，保存为pb文件，常用语推理
- tf.train.write_graph()/tf.Import_graph_def()：


Saver 保存加载恢复变量
- 可指定保存模型，默认保存所有
- _all_saveable_objects：返回所有必须设置检查点的variable和“SaveableObject”。
- shard: 如果为“True”，则对检查点进行分片，每个设备一个
- control_flow_ops.with_dependencies([save], filename_tensor): 仅在依赖后产生 output_tensor 的内容
- save_op: gen_io_ops.save(filename, tensor_names, tensors, name=name) ->_apply_op_helper:返回 output_structure 的 apply_op 的实现,创建一个op实例 g.create_op ->

创建ev的函数调用顺序:
- get_embedding_variable
- variableScope.get_embedding_variable
- _VariableStore.get_variable
- _VariableStore._true_getter
- _VariableStore._get_partitioned_variable; _VariableStore._get_single_variable
- variables.VariableV1(Variable)：__init__方法未被调用
- Variable(six.with_metaclass(VariableMetaclass, trackable.Trackable)):自身大部分方法未实现，依赖父类方法和元类， __init__方法未被调用
- _variable_v1_call 
- default_variable_creator variable_scope.py
- kv_variable_ops.EmbeddingVariable
- _embedding_variable_handle -> gen_kv_variable_ops.kv_var_handle_op ->调用KvVarHandleOp Op -> ResourceHandleOp<EmbeddingVar<ktype, vtype>> ->
- gen_kv_variable_ops.kv_var_is_initialized_op ->调用 KvVarIsInitializedOp Op
- gen_kv_variable_ops.initialize_kv_variable_op ->调用 InitializeKvVariableOp Op
- initializer -> init_val: TruncatedNormal-> gen_random_ops.truncated_normal ->调用TruncatedNormal OP
- gen_kv_variable_ops.kv_resource_gather -> KvResourceGather
```C++
struct TF_Operation {
  tensorflow::Node node;
};

class ResourceHandleOp{
  string container_;
  string name_;
  mutex mutex_;
  Tensor resource_;
  std::atomic<bool> initialized_{false};  
}

class ResourceHandle { // <=====> ResourceHandleProto

}
///usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/gen_
```
Save和Restore Op的创建
- gen_io_ops.restore -> 创建 RestoreOp 源码中无处调用
- Saver初始化中调用：_AddRestoreOps -> bulk_restore（其后遍历saveables并调用每个saveable的restore函数） -> io_ops.restore_v2 <==> gen_io_ops.restore_v2(遍历每个saveable.specs的spec调用restore_v2 ) -> 创建 RestoreV2 Op
- gen_io_ops.save ->创建 SaveOp

SaveableObject: 用于保存和恢复可保存对象的基类
- ReferenceVariableSaveable(saveable_object.SaveableObject)
  - restore -> state_ops.assign
- ResourceVariableSaveable(saveable_object.SaveableObject)
  - restore -> gen_array_ops.identity ->创建 identity OP 
  - restore -> gen_resource_variable_ops.assign_variable_op 创建 AssignVariableOp 
- EmbeddingVariableSaveable(saveable_object.SaveableObject)
    - restore ->gen_kv_variable_ops.kv_resource_import_v2 -> 创建"KvResourceImportV2" OP
- HashTableSaveable(saveable_object.SaveableObject)
- CoalescedVariableSaveable(saveable_object.SaveableObject)
  - restore -> state_ops.assign

ResourceVariable(BaseResourceVariable)
- 
resource_variable_ops.ResourceVariable
- EmbeddingVariable(resource_variable_ops.ResourceVariable)
  - def export -> gen_kv_variable_ops.kv_resource_export 创建KvResourceExport
- MultiHashVariable(resource_variable_ops.ResourceVariable)
- DynamicEmbeddingVariable(resource_variable_ops.ResourceVariable)


BaseSaverBuilder: savers的基类，可以扩展以创建不同的OP
- BulkSaverBuilder(BaseSaverBuilder)：SaverBuilder 支持批量恢复多个可保存文件。

SaveableObject: 用于保存和恢复可保存对象的基类。
```py
class SaveableObject
def __init__(self, op, specs, name):
    """Creates a `SaveableObject` object.

    Args:
      op: 这个类包装的“生产者”对象； 它会生成要保存的张量列表。 例如，保存其后备张量的“Varible”对象。
      specs: SaveSpec 列表，其中每个元素描述了一个要保存在该对象下的张量。 所有张量必须在同一台设备上。
      name: 保存对象的名称。
    """ 
```
SaveSpec: 用于描述需要保存的张量切片的类
```py
class SaveSpec:
  def __init__(self, tensor, slice_spec, name, dtype=None, device=None):
    """Creates a `SaveSpec` object.

    Args:
      tensor: 要保存的张量或生成要保存的张量的可调用张量。.
      slice_spec: 要保存的切片。 请参阅“Variable.SaveSliceInfo”.
      name: the name to save the tensor under.
      dtype: The data type of the Tensor. Required if `tensor` is callable.
        Used for error checking in the restore op.
      device: The device generating and consuming this tensor. Required if
        `tensor` is callable. Used to group objects to save by device.
    """

class SaveSliceInfo:
  def __init__(self,
                   full_name=None,
                   full_shape=None,
                   var_offset=None,
                   var_shape=None,
                   save_slice_info_def=None,
                   import_scope=None,
                   var_full_name=None):      
```

Variable.SaveSliceInfo: 有关如何将此变量保存为切片的信息 **Slice Tensor才有效？测试slice_spec为空**


variables.VariableV1
- BaseResourceVariable(variables.VariableV1)
- RefVariable(VariableV1)
- PartitionedVariables: create_partitioned_variables

_VariableStore: 承载多个命名变量的变量存储


ops.Tensor：代表一个op的输出
```python
def __init__(self, op, value_index, dtype): //tensor定义时,知道自己是某个op的第某个输出
```

- CheckpointInitialValue(ops.Tensor)：用于管理“Variable”中更新 UID 的张量包装器， c_api.TFE_Py_UID()，计数器 i++


Graph: TensorFlow 计算，表示为数据流图
- self._scoped_c_graph = c_api_util.ScopedTFGraph(), ->self.graph = c_api.TF_NewGraph()

NodeDef: proto. 
```proto
message NodeDef {
  string name = 1;
  string op = 2;
  repeated string input = 3;
  string device = 4;
  map<string, AttrValue> attr = 5;
}
```

Operation: 表示对tensor执行计算的图形节点
- self._c_op = _create_c_op -> op_desc = c_api.TF_NewOperation -> c_op = c_api.TF_FinishOperation(op_desc)

Trackable: 没有自动依赖项的 `Trackable` 对象的基类（可追踪的）
- _add_variable_with_custom_getter: 使用此“Trackable”保存变量的创建时还原
- add_variable ->_add_variable_with_custom_getter ->_default_getter

ops.name_scope:定义 Python op时使用的上下文管理器
init_scope: 将操作从控制流范围和功能构建图中提升出来的上下文管理器

Initializer
- VarianceScaling(Initializer): 
- 其他数据类型的默认initializer：Zeros(Initializer)
- 数据类型为float 时的默认initializer：GlorotUniform(VarianceScaling[方差缩放]): Glorot 统一初始化器，也称为 Xavier 统一初始化器

BaseResourceVariable(variables.VariableV1)
- resource_variable_ops.ResourceVariable(BaseResourceVariable):基于资源句柄的Variable
- DynamicEmbeddingVariable(resource_variable_ops.ResourceVariable): 


### C++端

SaveOp: ->Compute -> SaveTensors  ->TensorSliceWriter
- TensorSliceWriter(CreateTableTensorSliceBuilder) 通过** TensorSliceWriter::Builder返回
- class TableBuilder : public TensorSliceWriter::Builder
- file->Read(uint64 offset, size_t n, StringPiece* result, char* scratch) (file为指针类型)
- ssize_t pread(intfd, void *buf, size_t count, off_t offset); 从文件开始+off_toffset 读取size_tcount个字符，放入buf
- ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset); 从文件开始+off_toffset将buf中count个字符写入文件
```C++
s = StringPiece(*p, size_type length) //初始化一个StringPiece对象，不持有数据，只是一个指针和长度
class TensorSliceReader { //用于从ckpt文件中读取数据, 先读footer 再读index block最后读取数据
  const string filepattern_;
  const OpenTableFunction open_function_;
  std::vector<string> fnames_;
  std::unordered_map<string, int> fname_to_index_;

  // Guards the attributes below.
  mutable mutex mu_;
  mutable bool all_shards_loaded_ = false;
  mutable std::vector<std::unique_ptr<Table>> sss_;
  mutable std::unordered_map<string, TensorSliceSet*> tensors_;
  mutable Status status_;
}// SavedTensorSlices proto; SavedTensorSlices.data().data() <==> tensorProto

class TensorSlice {

}
class Footer { // Footer封装了每个表格文件尾部存储的固定信息。
 private:
  BlockHandle metaindex_handle_;
  BlockHandle index_handle_; 
}

class BlockHandle { //BlockHandle 是指向存储数据块或元块的文件范围的指针。
 private:
  uint64 offset_;
  uint64 size_;  
}

class Block { //用指定的内容初始化块。32bit crc循环校验
 private:
  uint32 NumRestarts() const;

  const char* data_;
  size_t size_;
  uint32 restart_offset_;  // Offset in data_ of restart array
  bool owned_;             // Block owns data_[]  
}

class TensorSliceWriter { // //用于从ckpt文件中写入数据
private:
  static const size_t kMaxMessageBytes = 1LL << 31;
  // Filling in the TensorProto in a SavedSlice will add the following
  // header bytes, in addition to the data:
  // - 1 byte: TensorProto tag and wire format
  // - <= 5 bytes: TensorProto length
  // - 1 byte: Repeated *_val tag and wire format
  // - <= 5 bytes: *_val length
  // However, we add 1KB of slack, to be conservative and guard
  // against other additions to the TensorProto.
  static const size_t kTensorProtoHeaderBytes = 1 << 10;

  const string filename_;
  const CreateBuilderFunction create_builder_;
  const string tmpname_;

  // A mapping from the tensor names to their index in meta_.saved_slice_meta()
  std::unordered_map<string, int> name_to_index_;
  // The metadata that holds all the saved tensor slices.
  SavedTensorSlices sts_;
  // The data to be written to the builder
  std::map<string, string> data_;
  // Total number of slices written
  int slices_;
}

struct TableBuilder{
    TableBuilder::Rep* rep_;
    TableBuilder::WriteRawBlock() //文件中写入数据
}

class Table {//线程安全，
  static Status Open(const Options& options, RandomAccessFile* file, 
                     uint64 file_size, Table** table); // 打开fiel检索[0,file_size]的表
   Rep* rep_;
}

struct TableBuilder::Rep {
  Options options;
  Options index_block_options;
  WritableFile* file;
  uint64 offset;
  Status status;
  BlockBuilder data_block;
  BlockBuilder index_block;
  string last_key;
  int64 num_entries;
  bool closed;  // Either Finish() or Abandon() has been called.
 
  bool pending_index_entry;
  BlockHandle pending_handle;  // Handle to add to index block

  string compressed_output;
}

class BlockBuilder{
  const Options* options_;
  string buffer_;                 // 目标缓冲区
  std::vector<uint32> restarts_;  // 重启点
  int counter_;                   // 自重启后发出的条目数
  bool finished_;                 // 是否调用了 Finish()?
  string last_key_;    
}


class BlockHandle { //BlockHandle 是指向存储数据块或元块的文件范围的指针。

}
```

OpRegistryInterface
- OpRegistry : public OpRegistryInterface
```C++
class OpRegistry : public OpRegistryInterface {
  mutable mutex mu_;
  // Functions in deferred_ may only be called with mu_ held.
  mutable std::vector<OpRegistrationDataFactory> deferred_ GUARDED_BY(mu_);
  // Values are owned.
  mutable std::unordered_map<string, const OpRegistrationData*> registry_
      GUARDED_BY(mu_);
  mutable bool initialized_ GUARDED_BY(mu_);

  // Registry watcher.
  mutable Watcher watcher_ GUARDED_BY(mu_);    
}

struct OpRegistrationData {
 public:
  OpRegistrationData() {}
  OpRegistrationData(const OpDef& def) : op_def(def) {}
  OpRegistrationData(const OpDef& def, const OpShapeInferenceFn& fn,
                     bool is_function = false)
      : op_def(def), shape_inference_fn(fn), is_function_op(is_function) {}

  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;
  bool is_function_op = false;
};


class ResourceHandle <===> ResourceHandleProto
// ResourceHandle 支持序列化反序列化

message ResourceHandleProto {
  // Unique name for the device containing the resource.
  string device = 1;

  // Container in which this resource is placed.
  string container = 2;

  // Unique name of this resource.
  string name = 3;

  // Hash code for the type of the resource. Is only valid in the same device
  // and in the same execution.
  uint64 hash_code = 4;

  // For debug-only, the name of the type pointed to by this handle, if
  // available.
  string maybe_type_name = 5;

  // Protocol buffer representing a pair of (data type, tensor shape).
  message DtypeAndShape {
    DataType dtype = 1;
    TensorShapeProto shape = 2;
  }

  // Data types and shapes for the underlying resource.
  repeated DtypeAndShape dtypes_and_shapes = 6;
};

class BundleReader  {
  Env* env_;  // Not owned.
  const string prefix_;

  Status status_;
  RandomAccessFile* metadata_;  // Owned. 会打开一个文件
  table::Table* table_;
  table::Iterator* iter_;
  // Owned the InputBuffer objects and their underlying RandomAccessFile's.
  std::unordered_map<int32, io::InputBuffer*> data_;
  // 使用 InputBuffer::FillBuffer/ReadLine -》填充数据data_

  // Maps each partitioned tensor's key to its stored slices (represented in a
  // TensorSliceSet).  Populated on-demand.
  std::unordered_map<string, checkpoint::TensorSliceSet*> tensor_slices_;

  std::map<std::string, LookupSegItem> tmp_lookupseg_items_;

  // Expected number of data file shards in the bundle.  Extracted by reading
  // the header entry in the metadata table.
  int num_shards_;

  // Flag that this class sets to true when the endianness of the target bundle
  // differs from that of the current system's processor architecture.
  bool need_to_swap_bytes_;

}
BundleReader::LookupTensorShape -> LookupDtypeAndShape 查看tensor的shape和dtype
BundleReader::LookupHeader(StringPiece tensor_key, int64 total_bytes) -> 根据上面获得的shape获取
reader->LookupSegment ->获取内容段
// pread/pread操作是原子性的，seek和read/write操作一起完成，适合用于多线程中
// 成功，返回成功读取数据的字节数,失败为-1

BundleEntryProto 描述与检查点张量相关的元数据
TensorSliceProto 只有知道相应的 TensorShape 才能解释
```

### 保存恢复ev的一些问题：
- 保存之前必须要sess.run(init), 否则会报错
- sess.run(init)后必须sess.run(emb)，否则保存的ev内容为空
- 恢复所有变量的时候，可以通过dim维度变化对原来的emb维度进行变化，如果想变化初始值为随机值，必须先初始化再restore
- 恢复部分变量的时候，可以通过dim维度变化对原来的emb维度进行变化， 填充值为0

```py
import tensorflow as tf
  
var = tf.get_embedding_variable("var_0",
                                embedding_dim=5,
#                                initializer=tf.ones_initializer(tf.float32),
                                partitioner=tf.fixed_size_partitioner(num_shards=4))


var2 = tf.get_embedding_variable("var_23",
                                embedding_dim=3,
                                partitioner=tf.fixed_size_partitioner(num_shards=4))

shape = [var1.total_count() for var1 in var]

emb = tf.nn.embedding_lookup(var, tf.cast([0,1,2,5,6,7], tf.int64))
emb2 = tf.nn.embedding_lookup(var2, tf.cast([0,1,2,5,6,7], tf.int64))
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.AdagradOptimizer(0.1)

g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)

init = tf.global_variables_initializer()

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#saver = tf.train.Saver([var], max_to_keep=2)
saver2 = tf.train.Saver( max_to_keep=2)
with tf.Session(config=sess_config) as sess:
  sess.run([init])
  print('----------------------------')
#  print(sess.run([emb]))
#  print(sess.run([emb2]))
#  saver2.save(sess, "Saved_model/test.ckpt", global_step=0)
  saver2.restore(sess, "Saved_model/test.ckpt-0")

  print('----------------------------')
  print(sess.run([emb]))
  print(sess.run([emb2]))
```

### 加载单个变量，进行验证restore的正确性
```py
import tensorflow as tf
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.framework import meta_graph

ckpt_path='ckpt_test/ckpt'
path=ckpt_path+'/model.meta'
meta_graph = meta_graph.read_meta_graph_file(path)
ev_node_list=[]
for node in meta_graph.graph_def.node:
  if node.op == 'KvVarHandleOp':
    ev_node_list.append(node.name)

print("ev node list", ev_node_list)
# filter ev-slot
non_slot_ev_list=[]
for node in ev_node_list:
  if "Adagrad" not in node:
    non_slot_ev_list.append(node)
print("ev (exculde slot) node list", non_slot_ev_list)

for name in non_slot_ev_list:
  print(name+'-keys', tf.train.load_variable(ckpt_path, name+'-keys'))
  print(name+'-values', tf.train.load_variable(ckpt_path, name+'-values'))
  print(name+'-freqs', tf.train.load_variable(ckpt_path, name+'-freqs'))
  print(name+'-versions', tf.train.load_variable(ckpt_path, name+'-versions'))
```
