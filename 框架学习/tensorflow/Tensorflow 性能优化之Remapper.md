# Tensorflow 性能优化之Remapper [参考链接](https://zhuanlan.zhihu.com/p/264946227)

```C++
// Matmul node followed by a Reshape, BiasAdd,.
// Matmul节点后面跟一个Reshape和BiasAdd
struct MatMulWithReshapeAndBias {
  MatMulWithReshapeAndBias() = default;
  MatMulWithReshapeAndBias(int matmul, int reshape, int bias_add)
      : matmul(matmul), reshape(reshape), bias_add(bias_add) {}

  //记录这3个节点的index
  int matmul = kMissingIndex;
  int reshape = kMissingIndex;
  int bias_add = kMissingIndex;
};

struct GrapplerItem {
  GrapplerItem() = default;
  GrapplerItem(const GrapplerItem& other) = default;
  GrapplerItem(GrapplerItem&& other) = default;
  GrapplerItem& operator=(const GrapplerItem& other) = default;
  GrapplerItem& operator=(GrapplerItem&& other) = default;
  virtual ~GrapplerItem() = default;

  // Create a copy of this GrapplerItem with graph swapped with the argument.
  GrapplerItem WithGraph(GraphDef&& graph) const;

  string id;  // A unique id for this item

  // 输入，一般由外界传入
  GraphDef graph;
  std::vector<std::pair<string, Tensor>> feed;
  std::vector<string> fetch;

  // Initialization op(s).
  std::vector<string> init_ops;
  // Expected initialization time in seconds, or 0 if unknown
  int64 expected_init_time = 0;

  // Save/restore ops (if any)
  string save_op;
  string restore_op;
  string save_restore_loc_tensor;

  // Queue runner(s) required to run the queue(s) of this model.
  std::vector<QueueRunnerDef> queue_runners;

//要保留在图中的op名称列表
  std::vector<string> keep_ops;
   
  // 下面函数皆是通过调用ComputeTransitiveFanin函数实现
  // 返回在常规训练/推理步骤中评估的节点集。
  std::vector<const NodeDef*> MainOpsFanin() const;
  // 返回节点运行集以填充队列（如果有）
  std::vector<const NodeDef*> EnqueueOpsFanin() const;
  // 返回 TensorFlow 用于初始化图的集合节点。
  std::vector<const NodeDef*> InitOpsFanin() const;
  // 返回在常规训练/推理步骤中访问的变量集。
  std::vector<const NodeDef*> MainVariables() const;
  // 返回一组必须保留的节点名称。 这包括 feed 和 fetch 节点、keep_ops、init_ops。
  std::unordered_set<string> NodesToPreserve() const;
}

//先遍历整个graph的图添加name2node map，再根据terminal_nodes，从下到上通过node-inputs和 BFS的方式，获取所有相关节点
std::vector<const NodeDef*> ComputeTransitiveFanin() 

// 收集GrapplerItem中所有需要保持的节点
struct RemapperContext {
  explicit RemapperContext(GrapplerItem* item, Status* status, bool xla_on)
      : nodes_to_preserve(item->NodesToPreserve()),
        graph_view(&item->graph, status),
        graph_properties(*item),
        inferred_graph_properties(false),
        xla_on_(xla_on) {}

  std::unordered_set<string> nodes_to_preserve;
  utils::MutableGraphView graph_view;
  GraphProperties graph_properties;
  bool inferred_graph_properties;
  bool xla_on_;
};

//查找符合条件的结构 BiasAdd <- Reshape <-Matmul,先由bias查起，返回相应的node_index()
bool FindMatMulWithReshapeAndBias(const RemapperContext& ctx, int node_index,
                                  MatMulWithReshapeAndBias& matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // 范式的根节点必须是Biasadd.
  // TODO: Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;
  const auto* node_def = node_view->node();
  if (!IsBiasAdd(*node_def)) return false;

  // Input to the BiasAdd must be a Reshape.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* reshape_node_view = regular_fanin_0.node_view();
  const auto* reshape_node_def = reshape_node_view->node();
  if (HasControlFaninOrFanout(*reshape_node_view)) return false;
  if (!IsReshape(*reshape_node_def)) return false;
  

  // First input to the Reshape must be a Matmul.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& reshape_regular_fanin_0 = reshape_node_view->GetRegularFanin(0);
  const auto* matmul_node_view = reshape_regular_fanin_0.node_view();
  const auto* matmul_node_def = matmul_node_view->node();
    

  if (!IsMatMul(*matmul_node_def) || !IsCpuCompatibleMatMul(matmul_node_def) ||
      !HaveSameDataType(node_def, matmul_node_def) ||
      HasControlFaninOrFanout(*matmul_node_view) ||
      !HasAtMostOneFanoutAtPort0(*matmul_node_view) ||
      IsInPreserveSet(ctx, matmul_node_def))
    return false;

  matched.bias_add = node_view->node_index();
  matched.reshape = reshape_node_view->node_index();
  matched.matmul = matmul_node_view->node_index();

  return true;
}
```

- 调用流程：MetaOptimizer::OptimizeGraph -> InitializeOptimizers()/InitializeOptimizersByName:根据cfs添加对应的优化器 ->InitializeCustomGraphOptimizers ->  MakeNewOptimizer -> return new Remapper(cfg_.remapping(), xla_on_)