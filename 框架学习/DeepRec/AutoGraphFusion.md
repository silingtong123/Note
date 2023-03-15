## 子图替换 [官方文档](https://deeprec.readthedocs.io/zh/latest/Auto-Fusion.html)
- enter节点的输入节点在父帧，普通节点和输入在同帧
- exit节点的输出节点在父帧，普通节点和输出在同帧
- 好处：
  - 消除不必要的中间结果实例化
  - 减少不必要的输入扫描
  - 实现其他优化机会。
### 步骤
- 注册fusion
```C++
//在tensorflow/core/graph/optimizer_fusion_engine.cc
templates.emplace_back(new TemplateLogicSumBase());

// 根据enter node的frameName 通过BFS收集节点信息放入node_frame_map_  map[node*, string]
```
- 编写子图替换逻辑

```C++
//tensorflow/core/graph/template_logicsum_base.h

class TemplateBase {
 public:
  std::vector<TempNode> temp_nodes_;

  std::string first_key_;
  int num_inputs_;
  int num_outputs_;
  int num_deps_inputs_ = 0;

  std::string fused_op_;
  // 存储具有来自同一个src 节点的动态入边数的节点
  std::map<std::string, int> nodes_dynamic_iedges_;
  // 将具有动态出边数的节点存储到同一个 dst 节点
  std::map<std::string, int> nodes_dynamic_oedges_;
  // 存储从添加节点的名称到模板中其键的映射
  std::map<std::string, std::string> node_to_temp_key_;
}
```
- 编写融合的新op
```
tensorflow/core/kernels/BUILD

":logical_sum_op",

tf_kernel_library(
    name = "logical_sum_op",
    prefix = "logical_sum_op",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ]
)
```
```C++
 //新op kernel tensorflow/core/kernels/logical_sum_op.cc

 //新op 定义 tensorflow/core/ops/math_ops.cc

 REGISTER_OP("LogicalSum")
    .Input("q_input: float")
    .Input("k_input: float")
    .Output("values: T")
    .Attr("T: {int16,int32,float}")
    .SetShapeFn([](InferenceContext* c) {
      auto input_p_shape = c->input(0);
      auto input_k_shape = c->input(1);
      auto batch_size = c->Value(c->Dim(input_p_shape, 0));
      tensorflow::int64 p_d1 = -1;
      if (c->ValueKnown(c->Dim(input_p_shape, 1))) {
        p_d1 = c->Value(c->Dim(input_p_shape, 1));
      }
      tensorflow::int64 k_d1 = -1;
      if (c->ValueKnown(c->Dim(input_k_shape, 1))) {
        k_d1 = c->Value(c->Dim(input_k_shape, 1));
      }
      auto seq_length = c->Value(std::max(p_d1, k_d1));
      auto result_shape = c->MakeShape({batch_size, seq_length});
      c->set_output(0, result_shape);
      return Status::OK();
    });
```

```C++
// template_base.h
add_iedge(Graph * g, Node * dst, int dst_input, const Edge * ori_edge)
//将ori的输出Node换为dst,即为dst增加输入边，默认会移除ori_edge
...
g->AddEdge


add_odeges((Graph * g, Node * src, int src_output, std::vector<const Edge*>& ori_edges))
// 将ori所有输入Node替换为src，即为src增加输出边， ，默认会移除ori_edges
g->AddEdge
```

```C++
Graph {
    FunctionLibraryDefinition ops_;
    std::vector<Node*> nodes_;
    std::vector<Edge*> edges_;
    int64 num_nodes_ = 0;
    int num_edges_ = 0;
    std::vector<Node*> free_nodes_;
    std::vector<Edge*> free_edges_;
    ...
    AddNode(NodeDef node_def, Status* status)
    RemoveNode(Node* node)
    AddEdge(Node* source, int x, Node* dest, int y)
    Graph::RemoveEdge(const Edge* e)
    ...
}

Node {
    int id_;       
    int cost_id_;  
    NodeClass class_;

    EdgeSet in_edges_;
    EdgeSet out_edges_;

    std::shared_ptr<NodeProperties> props_;
    EdgeSet {
        void *ptrs[N];
        //第0号元素特殊，1-N为Edge
    }
}

Edge {
    Node* src_;
    Node* dst_;
    int id_;
    int src_output_;
    int dst_input_;
}

struct NodeProperties {
 public:
  NodeProperties(const OpDef* op_def, NodeDef node_def,
                 const DataTypeSlice inputs, const DataTypeSlice outputs)
      : op_def(op_def),
        node_def(std::move(node_def)),
        input_types(inputs.begin(), inputs.end()),
        output_types(outputs.begin(), outputs.end()) {}

  const OpDef* op_def;  // not owned
  NodeDef node_def;
  const DataTypeVector input_types;
  const DataTypeVector output_types;
};

TF_NewOperation ->TF_OperationDescription

struct TF_OperationDescription {
  tensorflow::NodeBuilder node_builder;
  TF_Graph* graph;
  std::set<tensorflow::string> colocation_constraints;
};

class NodeBuilder{
  NodeDefBuilder def_builder_;
  std::vector<NodeOut> inputs_;
  std::vector<Node*> control_inputs_;
  std::vector<string> errors_;
  string assigned_device_;
};

class NodeDefBuilder{
  const OpDef* op_def_;
  NodeDef node_def_;
  int inputs_specified_;
  std::vector<string> control_inputs_;
  std::vector<string> errors_;    
}

// 描述src:src_output_ --> dst_：dst_input_的边
```