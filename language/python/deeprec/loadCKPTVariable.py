import tensorflow as tf
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.framework import meta_graph

ckpt_path='ckpt_test/test.ckpt'
ckpt_p = 'Saved_model'
path=ckpt_path+'/model.meta'
new_p = ckpt_p +'/test.ckpt-0.meta'

path = new_p
ckpt_path = ckpt_p

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