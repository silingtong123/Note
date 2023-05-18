from tensorflow.contrib.framework.python.framework import checkpoint_utils
ckpt_name="/Newdeeprec/gitdownload/demo/Saved_model/test.ckpt-0"
for name, shape in checkpoint_utils.list_variables(ckpt_name):
    print('loading... ', name, shape, checkpoint_utils.load_variable(ckpt_name, name))