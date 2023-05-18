import time
import argparse
import tensorflow as tf
import os
import sys
import math
import collections
import json

from tensorflow.python.feature_column import utils as fc_utils

result_dir='/tmp/tianchi/result/DeepFM/'
result_path=result_dir+'result'
global_time_cost = 0
global_auc = 0

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
LABEL_COLUMN = ['clicked']
LABEL_COLUMN_DEFAULTS = [0]
USER_COLUMNS = [
    'user_id', 'gender', 'visit_city', 'avg_price', 'is_supervip', 'ctr_30',
    'ord_30', 'total_amt_30'
]
USER_COLUMNS_DEFAULTS = ['', -99, '0', 0.0, 0, 0, 0, 0.0]
ITEM_COLUMN = [
    'shop_id', 'item_id', 'city_id', 'district_id', 'shop_aoi_id',
    'shop_geohash_6', 'shop_geohash_12', 'brand_id', 'category_1_id',
    'merge_standard_food_id', 'rank_7', 'rank_30', 'rank_90'
]
ITEM_COLUMN_DEFAULTS = ['', '', '0', '0', '', '', '', '0', '0', '0', 0, 0, 0]
HISTORY_COLUMN = [
    'shop_id_list', 'item_id_list', 'category_1_id_list',
    'merge_standard_food_id_list', 'brand_id_list', 'price_list',
    'shop_aoi_id_list', 'shop_geohash6_list', 'timediff_list', 'hours_list',
    'time_type_list', 'weekdays_list'
]
HISTORY_COLUMN_DEFAULTS = ['', '', '', '', '', '0', '', '', '0', '-0', '', '']
USER_TZ_COLUMN = ['times', 'hours', 'time_type', 'weekdays', 'geohash12']
USER_TZ_COLUMN_DEFAULTS = ['0', 0, '', 0, '']
DEFAULTS = LABEL_COLUMN_DEFAULTS + USER_COLUMNS_DEFAULTS + ITEM_COLUMN_DEFAULTS + HISTORY_COLUMN_DEFAULTS + USER_TZ_COLUMN_DEFAULTS

FEATURE_COLUMNS = USER_COLUMNS + ITEM_COLUMN + HISTORY_COLUMN + USER_TZ_COLUMN
TRAIN_DATA_COLUMNS = LABEL_COLUMN + FEATURE_COLUMNS
SHARE_EMBEDDING_COLS = [
    ['shop_id', 'shop_id_list'], ['item_id', 'item_id_list'],
    ['category_1_id', 'category_1_id_list'],
    ['merge_standard_food_id', 'merge_standard_food_id_list'],
    ['brand_id', 'brand_id_list'], ['shop_aoi_id', 'shop_aoi_id_list'],
    ['shop_geohash_12', 'geohash12'], ['shop_geohash_6', 'shop_geohash6_list'],
    ['visit_city', 'city_id']
]
EMBEDDING_COLS = ['user_id', 'district_id', 'times', 'timediff_list']
CONTINUOUS_COLUMNS = [
    'gender', 'avg_price', 'is_supervip', 'ctr_30', 'ord_30', 'total_amt_30',
    'rank_7', 'rank_30', 'rank_90', 'hours'
]
CONTINUOUS_HISTORY_COLUMNS = ['price_list', 'hours_list']
TYPE_COLS = ['time_type', 'time_type_list']
TYPE_LIST = ['lunch', 'night', 'dinner', 'tea', 'breakfast']

HASH_BUCKET_SIZES = 10000
EMBEDDING_DIMENSIONS = 16


class DeepFM():

    def __init__(self,
                 wide_column=None,
                 fm_column=None,
                 deep_column=None,
                 dnn_hidden_units=[512, 128, 32],
                 final_hidden_units=[128, 64],
                 optimizer_type='adam',
                 learning_rate=0.001,
                 inputs=None,
                 use_bn=True,
                 bf16=False,
                 stock_tf=None,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        self._feature = inputs[0]
        self._label = inputs[1]

        self._wide_column = wide_column
        self._deep_column = deep_column
        self._fm_column = fm_column
        if not wide_column or not fm_column or not deep_column:
            raise ValueError(
                'Wide column, FM column or Deep column is not defined.')

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True
        self.use_bn = use_bn

        self._dnn_hidden_units = dnn_hidden_units
        self._final_hidden_units = final_hidden_units
        self._optimizer_type = optimizer_type
        self._learning_rate = learning_rate
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner

        self._create_model()
        with tf.name_scope('head'):
            self._create_loss()
            self._create_optimizer()
            self._create_metrics()

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    def _dnn(self, dnn_input, dnn_hidden_units=None, layer_name=''):
        for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with tf.variable_scope(layer_name + '_%d' % layer_id,
                                   partitioner=self._dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE) as dnn_layer_scope:
                dnn_input = tf.layers.dense(dnn_input,
                                            units=num_hidden_units,
                                            activation=tf.nn.relu,
                                            name=dnn_layer_scope)
                if self.use_bn:
                    dnn_input = tf.layers.batch_normalization(
                        dnn_input, training=self.is_training, trainable=True)
                self._add_layer_summary(dnn_input, dnn_layer_scope.name)

        return dnn_input

    def _create_model(self):
        # input features
        with tf.variable_scope('input_layer',
                               partitioner=self._input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            for key in HISTORY_COLUMN:
                self._feature[key] = tf.strings.split(self._feature[key], ';')
            for key in CONTINUOUS_HISTORY_COLUMNS:
                length = fc_utils.sequence_length_from_sparse_tensor(
                    self._feature[key])
                length = tf.expand_dims(length, -1)
                self._feature[key] = tf.sparse.to_dense(self._feature[key],
                                                        default_value='0')
                self._feature[key] = tf.strings.to_number(self._feature[key])
                self._feature[key] = tf.reduce_sum(self._feature[key], 1, True)
                self._feature[key] = tf.math.divide(
                    self._feature[key], tf.cast(length, tf.float32))

            fm_cols = {}
            dnn_input = tf.feature_column.input_layer(
                self._feature,
                self._deep_column,
                cols_to_output_tensors=fm_cols)
            wide_input = tf.feature_column.input_layer(self._feature,
                                                       self._wide_column)

            fm_input = tf.stack([fm_cols[cols] for cols in self._fm_column], 1)

        if self.bf16:
            wide_input = tf.cast(wide_input, dtype=tf.bfloat16)
            fm_input = tf.cast(fm_input, dtype=tf.bfloat16)
            dnn_input = tf.cast(dnn_input, dtype=tf.bfloat16)

        # DNN part
        dnn_scope = tf.variable_scope('dnn')
        with dnn_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else dnn_scope:
            dnn_output = self._dnn(dnn_input, self._dnn_hidden_units,
                                   'dnn_layer')

        # linear / fisrt order part
        with tf.variable_scope('linear', reuse=tf.AUTO_REUSE) as linear:
            linear_output = tf.reduce_sum(wide_input, axis=1, keepdims=True)

        # FM second order part
        with tf.variable_scope('fm', reuse=tf.AUTO_REUSE) as fm:
            sum_square = tf.square(tf.reduce_sum(fm_input, axis=1))
            square_sum = tf.reduce_sum(tf.square(fm_input), axis=1)
            fm_output = 0.5 * tf.subtract(sum_square, square_sum)

        # Final dnn layer
        all_input = tf.concat([dnn_output, linear_output, fm_output], 1)
        final_dnn_scope = tf.variable_scope('final_dnn')
        with final_dnn_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else final_dnn_scope:
            dnn_logits = self._dnn(all_input, self._final_hidden_units,
                                   'final_dnn')

        if self.bf16:
            dnn_logits = tf.cast(dnn_logits, dtype=tf.float32)

        self._logits = tf.layers.dense(dnn_logits, units=1)
        self.probability = tf.math.sigmoid(self._logits)
        self.output = tf.round(self.probability)

    # compute loss
    def _create_loss(self):
        loss_func = tf.losses.mean_squared_error
        predict = tf.squeeze(self.probability)
        self.loss = tf.math.reduce_mean(loss_func(self._label, predict))
        tf.summary.scalar('loss', self.loss)

    # define optimizer and generate train_op
    def _create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        if self.tf or self._optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._learning_rate,
                initial_accumulator_value=1e-8)
        elif self._optimizer_type == 'adamasync':
            optimizer = tf.train.AdamAsyncOptimizer(
                learning_rate=self._learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagraddecay':
            optimizer = tf.train.AdagradDecayOptimizer(
                learning_rate=self._learning_rate,
                global_step=self.global_step)
        else:
            raise ValueError('Optimzier type error.')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss,
                                               global_step=self.global_step)

    # compute acc & auc
    def _create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self._label,
                                                    predictions=self.output)
        self.auc, self.auc_op = tf.metrics.auc(labels=self._label,
                                               predictions=self.probability,
                                               num_thresholds=1000)
        tf.summary.scalar('eval_acc', self.acc)
        tf.summary.scalar('eval_auc', self.auc)


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):

    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        column_headers = TRAIN_DATA_COLUMNS
        columns = tf.io.decode_csv(value, record_defaults=DEFAULTS)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    files = filename
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(files)
    dataset = dataset.shuffle(buffer_size=20000,
                              seed=args.seed)  # fix seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(2)
    return dataset


def build_feature_columns():
    deep_columns = []
    wide_columns = []
    fm_columns = []
    for columns in SHARE_EMBEDDING_COLS:
        cate_cols = []
        for col in columns:
            cate_col = tf.feature_column.categorical_column_with_hash_bucket(
                col, HASH_BUCKET_SIZES)
            cate_cols.append(cate_col)
            if col not in HISTORY_COLUMN:
                wide_columns.append(
                    tf.feature_column.indicator_column(cate_col))
        shared_emb_cols = tf.feature_column.shared_embedding_columns(
            cate_cols, EMBEDDING_DIMENSIONS)
        deep_columns.extend(shared_emb_cols)
        fm_columns.extend(shared_emb_cols)

    for column in EMBEDDING_COLS:
        cate_col = tf.feature_column.categorical_column_with_hash_bucket(
            column, HASH_BUCKET_SIZES)
        wide_columns.append(tf.feature_column.indicator_column(cate_col))

        if args.tf or not args.emb_fusion:
            emb_col = tf.feature_column.embedding_column(
                cate_col, EMBEDDING_DIMENSIONS)
        else:
            emb_col = tf.feature_column.embedding_column(
                cate_col, EMBEDDING_DIMENSIONS, do_fusion=args.emb_fusion)
        deep_columns.append(emb_col)
        fm_columns.append(emb_col)

    for column in CONTINUOUS_COLUMNS + CONTINUOUS_HISTORY_COLUMNS:
        num_column = tf.feature_column.numeric_column(column)
        wide_columns.append(num_column)
        deep_columns.append(num_column)

    for column in TYPE_COLS:
        cate_col = tf.feature_column.categorical_column_with_vocabulary_list(
            column, TYPE_LIST)
        if col not in HISTORY_COLUMN:
            wide_columns.append(tf.feature_column.indicator_column(cate_col))
        if args.tf or not args.emb_fusion:
            emb_col = tf.feature_column.embedding_column(
                cate_col, EMBEDDING_DIMENSIONS)
        else:
            emb_col = tf.feature_column.embedding_column(
                cate_col, EMBEDDING_DIMENSIONS, do_fusion=args.emb_fusion)
        deep_columns.append(emb_col)
        fm_columns.append(emb_col)

    return wide_columns, fm_columns, deep_columns


def train(sess_config,
          input_hooks,
          model,
          data_init_op,
          steps,
          checkpoint_dir,
          tf_config=None,
          server=None):
    model.is_training = True
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.tables_initializer(),
                               tf.local_variables_initializer(), data_init_op),
        saver=tf.train.Saver(max_to_keep=args.keep_checkpoint_max))

    stop_hook = tf.train.StopAtStepHook(last_step=steps)
    log_hook = tf.train.LoggingTensorHook(
        {
            'steps': model.global_step,
            'loss': model.loss
        }, every_n_iter=100)
    hooks.append(stop_hook)
    hooks.append(log_hook)
    if args.timeline > 0:
        hooks.append(
            tf.train.ProfilerHook(save_steps=args.timeline,
                                  output_dir=checkpoint_dir))
    save_steps = args.save_steps if args.save_steps or args.no_eval else steps

    time_start = time.perf_counter()
    with tf.train.MonitoredTrainingSession(
            master=server.target if server else '',
            is_chief=tf_config['is_chief'] if tf_config else True,
            hooks=hooks,
            scaffold=scaffold,
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=steps,
            summary_dir=checkpoint_dir,
            save_summaries_steps=None,
            config=sess_config) as sess:
        while not sess.should_stop():
            sess.run([model.loss, model.train_op])
    time_end = time.perf_counter();
    print("Training completed.")
    time_cost = time_end - time_start;
    global global_time_cost
    global_time_cost = time_cost


def eval(sess_config, input_hooks, model, data_init_op, steps, checkpoint_dir):
    model.is_training = False
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.tables_initializer(),
                               tf.local_variables_initializer(), data_init_op))
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
    writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval'))
    merged = tf.summary.merge_all()

    with tf.train.MonitoredSession(session_creator=session_creator,
                                   hooks=hooks) as sess:
        for _in in range(1, steps + 1):
            if (_in != steps):
                sess.run([model.acc_op, model.auc_op])
                if (_in % 100 == 0):
                    print("Evaluation complate:[{}/{}]".format(_in, steps))
            else:
                eval_acc, eval_auc, events = sess.run(
                    [model.acc_op, model.auc_op, merged])
                writer.add_summary(events, _in)
                print("Evaluation complate:[{}/{}]".format(_in, steps))
                print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))
                global global_auc
                global_auc = eval_auc


def main(tf_config=None, server=None):
    # check dataset
    print("Checking dataset...")
    train_file = os.path.join(args.data_location, 'train.csv')
    test_file = os.path.join(args.data_location, 'eval.csv')
    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print("Dataset does not exist in the given data_location.")
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))

    # set batch size, eporch & steps
    batch_size = args.batch_size

    if args.steps == 0:
        no_of_epochs = 1
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)
    print("The training steps is {}".format(train_steps))
    print("The testing steps is {}".format(test_steps))

    # set fixed random seed
    tf.set_random_seed(args.seed)

    # set directory path
    model_dir = os.path.join(args.output_dir,
                             'model_DeepFM_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
    test_dataset = build_model_input(test_file, batch_size, 1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create feature column
    wide_column, fm_column, deep_column = build_feature_columns()

    # Session config
    sess_config = tf.ConfigProto()
    # Session hooks
    hooks = []

    if args.smartstaged and not args.tf:
        '''Smart staged Feature'''
        next_element = tf.staged(next_element, num_threads=4, capacity=40)
        sess_config.graph_options.optimizer_options.do_smart_stage = True
        hooks.append(tf.make_prefetch_hook())
    if args.op_fusion and not args.tf:
        '''Auto Graph Fusion'''
        sess_config.graph_options.optimizer_options.do_op_fusion = True

    # create model
    model = DeepFM(wide_column=wide_column,
                   fm_column=fm_column,
                   deep_column=deep_column,
                   optimizer_type=args.optimizer,
                   learning_rate=args.learning_rate,
                   bf16=args.bf16,
                   stock_tf=args.tf,
                   inputs=next_element)

    # Run model training and evaluation
    train(sess_config, hooks, model, train_init_op, train_steps,
          checkpoint_dir, tf_config, server)
    if not (args.no_eval or tf_config):
        eval(sess_config, hooks, model, test_init_op, test_steps,
             checkpoint_dir)
    os.makedirs(result_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        f.write(str(global_time_cost)+'\n')
        f.write(str(global_auc)+'\n')


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to model output directory. \
                            Default to ./result. Covered by --checkpoint. ',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output. \
                            Default to ./result/$MODEL_TIMESTAMP',
                        required=False)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)
    parser.add_argument('--optimizer', \
                        type=str,
                        choices=['adam', 'adamasync', 'adagraddecay', 'adagrad'],
                        default='adamasync')
    parser.add_argument('--learning_rate',
                        help='Learning rate for deep model',
                        type=float,
                        default=0.001)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc', 'grpc++', 'star_server'],
                        default='grpc')
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument('--tf', \
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    parser.add_argument('--smartstaged', \
                        help='Whether to enable smart staged feature of DeepRec, Default to True.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--emb_fusion', \
                        help='Whether to enable embedding fusion, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--op_fusion', \
                        help='Whether to enable Auto graph fusion feature. Default to True',
                        type=boolean_string,
                        default=True)
    return parser


# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    print(TF_CONFIG)
    tf_config = json.loads(TF_CONFIG)
    cluster_config = tf_config.get('cluster')
    ps_hosts = []
    worker_hosts = []
    chief_hosts = []
    for key, value in cluster_config.items():
        if 'ps' == key:
            ps_hosts = value
        elif 'worker' == key:
            worker_hosts = value
        elif 'chief' == key:
            chief_hosts = value
    if chief_hosts:
        worker_hosts = chief_hosts + worker_hosts

    if not ps_hosts or not worker_hosts:
        print('TF_CONFIG ERROR')
        sys.exit()
    task_config = tf_config.get('task')
    task_type = task_config.get('type')
    task_index = task_config.get('index') + (1 if task_type == 'worker'
                                             and chief_hosts else 0)

    if task_type == 'chief':
        task_type = 'worker'

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.distribute.Server(cluster,
                                  job_name=task_type,
                                  task_index=task_index,
                                  protocol=args.protocol)
    if task_type == 'ps':
        server.join()
    elif task_type == 'worker':
        tf_config = {
            'ps_hosts': ps_hosts,
            'worker_hosts': worker_hosts,
            'type': task_type,
            'index': task_index,
            'is_chief': is_chief
        }
        tf_device = tf.device(
            tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % task_index,
                cluster=cluster))
        return tf_config, server, tf_device
    else:
        print("Task type or index error.")
        sys.exit()


# Some DeepRec's features are enabled by ENV.
# This func is used to set ENV and enable these features.
# A triple quotes comment is used to introduce these features and play an emphasizing role.
def set_env_for_DeepRec():
    '''
    Set some ENV for these DeepRec's features enabled by ENV. 
    More Detail information is shown in https://deeprec.readthedocs.io/zh/latest/index.html.
    START_STATISTIC_STEP & STOP_STATISTIC_STEP: On CPU platform, DeepRec supports memory optimization
        in both stand-alone and distributed trainging. It's default to open, and the 
        default start and stop steps of collection is 1000 and 1100. Reduce the initial 
        cold start time by the following settings.
    MALLOC_CONF: On CPU platform, DeepRec can use memory optimization with the jemalloc library.
        Please preload libjemalloc.so by `LD_PRELOAD=./libjemalloc.so.2 python ...`
    '''
    os.environ['START_STATISTIC_STEP'] = '100'
    os.environ['STOP_STATISTIC_STEP'] = '110'
    os.environ['MALLOC_CONF'] = \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'
    os.environ['ENABLE_MEMORY_OPTIMIZATION'] = '0'


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    if not args.tf:
        set_env_for_DeepRec()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        main()
    else:
        tf_config, server, tf_device = generate_cluster_info(TF_CONFIG)
        main(tf_config, server)