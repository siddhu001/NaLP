import tensorflow as tf
import numpy as np
np.random.seed(1234)
import os
import pickle
from log import Logger
from batching import *
from model import NaLP
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", dest="data_dir", type=str, help="The data dir.", default='./data')
parser.add_argument("--sub_dir", dest="sub_dir", type=str, help="The sub data dir.", default="WikiPeople")
parser.add_argument("--dataset_name", dest="dataset_name", type=str, help="The name of the dataset.", default="WikiPeople")
parser.add_argument("--wholeset_name", dest="wholeset_name", type=str, help="Name of the whole dataset for negative sampling or computing the filtered metrics.", default="WikiPeople_permutate")
parser.add_argument("--model_name", dest="model_name", type=str, help="", default='WikiPeople')
parser.add_argument("--embedding_dim", dest="embedding_dim", type=int, help="The embedding dimension.",  default=100)
parser.add_argument("--n_filters", dest="n_filters", type=int, help="The number of filters.", default=200)
parser.add_argument("--n_gFCN", dest="n_gFCN", type=int, help="The number of hidden units of fully-connected layer in g-FCN.", default=1200)
parser.add_argument("--batch_size", dest="batch_size", type=int, help="The batch size.", default=128)
parser.add_argument("--is_trainable", dest="is_trainable", type=bool, help="", default=True)
parser.add_argument("--learning_rate", dest="learning_rate", type=float, help="The learning rate.", default=0.00005)
parser.add_argument("--n_epochs", dest="n_epochs", type=int, help="The number of training epochs.", default=5000)
parser.add_argument("--if_restart", dest="if_restart", type=bool, help="", default= False)
parser.add_argument("--start_epoch", dest="start_epoch", type=int, help="Change this when restarting", default=0)
parser.add_argument("--saveStep", dest="saveStep", type=int, help="Save the model every saveStep", default=100)
parser.add_argument("--allow_soft_placement", dest="allow_soft_placement", type=bool, help="Allow device soft device placement", default=True)
parser.add_argument("--log_device_placement", dest="log_device_placement", type=bool, help="Log placement of ops on devices", default=False)
parser.add_argument("--run_folder", dest="run_folder", type=str, help="The dir to store models.", default="./")

args = parser.parse_args() 

print("\nParameters:")
print(args) 

# The log file to store the parameters and the training details of each epoch
logger = Logger('logs', 'run_'+args.model_name+'_'+str(args.embedding_dim)+'_'+str(args.n_filters)+'_'+str(args.n_gFCN)+'_'+str(args.batch_size)+'_'+str(args.learning_rate)).logger
logger.info("\nParameters:")

# Load training data
logger.info("Loading data...")
afolder = args.data_dir + '/'
if args.sub_dir != '':
    afolder = args.data_dir + '/' + args.sub_dir + '/'
with open(afolder + args.dataset_name + ".bin", 'rb') as fin:
    data_info = pickle.load(fin)
train = data_info["train_facts"]
values_indexes = data_info['values_indexes']
roles_indexes = data_info['roles_indexes']
role_val = data_info['role_val']
value_array = np.array(list(values_indexes.values()))
role_array = np.array(list(roles_indexes.values()))

# Load the whole dataset for negative sampling in "batching.py"
with open(afolder + args.wholeset_name + ".bin", 'rb') as fin:
    data_info1 = pickle.load(fin)
whole_train = data_info1["train_facts"]
logger.info("Loading data... finished!")

with tf.Graph().as_default():
    tf.set_random_seed(1234)
    session_conf = tf.ConfigProto(allow_soft_placement=args.allow_soft_placement, log_device_placement=args.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        aNaLP = NaLP(
            n_values=len(values_indexes),
            n_roles=len(roles_indexes),
            embedding_dim=args.embedding_dim,
            n_filters=args.n_filters,
            n_gFCN=args.n_gFCN,
            batch_size=args.batch_size*2,
            is_trainable=args.is_trainable)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        grads_and_vars = optimizer.compute_gradients(aNaLP.loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        
        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(args.run_folder, "runs", args.model_name))
        logger.info("Writing to {}\n".format(out_dir))

        # Train Summaries
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
   
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch, arity):
            """
            A single training step
            """
            feed_dict = {
              aNaLP.input_x: x_batch,
              aNaLP.input_y: y_batch,
              aNaLP.arity: arity
            }
            _, loss = sess.run([train_op, aNaLP.loss], feed_dict)
            return loss
        
        # If restart, then load the model
        if args.if_restart == True:
            _file = checkpoint_prefix + "-" + str(args.start_epoch)
            aNaLP.saver.restore(sess, _file)

        # Training
        n_batches_per_epoch = []
        for i in train:
            ll = len(i)
            if ll == 0:
                n_batches_per_epoch.append(0)
            else:
                n_batches_per_epoch.append(int((ll - 1) / args.batch_size) + 1)
        for epoch in range(args.start_epoch, args.n_epochs):
            train_loss = 0
            for i in range(len(train)):
                for batch_num in range(n_batches_per_epoch[i]):
                    arity = i + 2  # 2-ary in index 0
                    x_batch, y_batch = Batch_Loader(train[i], values_indexes, roles_indexes, role_val, args.batch_size, arity, whole_train[i])
                    tmp_loss = train_step(x_batch, y_batch, arity)
                    train_loss = train_loss + tmp_loss
                
            logger.info("nepoch: "+str(epoch+1)+", trainloss: "+str(train_loss))
            if (epoch+1) % args.saveStep == 0:
                path = aNaLP.saver.save(sess, checkpoint_prefix, global_step=epoch+1)
                logger.info("Saved model checkpoint to {}\n".format(path))
        train_summary_writer.close
