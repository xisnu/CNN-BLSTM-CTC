from __future__ import print_function
import tensorflow as tf
from ReadData import *
import time,sys

class CNNBLSTM:

    def __init__(self,max_timesteps,nb_features,nb_classes,modelname):
        self.hybridmodel=tf.Graph()
        self.max_timesteps=max_timesteps
        self.nb_features=nb_features
        self.nb_classes=nb_classes
        self.model_name=modelname
        print("Empty Graph Created, Number of Class=",nb_classes," Number of Features=",nb_features)

    def get_layer_shape(self, layer):
        thisshape = tf.Tensor.get_shape(layer)
        ts = [thisshape[i].value for i in range(len(thisshape))]
        return ts

    def readNetworkStructure(self,configfile):
        nw = {}
        f = open(configfile)
        line = f.readline()
        while line:
            info = line.strip("\n").split(",")
            nw[info[0]] = info[1]
            line = f.readline()
        self.filterwidth=int(nw['filterwidth'])
        self.nb_filters=[int(fi) for fi in nw['nb_filters'].split()]
        self.conv_stride = [int(fi) for fi in nw['conv_stride'].split()]
        self.pool_stride = [int(fi) for fi in nw['pool_stride'].split()]
        self.nb_hidden=int(nw['nb_hidden'])
        self.lr=float(nw['lr'])
        print('Network Configuration Understood')

    def createNetwork(self,configfile):
        self.readNetworkStructure(configfile)
        with self.hybridmodel.as_default():
            # tf.reset_default_graph()
            self.network_input_x = tf.placeholder(tf.float32, [None, 1, self.max_timesteps, self.nb_features])
            self.network_target_y = tf.sparse_placeholder(tf.int32)
            self.network_input_sequence_length = tf.placeholder(tf.int32, [None])
            #seq_len2 = tf.placeholder(tf.int32, [None])

            f1 = [1, self.filterwidth, self.nb_features, self.nb_filters[0]]
            W1 = tf.Variable(tf.truncated_normal(f1, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[self.nb_filters[0]]), name="b1")
            conv_1 = tf.nn.conv2d(self.network_input_x, W1, strides=[1, 1, self.conv_stride[0], 1], padding='SAME')
            nl_1 = tf.nn.relu(tf.add(conv_1, b1))  # batchsize,1,maxsteps,filter1
            mp_1 = tf.nn.max_pool(nl_1, ksize=[1, 1, 7, 1], strides=[1, 1, self.pool_stride[0], 1],padding='SAME')  # batchsize,1,maxsteps,filter1
            shape=self.get_layer_shape(mp_1)
            print("First Conv Block = ",shape)
            # ---------------1st Conv MP Block Ends----------------------#

            # ---------------2nd Block Starts---------------------#
            f2 = [1, self.filterwidth, self.nb_filters[0], self.nb_filters[1]]
            W2 = tf.Variable(tf.truncated_normal(f2, stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[self.nb_filters[1]]), name="b2")
            conv_2 = tf.nn.conv2d(mp_1, W2, strides=[1, 1, self.conv_stride[1], 1], padding='SAME')
            nl_2 = tf.nn.relu(tf.add(conv_2, b2))  # batchsize,1,maxsteps,filter2
            mp_2 = tf.nn.max_pool(nl_2, ksize=[1, 1, 5, 1], strides=[1, 1, self.pool_stride[1], 1],padding='SAME')  # batchsize,1,maxsteps,filter2
            shape = self.get_layer_shape(mp_2)
            print("Second Conv Block = ", shape)
            # --------------2nd Conv MP Block Ends-----------------------#


            # ---------------3rd Block Starts---------------------#
            f3 = [1, self.filterwidth, self.nb_filters[1], self.nb_filters[2]]
            W3 = tf.Variable(tf.truncated_normal(f3, stddev=0.1), name="W3")
            b3 = tf.Variable(tf.constant(0.1, shape=[self.nb_filters[2]]), name="b3")
            conv_3 = tf.nn.conv2d(mp_2, W3, strides=[1, 1, self.conv_stride[2], 1], padding='SAME')
            nl_3 = tf.nn.relu(tf.add(conv_3, b3))  # batchsize,1,maxsteps,filter2
            mp_3 = tf.nn.max_pool(nl_3, ksize=[1, 1, 5, 1], strides=[1, 1, self.pool_stride[2], 1],padding='SAME')  # batchsize,1,maxsteps,filter3
            shape = self.get_layer_shape(mp_3)
            print("Third Conv Block = ", shape)
            # --------------3rd Conv MP Block Ends-----------------------#


            conv_reshape = tf.squeeze(mp_3, squeeze_dims=[1])  # batchsize,maxsteps,filter3
            shape = self.get_layer_shape(conv_reshape)
            print("CNN --> RNN Reshape = ", shape)

            with tf.variable_scope("cell_def_1"):
                f_cell = tf.nn.rnn_cell.LSTMCell(self.nb_hidden, state_is_tuple=True)
                b_cell = tf.nn.rnn_cell.LSTMCell(self.nb_hidden, state_is_tuple=True)

            with tf.variable_scope("cell_op_1"):
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, conv_reshape, sequence_length=self.network_input_sequence_length,dtype=tf.float32)

            merge = tf.concat(2, outputs)
            shape = self.get_layer_shape(merge)
            print("First BLSTM = ", shape)

            nb_hidden_2=self.nb_hidden*2

            with tf.variable_scope("cell_def_2"):
                f1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden_2, state_is_tuple=True)
                b1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden_2, state_is_tuple=True)

            with tf.variable_scope("cell_op_2"):
                outputs2, _ = tf.nn.bidirectional_dynamic_rnn(f1_cell, b1_cell, merge, sequence_length=self.network_input_sequence_length, dtype=tf.float32)

            merge2 = tf.concat(2, outputs2)
            shape = self.get_layer_shape(merge2)
            print("Second BLSTM = ", shape)
            batch_s, timesteps = shape[0], shape[1]
            print(timesteps)

            blstm_features=shape[-1]


            output_reshape = tf.reshape(merge2, [-1, blstm_features])  # maxsteps*batchsize,nb_hidden
            shape = self.get_layer_shape(output_reshape)
            print("RNN Time Squeezed = ", shape)

            W = tf.Variable(tf.truncated_normal([blstm_features, self.nb_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0., shape=[self.nb_classes]), name="b")

            logits = tf.matmul(output_reshape, W) + b  # maxsteps*batchsize,nb_classes
            logits=tf.reshape(logits, [-1, timesteps, self.nb_classes])
            shape = self.get_layer_shape(logits)
            print("Logits = ", shape)

            logits_reshape = tf.transpose(logits,[1, 0, 2])  # maxsteps,batchsize,nb_classes
            shape = self.get_layer_shape(logits_reshape)
            print("RNN Time Distributed (Time Major) = ", shape)

            loss = tf.nn.ctc_loss(logits_reshape, self.network_target_y, self.network_input_sequence_length)
            self.cost = tf.reduce_mean(loss)

            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.cost)

            # for greedy decoder input(i.e. logits_reshape) must be of shape maxtime,batchsize,nb_classes
            # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits_reshape, seq_len)--very slow
            decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_reshape, self.network_input_sequence_length)

            self.decoded_words=tf.sparse_to_dense(decoded[0].indices,decoded[0].shape,decoded[0].values)
            self.actual_targets=tf.sparse_to_dense(self.network_target_y.indices,self.network_target_y.shape,self.network_target_y.values)

            actual_ed = tf.edit_distance(tf.cast(decoded[0], tf.int32), self.network_target_y, normalize=False)
            self.ler = tf.reduce_sum(actual_ed) #insertion+deletion+substitution
            self.new_saver = tf.train.Saver()
            print("Network Ready")

    def trainNetwork(self,nb_epochs,batchsize,x,y,seqlen,max_target_length,transcription_length,weightfiles,mode):
        x_train=x[0]
        x_test=x[1]
        y_train=y[0]
        y_test=y[1]
        seq_len_train=adjustSequencelengths(seqlen[0],self.conv_stride,self.pool_stride,max_target_length)
        seq_len_test = adjustSequencelengths(seqlen[1],self.conv_stride,self.pool_stride,max_target_length)
        weightfile_last=weightfiles[0]
        weightfile_best = weightfiles[1]
        train_transcription_length=transcription_length[0]
        test_transcription_length=transcription_length[1]

        with tf.Session(graph=self.hybridmodel) as session:
            if(mode=="New"):
                init_op = tf.global_variables_initializer()
                session.run(init_op)
                print("New Weights Initiated")
            elif(mode=="Load"):
                self.new_saver.restore(session, weightfile_best)
                print("Previous weights loaded")
            else:
                print("Unknown Mode")
                return
            nb_train=len(x_train)
            nb_test=len(x_test)
            trainbatch=int(np.ceil(float(nb_train)/batchsize))
            testbatch=int(np.ceil(float(nb_test)/batchsize))
            besttestacc=0
            for e in range(nb_epochs):
                totalloss = 0
                totalacc = 0
                starttime = time.time()
                train_batch_start=0
                logf = open("Training_log", "a")
                for b in range(trainbatch):
                    train_batch_end=min(nb_train,train_batch_start+batchsize)
                    sys.stdout.write("\rTraining Batch %d / %d" %(b,trainbatch))
                    sys.stdout.flush()
                    batch_x=pad_x(x_train[train_batch_start:train_batch_end],self.max_timesteps,self.nb_features)
                    batch_seq_len=seq_len_train[train_batch_start:train_batch_end]
                    batch_target_sparse=y_train[b]

                    feed = {self.network_input_x: batch_x, self.network_target_y: batch_target_sparse, self.network_input_sequence_length: batch_seq_len}

                    batchloss, batchacc, _ = session.run([self.cost, self.ler, self.optimizer], feed)

                    totalloss = totalloss + batchloss
                    totalacc = totalacc + batchacc
                    train_batch_start=train_batch_end

                trainloss = totalloss / trainbatch
                #avgacc = totalacc / trainbatch
                print("\nTraining Edit Distance ",totalacc,"/",train_transcription_length)
                trainacc=(1-(float(totalacc)/train_transcription_length))*100
                # Now save the model
                self.new_saver.save(session, weightfile_last)

                testloss = 0
                testacc = 0

                test_batch_start = 0
                output_words=[]
                target_words=[]
                for b in range(testbatch):
                    test_batch_end = min(nb_test, test_batch_start + batchsize)
                    sys.stdout.write("\rTesting Batch %d/%d"%(b,testbatch) )
                    sys.stdout.flush()
                    batch_x = pad_x(x_test[test_batch_start:test_batch_end],self.max_timesteps,self.nb_features)
                    batch_seq_len = seq_len_test[test_batch_start:test_batch_end]
                    batch_target_sparse = y_test[b]

                    testfeed = {self.network_input_x: batch_x, self.network_target_y: batch_target_sparse,self.network_input_sequence_length: batch_seq_len}

                    batchloss, batchacc,output_words_batch,target_words_batch = session.run([self.cost, self.ler,self.decoded_words,self.actual_targets], testfeed)
                    output_words.extend(output_words_batch)
                    target_words.extend(target_words_batch)
                    testloss = testloss + batchloss
                    testacc = testacc + batchacc
                    test_batch_start=test_batch_end

                testloss = testloss / testbatch
                testacc=(1-(float(testacc)/test_transcription_length))*100

                result=open("Decoded","w")
                corrects=0.0
                for w in range(nb_test):
                    if(sum(output_words[w])==sum(target_words[w])):
                        corrects=corrects+1
                        flag="Correct"
                    else:
                        flag="Incorrect"
                    result.write(str(target_words[w]) + "," + str(output_words[w])+","+flag + "\n")
                result.close()

                avg_word_accuracy=(corrects/nb_test)*100

                if (testacc > besttestacc):
                    besttestacc = testacc
                    print("\nNetwork Improvement")
                    self.new_saver.save(session, weightfile_best)
                endtime = time.time()
                timetaken = endtime - starttime
                msg = "\nEpoch " + str(e) + "(" + str(timetaken) + " sec) Training: Loss is " + str(trainloss) + " Accuracy " + str(trainacc) + "% Testing: Loss " + str(testloss) + " Accuracy " + str(testacc) + "% Best " + str(besttestacc) + "%\n"
                print(msg)
                logf.write(msg)
                logf.write("\nWord Accuracy"+str(avg_word_accuracy))
                logf.close()
                msg="Word Accuracy="+str(avg_word_accuracy)
                print(msg)

    def compare_prediction(self,input_data,sequence_length,weightfile,dbfile,max_target_length):
        input_data_y=input_data[1][0]
        print(input_data_y)
        nb_predicts=input_data_y[2][0]
        input_data_x = pad_x(input_data[0][:nb_predicts], self.max_timesteps, self.nb_features)
        sequence_length=sequence_length[:nb_predicts]
        sequence_length = adjustSequencelengths(sequence_length, self.conv_stride, self.pool_stride, max_target_length)
        with tf.Session(graph=self.hybridmodel) as predict_session:
            self.new_saver.restore(predict_session,weightfile)
            print("Saved Model Loaded")
            feed={self.network_input_x:input_data_x,self.network_input_sequence_length:sequence_length,self.network_target_y:input_data_y}
            output_words,actual_targets=predict_session.run([self.decoded_words,self.actual_targets],feed)
            #print(output_words)
            predicted_words=[]
            actual_words=[]
            for i in range(nb_predicts):
                #print(actual_targets[i],output_words[i])
                unicode_output, _ = int_to_unicode(output_words[i], "Character_Integer",dbfile)
                unicode_output = reset_unicode_order(unicode_output,dbfile)
                predicted_words.append(unicode_output)

                uo, _ = int_to_unicode(actual_targets[i], "Character_Integer", dbfile)
                uo = reset_unicode_order(uo, dbfile)
                actual_words.append(uo)

            f=open("Predicted","w")
            for i in range(nb_predicts):
                f.write(actual_words[i]+","+predicted_words[i]+"\n")
            f.close()
            print("Output Ready")
            return [predicted_words]

'''
Model should be fed with
Max_Time_steps,nb_features,nb_classes
Rest of the parameters written in Config file
'''
dbfile="bengalichardb.txt"
path="Data"
files=[path+"/Train_feat_Graves",path+"/Test_feat_Graves"]
weightfile="Weights"
batchsize=512
#To start a fresh training set runmode="New", for prediction and resume training set it to "Load"
runmode="New"
if(runmode=="Load"):
    generate=True
else:
    generate=False


[x_train,x_test],nb_classes,[train_seq_len,test_seq_len],[train_y,test_y],max_target_length,max_seq_length,char_int,transcription_length=load_data(files[0],files[1],batchsize,"nopad",generate)
nb_features=len(x_train[0][0])
x=[x_train,x_test]
y=[train_y,test_y]
seqlen=[train_seq_len,test_seq_len]
print("Training Data X=",len(x_train)," Testing Data X=",len(x_test)," Max Seq len=",max_seq_length," Max Target length=",max_target_length)

model=CNNBLSTM(max_seq_length,nb_features,nb_classes,"Hybrid")
model.createNetwork("Config")


weightfile_last=weightfile+"/last"
weightfile_best=weightfile+"/Best/best"
weightfiles=[weightfile_last,weightfile_best]

model.trainNetwork(5,batchsize,x,y,seqlen,max_target_length,transcription_length,weightfiles,runmode)
model.compare_prediction([x_test,test_y],test_seq_len,weightfile_best,dbfile,max_target_length)#A new file Predicted will show predictions from network
