# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:57:32 2016

@author: parthosarothi
"""
from __future__ import print_function
import sys,os
import tensorflow as tf
import math
from Dataset import *
import time

nb_hidden=64
lr=0.001
nb_epochs=5000
filterwidth=5
nb_filters1=16
nb_filters2=32
nb_filters3=64
nb_dense=100
batchsize=1024
conv_stride=[3,2,2]
poolstride=[2,1,1]
nblayers=3
cpath=os.getcwd()
filenames=["Data/ICBOHR-04_Gist_feat","Data/ICBOHR-04_Gist_feat"]
logfilename="Logs/log_"+str(time.strftime("%d-%m-%y_%H-%M",time.gmtime()))+".txt"
[inpx,inp_sparse_y,inpseqlen,trainbatch,nbtrain],[test_inpx,test_inp_sparse_y,test_inpseqlen,testbatch,nbtest],chars,nctr,ncts=loaddata(filenames,batchsize,"CNN",cpath)

#Modify sequence lengths due to effect of max pooling

for l in range(nblayers):
    for i in range(len(inpseqlen)):
        for j in range(len(inpseqlen[i])):
            inpseqlen[i][j]=math.ceil(inpseqlen[i][j]/(conv_stride[l]*poolstride[l]))

    for i in range(len(test_inpseqlen)):
        for j in range(len(test_inpseqlen[i])):
            test_inpseqlen[i][j]=math.ceil(test_inpseqlen[i][j]/(conv_stride[l]*poolstride[l]))

#Modification complete

f=open(logfilename,"w")
f.write("Train-"+filenames[0]+"\n")
f.write("Test-"+filenames[1]+"\n")
msg="Shape info X_Train="+str(inpx[0].shape)+" X_Test="+str(test_inpx[0].shape)
print(msg)
f.write(msg+"\n")

msg="Training samples "+str(nbtrain)+" testing samples "+str(nbtest)
print(msg)
f.write(msg+"\n")

msg="Training Batches "+str(trainbatch)+" Testing Batches "+str(testbatch)
print(msg)
f.write(msg+"\n")

msg="Total Number of characters in Train "+str(nctr)+" Test "+str(ncts)
print(msg)
f.write(msg+"\n")

msg="Distinct Characters "+str(len(chars))
print(msg)
f.write(msg+"\n")
f.close()

print("Data Prepared. Log in ",logfilename)
nb_classes=len(chars)
nb_features=inpx[0].shape[3]
ms=inpx[0].shape[2]
trainbatch=len(inpx)

thisgraph=tf.Graph()

with thisgraph.as_default():
    #tf.reset_default_graph()
    x=tf.placeholder(tf.float32,[None,1,ms,nb_features])
    global_step = tf.Variable(0, trainable=False)
    y=tf.sparse_placeholder(tf.int32)
    seq_len=tf.placeholder(tf.int64,[None])
    seq_len2=tf.placeholder(tf.int32,[None])

    f1=[1,filterwidth,nb_features,nb_filters1]
    W1 = tf.Variable(tf.truncated_normal(f1, stddev=0.1), name="W1")
    #W1 = tf.Variable(df,trainable=False, name="W1")
    b1 = tf.Variable(tf.constant(0.1, shape=[nb_filters1]), name="b1")
    conv_1=tf.nn.conv2d(x,W1,strides=[1, 1, conv_stride[0], 1], padding='SAME')
    nl_1=tf.nn.relu(tf.add(conv_1,b1))#batchsize,1,maxsteps,filter1
    mp_1=tf.nn.max_pool(nl_1,ksize=[1,1,7,1],strides=[1, 1, poolstride[0], 1], padding='SAME')#batchsize,1,maxsteps,filter1
    
    #---------------1st Block Ends----------------------#

    #---------------2nd Block Starts---------------------#
    f2 = [1, filterwidth, nb_filters1, nb_filters2]
    W2 = tf.Variable(tf.truncated_normal(f2, stddev=0.1), name="W2")
    b2 = tf.Variable(tf.constant(0.1, shape=[nb_filters2]), name="b2")
    conv_2 = tf.nn.conv2d(mp_1, W2, strides=[1, 1, conv_stride[1], 1], padding='SAME')
    nl_2 = tf.nn.relu(tf.add(conv_2, b2))  # batchsize,1,maxsteps,filter2
    mp_2 = tf.nn.max_pool(nl_2, ksize=[1, 1, 5, 1], strides=[1, 1, poolstride[1], 1],padding='SAME')  # batchsize,1,maxsteps,filter2
    # --------------2nd Block Ends-----------------------#

    conv_reshape = tf.squeeze(mp_2, squeeze_dims=[1])  # batchsize,maxsteps,filter3

    with tf.variable_scope("cell_def_1"):
        f_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
        b_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)

    with tf.variable_scope("cell_op_1"):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, conv_reshape, sequence_length=seq_len, dtype=tf.float32)

    merge=tf.concat(2, outputs)

    with tf.variable_scope("cell_def_2"):
        f1_cell=tf.nn.rnn_cell.LSTMCell(nb_hidden*2,state_is_tuple=True)
        b1_cell=tf.nn.rnn_cell.LSTMCell(nb_hidden*2,state_is_tuple=True)
    
    with tf.variable_scope("cell_op_2"):
        outputs2,_=tf.nn.bidirectional_dynamic_rnn(f1_cell,b1_cell,merge,sequence_length=seq_len,dtype=tf.float32)

    merge2=tf.concat(2, outputs2)

    shape = tf.shape(x)
    batch_s,maxtimesteps=shape[0],shape[2]

    output_reshape = tf.reshape(merge2, [-1, nb_hidden*2])#maxsteps*batchsize,nb_hidden

    
    W = tf.Variable(tf.truncated_normal([nb_hidden*2,nb_classes],stddev=0.1),name="W")

    b = tf.Variable(tf.constant(0., shape=[nb_classes]),name="b")
    
    logits = tf.matmul(output_reshape, W) + b #maxsteps*batchsize,nb_classes
    
    
    logits_reshape = tf.transpose(tf.reshape(logits, [batch_s, -1, nb_classes]),[1,0,2])#maxsteps,batchsize,nb_classes

    loss =tf.nn.ctc_loss(logits_reshape, y, seq_len2)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.RMSPropOptimizer(lr).minimize(cost,global_step=global_step)

    #for greedy decoder input(i.e. logits_reshape) must be of shape maxtime,batchsize,nb_classes
    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits_reshape, seq_len)--very slow
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_reshape, seq_len2)

    actual_ed=tf.edit_distance(tf.cast(decoded[0], tf.int32),y,normalize=False)
    ler = tf.reduce_sum(actual_ed)
    new_saver = tf.train.Saver()
    print("Network Ready")

with tf.Session(graph=thisgraph) as session:    

    if(sys.argv[1]=="load"):
        new_saver.restore(session,"Weights/model_last-298")
        print("Previous weights loaded")
    else:
        init_op = tf.initialize_all_variables()
        session.run(init_op)
        print("New Weights Initialized")
    
    best=0
    besttestacc=0
    #testfeed = {x:test_inpx[0],y:test_inp_sparse_y[0],seq_len:test_inpseqlen[0]}
    testcases=[0,5,17,39,60]
    true=[]
    for tr in range(len(testcases)):
        true1=label_from_sparse(test_inp_sparse_y[0],testcases[tr])
        true.append(true1)    
    print("Actual ",true)
    #print("Befor Training starts Weights ",session.run(W.value()[0][0][0]))
    
    for e in range(nb_epochs):
        f=open(logfilename,"a")
        totalloss=0
        totalacc=0
        starttime=time.time()
        for b in range(trainbatch):
            p=b+1
            print("Reading Batch ",p,"/",trainbatch,end="\r")
            feed = {x:inpx[b],y:inp_sparse_y[b],seq_len:inpseqlen[b],seq_len2:inpseqlen[b]}
            if(e==0)and(b==0):
                c1,m1,c2,m2,crs,mrg,mrg2=session.run([conv_1,mp_1,conv_2,mp_2,conv_reshape,merge,merge2],feed)

                f.write("Conv 1 "+str(c1.shape)+"\n")
                f.write("MP 1 "+str(m1.shape)+"\n")
                f.write("Conv 2 "+str(c2.shape)+"\n")
                f.write("MP 2 "+str(m2.shape)+"\n")
                f.write("CNN Outcome Reshaped " + str(crs.shape) + "\n")
                f.write("BLSTM1 " + str(mrg.shape) + "\n")
                f.write("CNN Outcome Reshaped " + str(crs.shape) + "\n")

                
                rnrs,log,td=session.run([output_reshape,logits,logits_reshape],feed)
                
                f.write("Block Outcome Reshaped "+str(rnrs.shape)+"\n")
                #f.write("Fully Connected "+str(d.shape)+"\n")
                f.write("Logits "+str(log.shape)+"\n")
                f.write("Time Distributed "+str(td.shape)+"\n")
                
            batchloss,batchacc,_,steps= session.run([cost,ler,optimizer,global_step], feed)
            totalloss=totalloss+batchloss
            totalacc=totalacc+batchacc


        avgloss=totalloss/trainbatch
        avgacc=1-(totalacc/nctr)
        
        testloss=0
        testacc=0

        for t in range(testbatch):
            testfeed = {x:test_inpx[t],y:test_inp_sparse_y[t],seq_len:test_inpseqlen[t],seq_len2:test_inpseqlen[t]}
            outcome,testbatchloss,testbatchacc=session.run([decoded[0],cost,ler],testfeed)
            if(t==0):
                first_batch_outcome=outcome
            testloss=testloss+testbatchloss
            testacc=testacc+testbatchacc

        testloss=testloss/testbatch
        testacc=1-(testacc/ncts)
        
        if(testacc>besttestacc):
            besttestacc=testacc


        endtime=time.time()        
        if(avgloss>best):
            best=avgloss
            print("Network Improvement")
            new_saver.save(session, "Weights/Best/model_best")
        new_saver.save(session, "Weights/model_last")
        timetaken=endtime-starttime
        msg="Epoch "+str(e)+"("+str(timetaken)+ " sec) Training: Cost is "+str(avgloss)+" Accuracy "+str(avgacc)+" Testing: Loss "+str(testloss)+" Accuracy "+str(testacc)+" Best "+str(besttestacc)+"\n"
        print(msg)
        f.write(msg)
        f.close()
