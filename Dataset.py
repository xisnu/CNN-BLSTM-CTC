# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:52:49 2016

@author: parthosarothi
"""
import csv
import numpy as np

import h5py
global characters


#Need
def loaddata(filenames,batchsize,mode,cpath):
    """
    Inputs 2 filenames one for training and one for testing
    returns 3D X, sparse Y for train and test and set of distinct chars
    """
    inpx,inpy,ms,nf=readdata_RNN(filenames[0])
    nbtrain=len(inpx)
    test_inpx,test_inpy,testms,nf=readdata_RNN(filenames[1])
    nbtest=len(test_inpx)
    if(testms>ms):
        ms=testms
    inpx,inpseqlen,trainbatch=splitintobatch_X(batchsize,inpx,inpy,ms,mode)
    
    chars,nb_chars_train=getallcharacters(inpy,cpath)
    test_chars,nb_chars_test=getallcharacters(test_inpy,cpath)
    inp_sparse_y=splitintobatch_Y(batchsize,inpy,chars)
    print("Data is ready for Training in ",trainbatch," batches")
    test_inpx,test_inpseqlen,testbatch=splitintobatch_X(batchsize,test_inpx,test_inpy,ms,mode)
    test_inp_sparse_y=splitintobatch_Y(batchsize,test_inpy,chars)
    print("Data is ready for Testing in ",testbatch," batches")
    return [inpx,inp_sparse_y,inpseqlen,trainbatch,nbtrain],[test_inpx,test_inp_sparse_y,test_inpseqlen,testbatch,nbtest],chars,nb_chars_train,nb_chars_test


#Need
def splitintobatch_X(batchsize,x,y,ms,mode):
    """
    inputs batchsize (int), x (3D Input),targets, maximum steps
    """
    total=len(x)
    print("Splitting Data (X,Y) into batches from ",total," samples")
    start=0
    all_x=[]
    all_seqlen=[]
    batchcount=0
    while(start<=total):
        xb=[]
        yb=[]
        seqlen=[]
        end=start+batchsize
        if(end>total):
            end=total
            print("Batch Split from ",start," to ",end)
            for i in range(start,end):
                xb.append(x[i])
                yb.append(y[i])
            
            xb=padwithzeros(xb,ms)
            seqlen=findsequencelength(xb)    
            if(mode=="CNN"):
                xb=np.expand_dims(xb,axis=1)
            all_x.append(xb)
            all_seqlen.append(seqlen)
            print("\n")
            batchcount=batchcount+1
            break
        else:
            print("Batch Split from ",start," to ",end)
            for i in range(start,end):
                xb.append(x[i])
                yb.append(y[i])
            
            xb=padwithzeros(xb,ms)
            seqlen=findsequencelength(xb)
            if(mode=="CNN"):
                xb=np.expand_dims(xb,axis=1)
            all_x.append(xb)
            all_seqlen.append(seqlen)
            print("\n")
            batchcount=batchcount+1
        start=end        
    return np.asarray(all_x),np.asarray(all_seqlen),batchcount

#Need
def getallcharacters(targets,cpath):
    """
    inputs all the characters strings and returns a set of distinct charcters followed by "undecided"
    """
    #print(targets)    
    characters=[]
    for b in range(len(targets)):
        thistarget=targets[b].decode("utf-8").split(" ")
        for e in range(len(thistarget)):
                characters.append(thistarget[e])
    allchars=len(characters)
    characters=list(set(characters))
    characters.append("ud")
    charfile=open(cpath+"/class-int.txt","w")
    for l in range(len(characters)):
        try:
            charfile.write(str(characters[l])+","+str(l)+"\n")
        except:
            pass
    print("Distinct characters ",characters)
    charfile.close()
    return characters,allchars
#need
def splitintobatch_Y(batchsize,y,characters):
    """
    inputs batchsize (int), targets, a set of distinct characters
    returns sparse represenation of targets splitted in batches
    """
    total=len(y)
    print("Splitting Targets into batches from ",total," samples")
    start=0
    all_y=[]
    batchcount=0
    while(start<=total):
        yb=[]
        end=start+batchsize
        if(end>total):
            end=total
            print("Batch Split from ",start," to ",end)
            for i in range(start,end):
                yb.append(y[i])            
            ixs,vals,shp=encodectclabel(yb,characters)
            all_y.append([ixs,vals,shp])
            print("\n")
            batchcount=batchcount+1
            break
        else:
            print("Batch Split from ",start," to ",end)
            for i in range(start,end):
                yb.append(y[i])            
            ixs,vals,shp=encodectclabel(yb,characters)
            all_y.append([ixs,vals,shp])
            print("\n")
            batchcount=batchcount+1
        start=end        
    return np.asarray(all_y)

#Need
def readdata_RNN(h5file):
    print("Read all samples from H5 and make Array of shape [Samples, Steps, nbfeatures]")
    f = h5py.File(h5file,"r")
    root=list(f.keys())
    total = len(root)
    allsamples = []
    alltargets = []
    k = 0
    maxsteps = 0
    for d in range(total):#For each samples in H5
        sample=f.get(root[d])
        sampleid = sample.attrs["SampleID"]
        target = sample.attrs["Custom_Target"]
        steps = []
        allfeatures = np.array(sample.get("Features"))
        try:
            nbsteps = len(allfeatures)
            for es in range(nbsteps):#For each Feature Vector
                steps.append(allfeatures[es])
        except:
            print("Exception for ",sampleid)


        nof=len(steps[-1])
        allsamples.append(np.array(steps))
        alltargets.append(target)
        totalsteps = len(steps)

        if(totalsteps>maxsteps):
            maxsteps = totalsteps
        print ("Sample ",k," ID ",sampleid," steps ",totalsteps," Each step has ",nof," elements. target ",target)

    print("Total ",k," samples processed, Maximum steps ",maxsteps)
    return np.asarray(allsamples),alltargets,maxsteps,nof
#Need
def padwithzeros(allsamples,maxsteps):
    total=len(allsamples)
    try:
        dim=len(allsamples[0][0])
    except:
        dim=1
    steps=maxsteps
    newsamples=np.zeros([total,steps,dim])
    mask=np.zeros([total,steps,dim])
    print("Zero padding to array ",total,",",steps,",",dim)
    if(dim==1):
         for t in range(total):
             for s in range(len(allsamples[t])):
                 newsamples[t][s]=allsamples[t][s]
    else:
        for t in range(total):
            for s in range(len(allsamples[t])):
                for d in range(dim):
                    newsamples[t][s][d]=allsamples[t][s][d]
                    mask[t][s][d]=1
    print("All samples padded with zeros")
    return newsamples


#Need
def encodectclabel(targets,characters):
    all_targets=[]
    seqlen=[]
    total=len(targets)
    maxlen=0
    print("Distinct characters --",len(characters)," ",characters)

    for t in range(total):
        newtarget=targets[t].decode("utf-8")
        newtarget=newtarget.split(" ")
        ctclabel=[]
        for c in range(len(newtarget)):
                ctclabel.append(characters.index(newtarget[c]))
        #ctclabel.append(cha.index("ud"))
            #if(c<len(newtarget)-1):
                #ctclabel.append(distclasses.index("ud"))

        thislen=len(ctclabel)
        if(thislen>maxlen):
            maxlen=thislen
        all_targets.append(ctclabel)
        seqlen.append(thislen)
    all_targets=np.array(all_targets)
    print("All Lables Shape ",all_targets.shape," First label Character",all_targets[0])

    sparse_indices=[]
    sparse_values=[]
    for t in range(total):#for every target
        #print("Reading target ",all_targets[t])
        for e in range(len(all_targets[t])):#for each element in target[t]
            sparse_indices.append([t,e])
            sparse_values.append(all_targets[t][e])
        #print("True Label ",newtarget,"CTC Label ",ctclabel)
    sparse_indices=np.array(sparse_indices)
    sparse_values=np.array(sparse_values)
    sparse_shape=[total,maxlen]
    return sparse_indices,sparse_values,sparse_shape

#need
def findsequencelength(inputs):
    total=len(inputs)
    seqlen=[]
    for t in range(total):
        seqlen.append(len(inputs[t]))
    return seqlen

def label_from_sparse(sparse,index):
    #sparse=sparse[0]
    #print("Length of sparse ",len(sparse[1]))
    i=sparse[0]
    v=sparse[1]
    label=[]
    t=0
    while(t<len(i)):
        if(i[t][0]==index):
            while(i[t][0]==index):
                label.append(v[t])
                t=t+1
            break
        else:
            t=t+1
    return label

