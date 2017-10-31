from __future__ import print_function
import h5py,math,sys
import numpy as np

#First
def readMainH5(h5file):
    #Reads a standard format h5 file and collects features, targets and sequence_lengths
    f=h5py.File(h5file)
    samplenames=list(f.keys())
    total=len(samplenames)
    all_x=[]
    all_y=[]
    all_seq_len=[]
    c=1.0
    for t in range(total):
        try:
            sample=f.get(samplenames[t])
            print("Reading ",sample.attrs["SampleID"])
            features=np.asarray(sample.get("Features"))
            all_x.append(features)
            steps=len(features)
            all_seq_len.append(steps)
            target=sample.attrs["Reorder_Target"]
            all_y.append(target)
            completed = (c / total) * 100
            #sys.stdout.write("Reading " + sample.attrs["SampleID"]+"--Complete %f\r",completed)
            #sys.stdout.flush()
        except:
            pass
    print("\nReading ",h5file," complete")
    return all_x,all_y,all_seq_len

#Second
def findDistinctCharacters(targets):
    '''
    Reads all targets (targets) and splits them to extract individual characters
    Creates an array of character-integer map (char_int)
    Finds the maximum target length 
    Finds number of distinct characters (nbclasses)
    :param targets:
    :return char_int,max_target_length,nbclasses:
    '''
    total=len(targets)
    max_target_length=0
    char_int=[]
    all_chars=[]
    total_transcription_length=0 #Total number of characters
    for t in range(total):
        this_target=targets[t]
        chars=this_target.split()
        target_length=len(chars)
        total_transcription_length=total_transcription_length+target_length
        if(target_length>max_target_length):
            max_target_length=target_length
        for ch in chars:
            all_chars.append(ch)


    charset=list(set(all_chars))
    '''
    char_int.append("PD") #A special character representing padded value
    for ch in charset:
        char_int.append(ch)
    '''
    nbclasses=len(charset)
    print("Character Set processed for ",total," data")
    print(charset)
    '''
    f=open("Character_Integer","w")
    for c in char_int:
        f.write(c+"\n")
    f.close()
    '''
    return charset,max_target_length,nbclasses,total_transcription_length

#Third
def pad_x(x,maxlen,nbfeatures):
    total=len(x)
    #print("Padding X Shape=",len(x),len(x[0]),len(x[0][0]))
    padded_x=np.zeros([total,1,maxlen,nbfeatures])
    for t in range(total):
        for step in range(len(x[t])):
            padded_x[t][0][step]=x[t][step]
    #print("\tPadding complete for ",total," data")
    return padded_x

#Call inside Training Module
def make_sparse_y(targets,char_int,max_target_length):
    total = len(targets)
    indices=[]
    values=[]
    shape=[total,max_target_length]
    for t in range(total):
        chars=targets[t].split()
        for c_pos in range(len(chars)):
            sparse_pos=[t,c_pos]
            sparse_val=char_int.index(chars[c_pos])
            indices.append(sparse_pos)
            values.append(sparse_val)
    return [indices,values,shape]


#Adjust Sequence lengths after CNN and Pooling
def adjustSequencelengths(seqlen,convstride,poolstride,maxtargetlength):
    total=len(seqlen)
    layers=len(convstride)
    for l in range(layers):
        for s in range(total):
            seqlen[s]=max(maxtargetlength,math.ceil(seqlen[s]/(convstride[l]*poolstride[l])))
    return seqlen

#Main
def load_data(trainh5,testh5,batchsize,mode,generate_char_table):
    train_x, train_y, train_seq_len=readMainH5(trainh5)
    test_x,test_y,test_seq_len=readMainH5(testh5)

    train_charset, train_max_target_length, train_nbclasses,train_transcription_length=findDistinctCharacters(train_y)
    test_charset, test_max_target_length, test_nbclasses, test_trainscription_length = findDistinctCharacters(test_y)
    print("Train Char Set ",train_nbclasses," Test Character set ",test_nbclasses)

    if(train_nbclasses<test_nbclasses):
        print("Warning ! Test set have more characters")

    train_charset.extend(test_charset)

    char_int = []
    if(generate_char_table):
        charset=list(set(train_charset)) # A combined Character set is created from Train and test Character set
        charset.sort()
        charset.insert(0,"PD")
        charset.append("BLANK")
        nb_classes=len(charset) #For Blank

        for ch in charset:
           char_int.append(ch)

        ci = open("Character_Integer","w")
        for ch in char_int:
            ci.write(ch+"\n")
        ci.close()
        print("Character Table Generated and Written")
    else:
        ci=open("Character_Integer")
        line=ci.readline()
        while line:
            char=line.strip("\n")
            char_int.append(char)
            line=ci.readline()
        nb_classes=len(char_int)
        print("Character Table Loaded from Generated File")
    print(char_int)

    max_target_length=max(train_max_target_length,test_max_target_length)

    max_train_seq_len=max(train_seq_len)
    max_test_seq_len=max(test_seq_len)

    max_seq_len=max(max_train_seq_len,max_test_seq_len)

    if(mode=="pad"):
        x_train=pad_x(train_x,max_seq_len)
        x_test=pad_x(test_x,max_seq_len)
        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
    else:
        x_train=train_x
        x_test=test_x


    nbtrain=len(train_y)
    nbtest=len(train_y)

    y_train=[]
    y_test=[]

    batches=int(np.ceil(nbtrain/float(batchsize)))
    start=0
    for b in range(batches):
        end=min(nbtrain,start+batchsize)
        sparse_target=make_sparse_y(train_y[start:end],char_int,max_target_length)
        y_train.append(sparse_target)
        start=end

    batches = int(np.ceil(nbtest / float(batchsize)))
    start = 0
    for b in range(batches):
        end = min(nbtest, start + batchsize)
        sparse_target = make_sparse_y(test_y[start:end], char_int, max_target_length)
        y_test.append(sparse_target)
        start = end
    transcription_length=[train_transcription_length,test_trainscription_length]
    return [x_train,x_test],nb_classes,[train_seq_len,test_seq_len],[y_train,y_test],max_target_length,max_seq_len,char_int,transcription_length

#Convert integer representation of string to unicode representation
def int_to_unicode(intarray,char_int_file,dbfile):
    '''
    Takes an array of integers (each representing a character as given in char_int_file
    dbfile contains global mapping
    :param intarray:
    :param char_int_file:
    :param dbfile:
    :return:unicode string,mapped character string
    '''
    char_int=[]
    f=open(char_int_file)
    line=f.readline()
    while line:
        info=line.strip("\n")
        char_int.append(info)
        line=f.readline()
    f.close()

    chars=[]
    for i in intarray:
        chars.append(char_int[i])

    unicodestring=""
    for ch in chars:
        f=open(dbfile)
        line=f.readline()
        while line:
            info=line.strip("\n").split(",")
            if(len(info)<=5):
                if(info[3]==ch):
                    unicodestring=unicodestring+" "+info[2]
            line=f.readline()
    return unicodestring,chars

def find_unicode_info(char,dbfile):
    #returns type and actual unicode position of a character
    f=open(dbfile)
    line=f.readline()
    type="v"
    pos="#"
    while line:
        info=line.strip("\n").split(",")
        if(len(info)>5):
            #skip line
            line=f.readline()
        else:
            if(char==info[2]):#Found it in DB
                type=info[0]
                if(type=="m"):#its a modifier
                    pos=info[-1]
                break
            line=f.readline()
    f.close()
    return [type,pos]


def reset_unicode_order(unicodestring,dbfile):
    #Takes unicodestring seperated by space
    #returns properly ordered unicodestring
    unicodearray=unicodestring.split()
    nbchars=len(unicodearray)
    i=0
    while (i<nbchars-2):
        [type, pos]=find_unicode_info(unicodearray[i],dbfile)
        if(type=="m"):# May need swap
            if(pos=="p"):#swap
                temp=unicodearray[i]
                unicodearray[i]=unicodearray[i+1]
                unicodearray[i+1]=temp
                i=i+1
        i=i+1
    reorder_string=""
    for u in unicodearray:
        reorder_string=reorder_string+u
    return reorder_string
