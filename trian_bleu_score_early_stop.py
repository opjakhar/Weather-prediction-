
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import tensorflow.contrib.rnn 
import pickle
import findbleu as bl
#------------------------------parameter setting ----------------------
epochs =1
batch_size = 100
hidden_size = 256    
num_layers = 1
inembsize=1024
decmbsize=512
learning_rate = 0.0075
model_name="model7"
flag_1 = True      # for   bidirectional LSTM 
flag_2 =False        # for unidirectional LSTM 
flag_3 = True       # for attention 
flag_4 = False       # for without attention
flag_5 = True       # for early stop 

#-----------------------Data loading  and vocab generating -----------------------
import re
train_data=""

train_count1=0
train_count2=0
train_count3=0
train_count4=0
train_count5=0
train_data_dummy=[]
file = open('./WeatherGov/train/train.combined', 'r') 
for line in file: 
    #print(line)
    train_count1=train_count1+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    train_data_dummy.extend(split_data)
file = open('./WeatherGov/dev/dev.combined', 'r') 
for line in file: 
    #print(line)
    train_count3=train_count3+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    train_data_dummy.extend(split_data)

file = open('./WeatherGov/test/test.combined', 'r') 
for line in file: 
    #print(line)
    train_count5=train_count5+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    train_data_dummy.extend(split_data)

combined_vocab=set(train_data_dummy)
#print(len(combined_vocab))
#print(combined_vocab)

train_data_dummy=[]
train_data=""
#----------------------------------------------------

file = open('./WeatherGov/train/summaries.txt', 'r') 
for line in file: 
    #print(line)
    train_count2=train_count2+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    train_data_dummy.extend(split_data)

file = open('./WeatherGov/dev/summaries.txt', 'r') 
for line in file: 
    #print(line)
    train_count4=train_count4+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    train_data_dummy.extend(split_data)   

summary_vocab=set(train_data_dummy)

#-------------------------------------------------------------------------
combined_vocab   
summary_vocab
#-------------------------------------------------------------------------


#----specail words-------------------------------------------------------
special_words = ['<PAD>', '<GO>',  '<EOS>', '<UNK>']

#-----vocab for combined-------------------------------------------------


full_combined_vocab=special_words + list(combined_vocab)
vcb_count=0
combined_int_to_vocab=dict()
for wrd in full_combined_vocab:
    combined_int_to_vocab[vcb_count]=wrd
    vcb_count=vcb_count+1

    
    
vcb_count=0
combined_vocab_to_int=dict()
for wrd in full_combined_vocab:
    combined_vocab_to_int[wrd]=vcb_count
    vcb_count=vcb_count+1

#combined_vocab_to_int    
#combined_int_to_vocab    


#-----vocab for summary---------------------------------------

full_summary_vocab=special_words + list(summary_vocab)

vcb_count=0
summary_int_to_vocab=dict()
for wrd in full_summary_vocab:
    summary_int_to_vocab[vcb_count]=wrd
    vcb_count=vcb_count+1

vcb_count=0
summary_vocab_to_int=dict()
for wrd in full_summary_vocab:
    summary_vocab_to_int[wrd]=vcb_count
    vcb_count=vcb_count+1
    
    

# summary_vocab_to_int
# summary_int_to_vocab
    


#----------------------------------dividing data into testing and validation --------------------

#----------------data for training ------------------
file = open('./WeatherGov/train/train.combined', 'r') 
text_data_for_training=[]
train_count1=0
for line in file: 
    train_count1=train_count1+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    list_ele=[]
    dummy_count=0
    for ele in split_data:
            mod_ele=combined_vocab_to_int.get(ele, combined_vocab_to_int['<UNK>'])
            #print(mod_ele)
            list_ele.insert(dummy_count,mod_ele)
            dummy_count=dummy_count+1
    text_data_for_training.append(list_ele)

#combined_word_id = text_data_for_training
training_combined_word_id=text_data_for_training
 
file = open('./WeatherGov/train/summaries.txt', 'r') 
label_data_for_training=[]
train_count2=0
for line in file: 
    train_count2=train_count2+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    list_ele=[]
    dummy_count=0
    for ele in split_data:
            mod_ele=summary_vocab_to_int.get(ele, summary_vocab_to_int['<UNK>'])
            #print(mod_ele)
            list_ele.insert(dummy_count,mod_ele)
            dummy_count=dummy_count+1
    label_data_for_training.append(list_ele)

#summary_word_id = label_data_for_training
training_summary_word_id = label_data_for_training


#----------------data for validation------------------

file = open('./WeatherGov/dev/dev.combined', 'r') 
text_data_for_training=[]
train_count1=10
for line in file: 
    train_count1=train_count1+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    list_ele=[]
    dummy_count=0
    for ele in split_data:
            mod_ele=combined_vocab_to_int.get(ele, combined_vocab_to_int['<UNK>'])
            #print(mod_ele)
            list_ele.insert(dummy_count,mod_ele)
            dummy_count=dummy_count+1
    text_data_for_training.append(list_ele)

valid_combined_word_id = text_data_for_training
 
file = open('./WeatherGov/dev/summaries.txt', 'r') 
label_data_for_training=[]
train_count2=0
for line in file: 
    train_count2=train_count2+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    list_ele=[]
    dummy_count=0
    for ele in split_data:
            mod_ele=summary_vocab_to_int.get(ele, summary_vocab_to_int['<UNK>'])
            #print(mod_ele)
            list_ele.insert(dummy_count,mod_ele)
            dummy_count=dummy_count+1
    label_data_for_training.append(list_ele)

valid_summary_word_id = label_data_for_training
#----------------------------------------------------------

print(train_count1)
print(train_count2)


training_combined_word_id
training_summary_word_id

valid_combined_word_id
valid_summary_word_id

# test_combined_word_id

print(train_count1)
print(train_count2)
print("complete vocab  operation")
#----------------------------------------------------------
valid_combined = valid_combined_word_id
valid_summary = valid_summary_word_id
training_combined = training_combined_word_id
training_summary = training_summary_word_id
#--------------------------------------------------------

#-----------------------------function Definitions-----------------------------------

def model_inputs():
    lr = tf.placeholder(tf.float32, name='learning_rate')
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    combined_sequence_length = tf.placeholder(tf.int32, (None,), name='combined_sequence_length')
    summary_data = tf.placeholder(tf.int32, [None, None], name='summary_data')
    summary_sequence_length = tf.placeholder(tf.int32, (None,), name='summary_sequence_length')
    max_summary_sequence_length = tf.reduce_max(summary_sequence_length, name='max_summary_len')
    
    
    return input_data, summary_data, lr, summary_sequence_length, max_summary_sequence_length, combined_sequence_length


def combined_data_encoding(input_data, hidden_size, num_layers,
                   combined_sequence_length, combined_vocab_size, 
                   inembsize):

   
    encoder_input = tf.contrib.layers.embed_sequence(input_data, combined_vocab_size, inembsize)
    
    def create_state_cell(hidden_size):
        encoder_cell = tf.contrib.rnn.LSTMCell(hidden_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return encoder_cell
    
  


    if flag_1 == True :
        print("using Bidirectional LSTM")
        encoder_state_cell=create_state_cell(hidden_size/2)

        ((encoder_forward_outputs,
          encoder_backward_outputs),
         (encoder_forward_final_state,
          encoder_backward_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_state_cell,
                                            cell_bw=encoder_state_cell,
                                            inputs=encoder_input,
                                            sequence_length=combined_sequence_length,
                                            dtype=tf.float32, time_major=False)
             )


        encoder_output = tf.concat((encoder_forward_outputs, encoder_backward_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_forward_final_state.c, encoder_backward_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_forward_final_state.h, encoder_backward_final_state.h), 1)

        encoder_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )
    
    if flag_2 == True :
        print("using unidirectional LSTM")
        encoder_state_cell =  create_state_cell(hidden_size)
        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_state_cell, encoder_input, sequence_length=combined_sequence_length, dtype=tf.float32)

    print(tf.shape(encoder_output))
    print(tf.shape(encoder_state))
    
    
    return encoder_output, encoder_state

def process_decoder_input(summary_data, vocab_to_int, batch_size):
    ending = tf.strided_slice(summary_data, [0, 0], [batch_size, -1], [1, 1])
    decode_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decode_input
#--------------------------------------

def decoding_summary_layer(summary_vocab_to_int, decmbsize, num_layers, hidden_size,
                   summary_sequence_length,  max_summary_sequence_length, encoder_state, decode_input,encoder_output,combined_sequence_length):
    
    
    summary_vocab_size = len(summary_vocab_to_int)
    
    decoder_input = tf.Variable(tf.random_uniform([summary_vocab_size, decmbsize]))
    print("check point one3")
    decoder_input_embed = tf.nn.embedding_lookup(decoder_input, decode_input)
    print("check point one4")
    
    def create_cell(hidden_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(hidden_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell
    print("check point one5")
    if flag_3 == True :
        print("using attention ")
        attention_states=encoder_output

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        hidden_size, attention_states,
        memory_sequence_length=combined_sequence_length,normalize=True)


        decoder_cell=create_cell(hidden_size)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=hidden_size)
    if flag_4 == True :
        print("without attention")
        decoder_cell = create_cell(hidden_size)
    print("check point one6")

    decoder_output_layer = Dense(summary_vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    print("check point one7")
    with tf.variable_scope("decode"):
        print("check point one8")
       
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input_embed,
                                                            sequence_length=summary_sequence_length,
                                                            time_major=False)
        print("check point one9")     
        
       
        if flag_3 == True:
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                           training_helper,
                                                           decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state),
                                                           decoder_output_layer) 
        if flag_4 == True :
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                           training_helper,
                                                           encoder_state,
                                                           decoder_output_layer) 

        print(training_decoder)
        print("check point one10")
        
       
        
        training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations= max_summary_sequence_length)[0]
       
        print("check point one11")
    with tf.variable_scope("decode", reuse=True):
        start_symboll = tf.tile(tf.constant([summary_vocab_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_symboll')


        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_input,
                                                                start_symboll,
                                                                summary_vocab_to_int['<EOS>'])


        if flag_3 == True :
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                            inference_helper,
                                                            decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state),
                                                            decoder_output_layer)
        if flag_4 == True: 
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                        inference_helper,
                                                        encoder_state,
                                                        decoder_output_layer)
            
        

        inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            impute_finished=True,
                                                            maximum_iterations= max_summary_sequence_length)[0]
         

    
    return training_decoder_output, inference_decoder_output
#-------------------------------------------------
def seq2seq_model(input_data, summaries, lr, summary_sequence_length, 
                  max_summary_sequence_length, combined_sequence_length,
                  combined_vocab_size, summary_vocab_size,
                  enc_embedding_size, dec_embedding_size, 
                  hidden_size, num_layers):
    
    dec_input = process_decoder_input(summaries, summary_vocab_to_int, batch_size)
    encoder_output,encoder_state = combined_data_encoding(input_data,hidden_size,num_layers,combined_sequence_length,combined_vocab_size,inembsize)
   

    train_decoder_output, infrnc_decoder_output = decoding_summary_layer(summary_vocab_to_int,decmbsize,num_layers,hidden_size,summary_sequence_length,max_summary_sequence_length,encoder_state,dec_input,encoder_output,combined_sequence_length) 
    print("check point one2")
    return train_decoder_output, infrnc_decoder_output
#-----------------------------------------------------------
def add_pad(Batch_data):
    max_batch_entry_size = max([len(batch_entry) for batch_entry in Batch_data])
    batch_entry_count=0
    final_padded_entry=[]
    for batch_entry in Batch_data:
        btach_entry_length=len(batch_entry)
        pad_length=max_batch_entry_size - btach_entry_length
        for pad_len in range(pad_length):
            batch_entry.append(0)
        final_padded_entry.insert(batch_entry_count,batch_entry)
        batch_entry_count=batch_entry_count+1
        
    return final_padded_entry
#-------------------------------------------------------------
def batches_data(summary_batch_data, combined_batch_data, batch_size, combined_pad_int, summary_pad_int):
    number_of_batches=len(combined_batch_data)//batch_size
    for batch_number in range(0, number_of_batches):
        index = batch_number * batch_size
        combined_batch = combined_batch_data[index:index + batch_size]
        summary_batch = summary_batch_data[index:index + batch_size]
        padded_combined_batch = np.array(add_pad(combined_batch))
        padded_summary_batch = np.array(add_pad(summary_batch))
        
        padded_summary_lengths = []
        for summary_list_data in padded_summary_batch:
            padded_summary_lengths.append(len(summary_list_data)) 
        
        padded_combined_lengths = []
        for combined_list_data in padded_combined_batch:
            padded_combined_lengths.append(len(combined_list_data))
        
        yield padded_summary_batch, padded_combined_batch, padded_summary_lengths, padded_combined_lengths


#-------------------creating graph----------------------------------------
training_graph = tf.Graph()
with training_graph.as_default():
        
    input_data, summary_data, lr, summary_sequence_length, max_summary_sequence_length, combined_sequence_length = model_inputs()
   
    train_decoder_output, infrnc_decoder_output = seq2seq_model(input_data, 
                                                                      summary_data, 
                                                                      lr, 
                                                                      summary_sequence_length, 
                                                                      max_summary_sequence_length, 
                                                                      combined_sequence_length,
                                                                      len(combined_vocab_to_int),
                                                                      len(summary_vocab_to_int),
                                                                      inembsize, 
                                                                      decmbsize, 
                                                                      hidden_size, 
                                                                      num_layers)    
    
   
    print("check point one1")
    train_logits = tf.identity(train_decoder_output.rnn_output, 'logits')
    infrnc_logits = tf.identity(infrnc_decoder_output.sample_id, name='predictions')
    
   
    sequence_masks = tf.sequence_mask(summary_sequence_length, max_summary_sequence_length, dtype=tf.float32, name='sequence_masks')

    with tf.name_scope("optimization"):
        
        
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            summary_data,
            sequence_masks)

        
        adam_optimizer = tf.train.AdamOptimizer(lr)

        
        gradients = adam_optimizer.compute_gradients(cost)
        clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = adam_optimizer.apply_gradients(clipped_gradients)





early_stop=6
train_loss_file = open("./"+model_name+"/train_loss.txt", "w")
val_loss_file = open("./"+model_name+"/val_loss.txt", "w")
with tf.Session(graph=training_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    prev_val_bleu=0    
    val_loss_dict=dict()
    train_loss_dict=dict()
    for epoch_number in range(0, epochs):
        val_loss_batch=0
        train_loss_batch=0
        if early_stop==0 and flag_5 == True:
            print("using early stop")
            break
        tbatch_count=0
        vbatch_count=0
        for batch_number, (padded_summary_batch, padded_combined_batch, padded_summary_lengths, padded_combined_lengths) in enumerate(batches_data(training_summary, training_combined, batch_size,
                           combined_vocab_to_int['<PAD>'],
                           summary_vocab_to_int['<PAD>'])):
            
            # Training part-----------------------
            _, train_error = sess.run(
                [train_op, cost],
                {input_data: padded_combined_batch,
                 summary_data: padded_summary_batch,
                 lr: learning_rate,
                 summary_sequence_length: padded_summary_lengths,
                 combined_sequence_length: padded_combined_lengths})
            train_loss_batch=train_loss_batch+train_error
            tbatch_count=tbatch_count+1
            print("running epoch is :{} running_train_batch : {}".format(epoch_number,batch_number))
        #validation part---------------------------------
        for batch_number, (valid_summary_batch, valid_combined_batch, valid_summary_lengths, valid_combined_lengths) in enumerate(batches_data(valid_summary, valid_combined, batch_size,
                           combined_vocab_to_int['<PAD>'],
                           summary_vocab_to_int['<PAD>'])):
    
            val_error = sess.run(
                [cost],
                {input_data: valid_combined_batch,
                 summary_data: valid_summary_batch,
                 lr: learning_rate,
                 summary_sequence_length: valid_summary_lengths,
                 combined_sequence_length: valid_combined_lengths})
            
            val_loss_batch=val_loss_batch+val_error[0]
            vbatch_count=vbatch_count+1
            print("running epoch is :{} running_val_batch : {}".format(epoch_number,batch_number))

        train_loss_dict[epoch_number]=train_loss_batch/tbatch_count        
        val_loss_dict[epoch_number]=val_loss_batch/vbatch_count  
        
        total_padded_size=200
        train_count1=0
        label_data_for_training=[]
        file = open('./WeatherGov/test/test.combined', 'r') 
        for line in file: 
            train_count1=train_count1+1
            train_data=line
            train_data=re.sub(r"\n","",train_data)
            split_data=train_data.split(' ')
            dummy_count=0
            list_ele121=[]
            for ele in split_data:
                    #print(ele)
                    mod_ele=combined_vocab_to_int.get(ele, combined_vocab_to_int['<UNK>'])
                   # print(mod_ele)
                    list_ele121.insert(dummy_count,mod_ele)
                    dummy_count=dummy_count+1
            if len(list_ele121)<total_padded_size :
                pad_length = total_padded_size - len(list_ele121)
                for pad_len in range(pad_length):
                    list_ele121.append(0)
            label_data_for_training.append(list_ele121)

        file_out = open("./"+model_name+"/valid_summary_output.txt",'w')
        print(train_count1)
        for ii in range(0,train_count1):

                text=label_data_for_training[ii]
                output_summary = sess.run(logits, {input_data: [text]*batch_size, 
                                                  summary_sequence_length: [len(text)]*batch_size, 
                                                  combined_sequence_length: [len(text)]*batch_size})[0] 
                pad = combined_vocab_to_int["<PAD>"] 
                unk = combined_vocab_to_int["<UNK>"] 
                c=" ".join([summary_int_to_vocab[i] for i in output_summary if i != pad and i!= unk] )
                file_out.write(c)
                print(ii)
                file_out.write('\n')
        
        val_loss_file.close()
        val_bleu=bl.find_blue("./"+model_name+"/valid_summary_output.txt","./summaries.txt",4)
            

        if val_bleu>prev_val_bleu:
            early_stop=early_stop-1
            prev_val_bleu=val_bleu
            
            

    print("val_error {}".format(train_loss_batch/tbatch_count))
    print("train_error {}".format(val_loss_batch/vbatch_count))
    print("\n")

    for tindex,tlist in train_loss_dict.items():
        train_loss_file.write(str(tlist))
        train_loss_file.write("\n")
    for vindex,vlist in val_loss_dict.items():
        val_loss_file.write(str(vlist))
        val_loss_file.write("\n")

    
    #----------Saveing model-----------------
    saver = tf.train.Saver()
    saver.save(sess, "./"+model_name+"/model_file.ckpt")
    print('Model Trained and Saved')
#-----------testing part----------------------------------------
import re
total_padded_size=200
train_count1=0
label_data_for_training=[]
file = open('./WeatherGov/test/test.combined', 'r') 
for line in file: 
    train_count1=train_count1+1
    train_data=line
    train_data=re.sub(r"\n","",train_data)
    split_data=train_data.split(' ')
    dummy_count=0
    list_ele121=[]
    for ele in split_data:
            #print(ele)
            mod_ele=combined_vocab_to_int.get(ele, combined_vocab_to_int['<UNK>'])
           # print(mod_ele)
            list_ele121.insert(dummy_count,mod_ele)
            dummy_count=dummy_count+1
    if len(list_ele121)<total_padded_size :
        pad_length = total_padded_size - len(list_ele121)
        for pad_len in range(pad_length):
            list_ele121.append(0)
    label_data_for_training.append(list_ele121)

file_out = open("./"+model_name+"/test_summary_output.txt",'w')
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph("./"+model_name+"/model_file.ckpt" + '.meta')
    loader.restore(sess, "./"+model_name+"/model_file.ckpt")

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    combined_sequence_length = loaded_graph.get_tensor_by_name('combined_sequence_length:0')
    summary_sequence_length = loaded_graph.get_tensor_by_name('summary_sequence_length:0')    
    print(train_count1)
    for ii in range(0,train_count1):

            text=label_data_for_training[ii]
            output_summary = sess.run(logits, {input_data: [text]*batch_size, 
                                              summary_sequence_length: [len(text)]*batch_size, 
                                              combined_sequence_length: [len(text)]*batch_size})[0] 
            pad = combined_vocab_to_int["<PAD>"] 
            unk = combined_vocab_to_int["<UNK>"] 
            c=" ".join([summary_int_to_vocab[i] for i in output_summary if i != pad and i!= unk] )
            file_out.write(c)
            print(ii)
            file_out.write('\n')
file_out.close()
train_loss_file.close()
val_loss_file.close()
print("Program successfully completed")

