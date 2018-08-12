
# coding: utf-8

# In[ ]:


import re
import matplotlib.pyplot as plt
t_loss = []
t_index=[]
t_count=0
file1=open('op_train_loss11.txt','r')
for  line in file1:
    t_count=t_count+1
    t_index.append(str(t_count))
    #t_loss.insert(t_count,line)
    line =re.sub(r"\n","",line)
    #print(int(line))
    t_loss.append(line)
print(t_loss)
t_loss = [float(i) for i in t_loss]
#list(map(int, t_loss[0]))
v_loss = []
v_index=[]
v_count=0
file2=open('op_val_loss11.txt','r')
for  line in file2:
    #print(line)
    v_count=v_count+1
    v_index.append(str(v_count))
    #t_loss.insert(t_count,line)
    line =re.sub(r"\n","",line)
    v_loss.append(line)
print(v_loss)
v_loss = [float(i) for i in v_loss]
file1.close()
file2.close()
plt.plot(t_index,t_loss,'b',label="train_loss")
plt.plot(v_index,v_loss,label="val_loss")
plt.ylabel('loss')
plt.xlabel('Number of epochs')
plt.legend()
plt.savefig('op_train_val_loss_bidirectional_with attention.png')
plt.show()


