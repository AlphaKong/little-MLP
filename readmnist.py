# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle

'''
读取原始数据集
'''

def read_data():
      data_dir='data/mnist'#数据目录
      fd=open(os.path.join(data_dir,'train-images.idx3-ubyte'))
      #转化成numpy数组
      loaded=np.fromfile(file=fd,dtype=np.uint8)
      #根据mnist官网描述的数据格式，图像像素从16字节开始
      
      trX=loaded[16:].reshape((60000,784)).astype(np.float)
      #print(trX.shape)
      
      #训练label
      fd=open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
      loaded=np.fromfile(file=fd,dtype=np.uint8)
      trY=loaded[8:].reshape((60000)).astype(np.float)
      
      # 测试数据
      fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
      loaded = np.fromfile(file=fd,dtype=np.uint8)
      teX = loaded[16:].reshape((10000,784)).astype(np.float)
      #print(teX.shape)
      
      # 测试 label
      fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
      loaded = np.fromfile(file=fd,dtype=np.uint8)
      teY = loaded[8:].reshape((10000)).astype(np.float)
      
      #热编码
      trYL = []
      for i in trY:
            hot=np.zeros(10)
            hot[int(i)]=1
            trYL.append(hot)
      trY=np.array(trYL)
      teYL = []
      for i in teY:
            hot=np.zeros(10)
            hot[int(i)]=1
            teYL.append(hot)
      teY=np.array(teYL)
      return trX,trY,teX,teY

#trX,trY,teX,teY=read_data()
#print(np.shape(trX))
#print(np.shape(trY))
#print(np.shape(teX))
#print(np.shape(teY))
#归一化
#trX=(trX-128)/128
#teX=(teX-128)/128

'''
用numpy保存和读取数据
'''
def np_dataset():
      trX,trY,teX,teY=read_data()
      np.save('data/trX',trX)
      np.save('data/trY',trY)
      np.save('data/teX',teX)
      np.save('data/teY',teY)
      

def read_np_dataset():
      trX=np.load('data/trX.npy')
      trY=np.load('data/trY.npy')
      teX=np.load('data/teX.npy')
      teY=np.load('data/teY.npy')
      return trX,trY,teX,teY

#np_dataset()
#trX,trY,teX,teY=read_np_dataset()
#print(np.shape(trX))
#print(np.shape(trY))
#print(np.shape(teX))
#print(np.shape(teY))


'''
用pickle保存和读取数据
可以用于任何python对象
'''
def pickle_dataset():
      trX,trY,teX,teY=read_data()
      with open('data/trX.pkl','wb') as f:
            pickle.dump(trX,f)
      with open('data/trY.pkl','wb') as f:
            pickle.dump(trY,f)
      with open('data/teX.pkl','wb') as f:
            pickle.dump(teX,f)
      with open('data/teY.pkl','wb') as f:
            pickle.dump(teY,f)

def read_pickle_dataset():
      with open('data/trX.pkl','rb') as f:
            trX=pickle.load(f)
      with open('data/trY.pkl','rb') as f:
            trY=pickle.load(f)
      with open('data/teX.pkl','rb') as f:
            teX=pickle.load(f)
      with open('data/teY.pkl','rb') as f:
            teY=pickle.load(f)
      return trX,trY,teX,teY


#pickle_dataset()


#x,y,tx,ty=read_pickle_dataset()
#print(np.shape(x))
#print(np.shape(y))
#print(np.shape(tx))
#print(np.shape(ty))

