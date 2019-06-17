
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *
from fastai import *
from fastai.metrics import error_rate
import pandas as pd
import torch


# In[3]:


path ="../input/"
train_df=pd.read_csv(path+"train.csv")
test_df=pd.read_csv(path+"sample_submission.csv")


# In[4]:


bs = 128
data = ImageDataBunch.from_csv(path=path, folder='train/train', csv_labels='train.csv', ds_tfms=get_transforms(), size=32, bs=bs).normalize(imagenet_stats)


# In[5]:


data.show_batch(rows=3, figsize=(7,6))


# In[6]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[7]:


learn = cnn_learner(data, models.densenet161, metrics=error_rate, model_dir="/tmp/model/")


# In[8]:


learn.fit_one_cycle(6)


# In[9]:


learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[10]:


learn.fit_one_cycle(5, max_lr=slice(3e-5,3e-4))


# In[11]:


interp = ClassificationInterpretation.from_learner(learn)


# In[12]:


interp.plot_confusion_matrix()


# In[13]:


a,b,c=learn.predict(open_image("../input/test/test/"+"000940378805c44108d287872b2f04ce.jpg"))
print(c)
print(c[1].numpy())


# In[14]:


test_df.head()


# In[15]:


def pred(name):
    a,b,c=learn.predict(open_image("../input/test/test/"+name))
    return c[1].numpy()


# In[16]:


test_df["has_cactus"]=test_df["id"].apply(lambda x:pred(x))


# In[17]:


test_df.to_csv('submission.csv',index=False)

