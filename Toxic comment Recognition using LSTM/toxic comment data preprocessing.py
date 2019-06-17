
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))


# In[ ]:


train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
test=pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)
train.head()


# In[ ]:


test.head() 


# In[ ]:


import gc
df = pd.concat([train[['id','comment_text']], test], axis=0)
df.head()
del(train, test)
gc.collect()


# In[ ]:


from gensim.models import KeyedVectors
ft_common_crawl = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
embeddings_index = KeyedVectors.load_word2vec_format(ft_common_crawl)


# In[ ]:


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


# In[ ]:


import operator
def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    return unknown_words        


# In[ ]:


df['comment_text'] = df['comment_text'].apply(lambda x: x.lower())
gc.collect()


# In[ ]:


vocab = build_vocab(df['comment_text'])
oov = check_coverage(vocab, embeddings_index)
oov[:10]


# In[ ]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 
                       "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is",
                       "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would",
                       "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have",
                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                       "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have",
                       "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 
                       "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", 
                       "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                       "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }


# In[ ]:


del(vocab,oov)
gc.collect()


# In[ ]:


def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known
print(known_contractions(embeddings_index))


# In[ ]:


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


# In[ ]:


df['comment_text'] = df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))


# In[ ]:


vocab = build_vocab(df['comment_text'])
oov = check_coverage(vocab, embeddings_index)
oov[:10]


# In[ ]:


del(vocab,oov)
gc.collect()


# In[ ]:


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


# In[ ]:


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


# In[ ]:


print(unknown_punct(embeddings_index, punct))


# In[ ]:


punct_mapping = {"_":" ", "`":" "}
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])    
    for p in punct:
        text = text.replace(p, f' {p} ')     
    return text
df['comment_text'] = df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))


# In[ ]:


vocab = build_vocab(df['comment_text'])
oov = check_coverage(vocab, embeddings_index)


# In[ ]:


oov[1:10]


# In[ ]:


del(vocab,oov)
gc.collect()


# In[ ]:


swear_words = [
    ' 4r5e ',
    ' 5h1t ',
    ' 5hit ',
    ' a55 ',
    ' anal ',
    ' anus ',
    ' ar5e ',
    ' arrse ',
    ' arse ',
    ' ass ',
    ' ass-fucker ',
    ' asses ',
    ' assfucker ',
    ' assfukka ',
    ' asshole ',
    ' assholes ',
    ' asswhole ',
    ' a_s_s ',
    ' b!tch ',
    ' b00bs ',
    ' b17ch ',
    ' b1tch ',
    ' ballbag ',
    ' balls ',
    ' ballsack ',
    ' bastard ',
    ' beastial ',
    ' beastiality ',
    ' bellend ',
    ' bestial ',
    ' bestiality ',
    ' biatch ',
    ' bitch ',
    ' bitcher ',
    ' bitchers ',
    ' bitches ',
    ' bitchin ',
    ' bitching ',
    ' bloody ',
    ' blow job ',
    ' blowjob ',
    ' blowjobs ',
    ' boiolas ',
    ' bollock ',
    ' bollok ',
    ' boner ',
    ' boob ',
    ' boobs ',
    ' booobs ',
    ' boooobs ',
    ' booooobs ',
    ' booooooobs ',
    ' breasts ',
    ' buceta ',
    ' bugger ',
    ' bum ',
    ' bunny fucker ',
    ' butt ',
    ' butthole ',
    ' buttmuch ',
    ' buttplug ',
    ' c0ck ',
    ' c0cksucker ',
    ' carpet muncher ',
    ' cawk ',
    ' chink ',
    ' cipa ',
    ' cl1t ',
    ' clit ',
    ' clitoris ',
    ' clits ',
    ' cnut ',
    ' cock ',
    ' cock-sucker ',
    ' cockface ',
    ' cockhead ',
    ' cockmunch ',
    ' cockmuncher ',
    ' cocks ',
    ' cocksuck ',
    ' cocksucked ',
    ' cocksucker ',
    ' cocksucking ',
    ' cocksucks ',
    ' cocksuka ',
    ' cocksukka ',
    ' cok ',
    ' cokmuncher ',
    ' coksucka ',
    ' coon ',
    ' cox ',
    ' crap ',
    ' cum ',
    ' cummer ',
    ' cumming ',
    ' cums ',
    ' cumshot ',
    ' cunilingus ',
    ' cunillingus ',
    ' cunnilingus ',
    ' cunt ',
    ' cuntlick ',
    ' cuntlicker ',
    ' cuntlicking ',
    ' cunts ',
    ' cyalis ',
    ' cyberfuc ',
    ' cyberfuck ',
    ' cyberfucked ',
    ' cyberfucker ',
    ' cyberfuckers ',
    ' cyberfucking ',
    ' d1ck ',
    ' damn ',
    ' dick ',
    ' dickhead ',
    ' dildo ',
    ' dildos ',
    ' dink ',
    ' dinks ',
    ' dirsa ',
    ' dlck ',
    ' dog-fucker ',
    ' doggin ',
    ' dogging ',
    ' donkeyribber ',
    ' doosh ',
    ' duche ',
    ' dyke ',
    ' ejaculate ',
    ' ejaculated ',
    ' ejaculates ',
    ' ejaculating ',
    ' ejaculatings ',
    ' ejaculation ',
    ' ejakulate ',
    ' f u c k ',
    ' f u c k e r ',
    ' f4nny ',
    ' fag ',
    ' fagging ',
    ' faggitt ',
    ' faggot ',
    ' faggs ',
    ' fagot ',
    ' fagots ',
    ' fags ',
    ' fanny ',
    ' fannyflaps ',
    ' fannyfucker ',
    ' fanyy ',
    ' fatass ',
    ' fcuk ',
    ' fcuker ',
    ' fcuking ',
    ' feck ',
    ' fecker ',
    ' felching ',
    ' fellate ',
    ' fellatio ',
    ' fingerfuck ',
    ' fingerfucked ',
    ' fingerfucker ',
    ' fingerfuckers ',
    ' fingerfucking ',
    ' fingerfucks ',
    ' fistfuck ',
    ' fistfucked ',
    ' fistfucker ',
    ' fistfuckers ',
    ' fistfucking ',
    ' fistfuckings ',
    ' fistfucks ',
    ' flange ',
    ' fook ',
    ' fooker ',
    ' fuck ',
    ' fucka ',
    ' fucked ',
    ' fucker ',
    ' fuckers ',
    ' fuckhead ',
    ' fuckheads ',
    ' fuckin ',
    ' fucking ',
    ' fuckings ',
    ' fuckingshitmotherfucker ',
    ' fuckme ',
    ' fucks ',
    ' fuckwhit ',
    ' fuckwit ',
    ' fudge packer ',
    ' fudgepacker ',
    ' fuk ',
    ' fuker ',
    ' fukker ',
    ' fukkin ',
    ' fuks ',
    ' fukwhit ',
    ' fukwit ',
    ' fux ',
    ' fux0r ',
    ' f_u_c_k ',
    ' gangbang ',
    ' gangbanged ',
    ' gangbangs ',
    ' gaylord ',
    ' gaysex ',
    ' goatse ',
    ' God ',
    ' god-dam ',
    ' god-damned ',
    ' goddamn ',
    ' goddamned ',
    ' hardcoresex ',
    ' hell ',
    ' heshe ',
    ' hoar ',
    ' hoare ',
    ' hoer ',
    ' homo ',
    ' hore ',
    ' horniest ',
    ' horny ',
    ' hotsex ',
    ' jack-off ',
    ' jackoff ',
    ' jap ',
    ' jerk-off ',
    ' jism ',
    ' jiz ',
    ' jizm ',
    ' jizz ',
    ' kawk ',
    ' knob ',
    ' knobead ',
    ' knobed ',
    ' knobend ',
    ' knobhead ',
    ' knobjocky ',
    ' knobjokey ',
    ' kock ',
    ' kondum ',
    ' kondums ',
    ' kum ',
    ' kummer ',
    ' kumming ',
    ' kums ',
    ' kunilingus ',
    ' l3itch ',
    ' labia ',
    ' lmfao ',
    ' lust ',
    ' lusting ',
    ' m0f0 ',
    ' m0fo ',
    ' m45terbate ',
    ' ma5terb8 ',
    ' ma5terbate ',
    ' masochist ',
    ' master-bate ',
    ' masterb8 ',
    ' masterbat3 ',
    ' masterbate ',
    ' masterbation ',
    ' masterbations ',
    ' masturbate ',
    ' mo-fo ',
    ' mof0 ',
    ' mofo ',
    ' mothafuck ',
    ' mothafucka ',
    ' mothafuckas ',
    ' mothafuckaz ',
    ' mothafucked ',
    ' mothafucker ',
    ' mothafuckers ',
    ' mothafuckin ',
    ' mothafucking ',
    ' mothafuckings ',
    ' mothafucks ',
    ' mother fucker ',
    ' motherfuck ',
    ' motherfucked ',
    ' motherfucker ',
    ' motherfuckers ',
    ' motherfuckin ',
    ' motherfucking ',
    ' motherfuckings ',
    ' motherfuckka ',
    ' motherfucks ',
    ' muff ',
    ' mutha ',
    ' muthafecker ',
    ' muthafuckker ',
    ' muther ',
    ' mutherfucker ',
    ' n1gga ',
    ' n1gger ',
    ' nazi ',
    ' nigg3r ',
    ' nigg4h ',
    ' nigga ',
    ' niggah ',
    ' niggas ',
    ' niggaz ',
    ' nigger ',
    ' niggers ',
    ' nob ',
    ' nob jokey ',
    ' nobhead ',
    ' nobjocky ',
    ' nobjokey ',
    ' numbnuts ',
    ' nutsack ',
    ' orgasim ',
    ' orgasims ',
    ' orgasm ',
    ' orgasms ',
    ' p0rn ',
    ' pawn ',
    ' pecker ',
    ' penis ',
    ' penisfucker ',
    ' phonesex ',
    ' phuck ',
    ' phuk ',
    ' phuked ',
    ' phuking ',
    ' phukked ',
    ' phukking ',
    ' phuks ',
    ' phuq ',
    ' pigfucker ',
    ' pimpis ',
    ' piss ',
    ' pissed ',
    ' pisser ',
    ' pissers ',
    ' pisses ',
    ' pissflaps ',
    ' pissin ',
    ' pissing ',
    ' pissoff ',
    ' poop ',
    ' porn ',
    ' porno ',
    ' pornography ',
    ' pornos ',
    ' prick ',
    ' pricks ',
    ' pron ',
    ' pube ',
    ' pusse ',
    ' pussi ',
    ' pussies ',
    ' pussy ',
    ' pussys ',
    ' rectum ',
    ' retard ',
    ' rimjaw ',
    ' rimming ',
    ' s hit ',
    ' s.o.b. ',
    ' sadist ',
    ' schlong ',
    ' screwing ',
    ' scroat ',
    ' scrote ',
    ' scrotum ',
    ' semen ',
    ' sex ',
    ' sh!t ',
    ' sh1t ',
    ' shag ',
    ' shagger ',
    ' shaggin ',
    ' shagging ',
    ' shemale ',
    ' shit ',
    ' shitdick ',
    ' shite ',
    ' shited ',
    ' shitey ',
    ' shitfuck ',
    ' shitfull ',
    ' shithead ',
    ' shiting ',
    ' shitings ',
    ' shits ',
    ' shitted ',
    ' shitter ',
    ' shitters ',
    ' shitting ',
    ' shittings ',
    ' shitty ',
    ' skank ',
    ' slut ',
    ' sluts ',
    ' smegma ',
    ' smut ',
    ' snatch ',
    ' son-of-a-bitch ',
    ' spac ',
    ' spunk ',
    ' s_h_i_t ',
    ' t1tt1e5 ',
    ' t1tties ',
    ' teets ',
    ' teez ',
    ' testical ',
    ' testicle ',
    ' tit ',
    ' titfuck ',
    ' tits ',
    ' titt ',
    ' tittie5 ',
    ' tittiefucker ',
    ' titties ',
    ' tittyfuck ',
    ' tittywank ',
    ' titwank ',
    ' tosser ',
    ' turd ',
    ' tw4t ',
    ' twat ',
    ' twathead ',
    ' twatty ',
    ' twunt ',
    ' twunter ',
    ' v14gra ',
    ' v1gra ',
    ' vagina ',
    ' viagra ',
    ' vulva ',
    ' w00se ',
    ' wang ',
    ' wank ',
    ' wanker ',
    ' wanky ',
    ' whoar ',
    ' whore ',
    ' willies ',
    ' willy ',
    ' xrated ',
    ' xxx '    
]


# In[ ]:


replace_with_fuck = []

for swear in swear_words:
    if swear[1:(len(swear)-1)] not in embeddings_index:
        replace_with_fuck.append(swear)
        
replace_with_fuck = '|'.join(replace_with_fuck)


# In[ ]:


import re
def handle_swears(text):
    text = re.sub(replace_with_fuck, ' fuck ', text)
    return text
df['comment_text'] = df['comment_text'].apply(lambda x: handle_swears(x))
gc.collect()


# In[ ]:


vocab = build_vocab(df['comment_text'])
oov = check_coverage(vocab, embeddings_index)


# In[ ]:


print(oov[:500])
del(oov,vocab)
gc.collect()


# In[ ]:


from wordsegment import load, segment
load()
def splitit(text, embeddings_index):
    for words in text.split():
        if words not in embeddings_index:
            text=text.replace(words,(" ".join(segment(words))))
    return text
df['comment_text'] = df['comment_text'].apply(lambda x: splitit(x,embeddings_index))




# In[ ]:


vocab = build_vocab(df['comment_text'])
oov = check_coverage(vocab, embeddings_index)
print(oov[:500])


# In[ ]:


del(oov,vocab)
gc.collect()


# In[ ]:


train = df.iloc[:1804874,:]
test = df.iloc[1804874:,:]

train.head()


# In[ ]:


del(df)
gc.collect()


# In[ ]:


train_orig = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
train = pd.concat([train,train_orig[['target']]],axis=1)
train.head()


# In[ ]:


del(train_orig)
gc.collect()


# In[ ]:


train['target'] = np.where(train['target'] >= 0.5, True, False)


# In[ ]:


from keras.preprocessing.text import Tokenizer
MAX_NUM_WORDS = 100000
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train[TEXT_COLUMN])

MAX_SEQUENCE_LENGTH = 256
def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


# In[ ]:


gc.collect()


# In[ ]:


EMBEDDINGS_DIMENSION = 300
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,EMBEDDINGS_DIMENSION))


# In[ ]:


num_words_in_embedding = 0

for word, i in tokenizer.word_index.items():
    if word in embeddings_index.vocab:
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector        
        num_words_in_embedding += 1


# In[ ]:


from keras.preprocessing.sequence import pad_sequences
train_text = pad_text(train[TEXT_COLUMN], tokenizer)
train_labels = train[TOXICITY_COLUMN]


# In[ ]:


from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import CuDNNGRU,concatenate,Dense, Input, Dropout, LSTM,Conv1D, Activation, Bidirectional,GlobalMaxPooling1D,GlobalAveragePooling1D ,CuDNNLSTM, GlobalMaxPool1D, SpatialDropout1D
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
def Toxic_model():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                            EMBEDDINGS_DIMENSION,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
    embeddings = embedding_layer(sequence_input)
    X = SpatialDropout1D(0.2)(embeddings)
    X = Bidirectional(CuDNNGRU(64, return_sequences=True))(X)
    X = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(X)
    avg_pool1 = GlobalAveragePooling1D()(X)
    max_pool1 = GlobalMaxPooling1D()(X)     

    X = concatenate([avg_pool1, max_pool1])
    X = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=sequence_input, outputs=X)
    return model
model = Toxic_model()


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])


# In[ ]:


gc.collect()
model.summary()


# In[ ]:


model.fit(train_text,train_labels, epochs = 5, batch_size = 1000, shuffle=True,verbose=1)


# In[ ]:


submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))
submission.reset_index(drop=False, inplace=True)
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

