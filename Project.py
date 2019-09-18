#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:18:32 2019

@author: roopali
"""

import pandas as pd
import tensorflow as tf

## Importing already randomly divided dataset using R
df_train=pd.read_csv('Train.csv')
df_train=df_train.drop(columns=['Unnamed: 0'])
df_test=pd.read_csv('Test.csv')
df_test=df_test.drop(columns=['Unnamed: 0'])

train=df_train.drop(columns=['Cannabis','Coke','Ecstasy','LSD','Mushrooms'])
test=df_test.drop(columns=['Cannabis','Coke','Ecstasy','LSD','Mushrooms'])

features=['V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13']
feature_columns = [tf.feature_column.numeric_column(key) for key in features]

## Structure of the network was chosen by trial

### For Cannabis

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[8,8,8], n_classes=2,model_dir='tmp/model')
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['Cannabis'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['Cannabis'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)

### For Coke

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[8,8,8], n_classes=2,model_dir='tmp/model')

train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['Coke'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['Coke'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)

### For Ecstasy

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[8,8,8], n_classes=2,model_dir='tmp/model')

train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['Ecstasy'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['Ecstasy'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)

### For LSD
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['LSD'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['LSD'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)

### For Mushrooms
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['Mushrooms'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['Mushrooms'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)


#### Without gender and race
df_train=pd.read_csv('Train_new.csv')
df_train=df_train.drop(columns=['Unnamed: 0'])
df_test=pd.read_csv('Test_new.csv')
df_test=df_test.drop(columns=['Unnamed: 0'])

train=df_train.drop(columns=['Cannabis','Coke','Ecstasy','LSD','Mushrooms'])
test=df_test.drop(columns=['Cannabis','Coke','Ecstasy','LSD','Mushrooms'])

features=['V2','V4','V5','V7','V8','V9','V10','V11','V12','V13']
feature_columns = [tf.feature_column.numeric_column(key) for key in features]

### For Cannabis
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[8,8,8], n_classes=2,model_dir='tmp/model')
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['Cannabis'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['Cannabis'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)

### For Coke
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['Coke'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['Coke'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)

### For Ecstasy
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['Ecstasy'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['Ecstasy'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)

### For LSD
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['LSD'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['LSD'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)

### For Mushrooms
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=train, y=df_train['Mushrooms'], num_epochs=15, shuffle=True)
classifier.train(input_fn=train_input_fn,steps=1000)
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test,y=df_test['Mushrooms'],num_epochs=15,shuffle=False)
classifier.evaluate(input_fn=test_input_fn)
