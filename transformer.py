#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping # type: ignore


# ## Load the dataset

# In[ ]:


# Load Data
#from google.colab import files
#uploaded = files.upload()

# Assuming 'mallika_combined.pkl' is the name of the uploaded file
data = pd.read_pickle('final_combined_dataset.pickle')




# In[ ]:


# Preprocess Data: Convert boards into 6x7x2 representation
def preprocess_data(data):
    boards = []
    labels = []
    for _, item in data.iterrows():
        board = np.array(item['board'])

        # Convert to 6x7x2 format
        if board.shape == (6, 7, 2):
            board_6x7x2 = board
        else:
            board_6x7x2 = np.zeros((6, 7, 2))
            board_6x7x2[:, :, 0] = (board == 1).astype(int)
            board_6x7x2[:, :, 1] = (board == -1).astype(int)

        label = item['recommended_column']
        boards.append(board_6x7x2)
        labels.append(label)

    boards = np.array(boards)
    labels = np.array(labels)

    # One-hot encode labels (7 possible moves)
    labels = tf.keras.utils.to_categorical(labels, num_classes=7)
    return boards, labels

# Get processed data
boards, labels = preprocess_data(data)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(boards, labels, test_size=0.2, random_state=42)

# Reshape Data for Transformer (Flatten the board)
num_samples, n, m, channels = X_train.shape  # (num_samples, 6, 7, 2)
X_train = X_train.reshape(num_samples, n * m, channels)
X_val = X_val.reshape(X_val.shape[0], n * m, channels)

# Debugging Step: Check shape before proceeding
print(f"X_train shape: {X_train.shape}")  # Expected: (num_samples, 42, 2)
print(f"y_train shape: {y_train.shape}")  # Expected: (num_samples, 7)


# In[ ]:


import tensorflow as tf

class PositionalIndex(tf.keras.layers.Layer):
    def call(self, x):
        bs = tf.shape(x)[0]  # Extract batch size
        number_of_vectors = tf.shape(x)[1]  # Count the number of vectors (should be m*n)
        indices = tf.range(number_of_vectors)  # Index for each vector
        indices = tf.expand_dims(indices, 0)  # Reshape appropriately
        return tf.tile(indices, [bs, 1])  # Repeat for each batch


class ClassTokenIndex(tf.keras.layers.Layer):
    def call(self, x):
        bs = tf.shape(x)[0]  # Extract batch size
        number_of_vectors = 1  # We want just 1 vector for the class token
        indices = tf.range(number_of_vectors)  # Index for the vector
        indices = tf.expand_dims(indices, 0)  # Reshape appropriately
        return tf.tile(indices, [bs, 1])  # Repeat for each batch


# ## Transformer

# In[ ]:


def build_ViT(n,m,block_size,hidden_dim,num_layers,num_heads,key_dim,value_dim,mlp_dim,dropout_rate,num_classes):
    # n is number of rows of blocks
    # m is number of cols of blocks
    # block_size is number of pixels (with rgb) in each block
    inp = tf.keras.layers.Input(shape=(n*m,block_size))
    mid = tf.keras.layers.Dense(hidden_dim)(inp) # transform to vectors with different dimension
    # the positional embeddings
    inp2 = PositionalIndex()(inp)
    emb = tf.keras.layers.Embedding(input_dim=n*m, output_dim=hidden_dim)(inp2) # learned positional embedding for each of the n*m possible possitions
    mid = tf.keras.layers.Add()([mid, emb]) # for some reason, tf.keras.layers.Add causes an error, but + doesn't?
    # create and append class token to beginning of all input vectors
    tokenInd = ClassTokenIndex()(mid)
    token = tf.keras.layers.Embedding(input_dim=1, output_dim=hidden_dim)(tokenInd)
    mid = tf.keras.layers.Concatenate(axis=1)([token, mid])

    for l in range(num_layers): # how many Transformer Head layers are there?
        ln  = tf.keras.layers.LayerNormalization()(mid) # normalize
        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,value_dim=value_dim)(ln,ln,ln) # self attention!
        add = tf.keras.layers.Add()([mid,mha]) # add and norm
        ln  = tf.keras.layers.LayerNormalization()(add)
        den = tf.keras.layers.Dense(mlp_dim,activation='gelu')(ln) # maybe should be relu...who knows...
        den = tf.keras.layers.Dropout(dropout_rate)(den) # regularization
        den = tf.keras.layers.Dense(hidden_dim)(den) # back to the right dimensional space
        den = tf.keras.layers.Dropout(dropout_rate)(den)
        mid = tf.keras.layers.Add()([den,add]) # add and norm again

    fl = mid[:,0,:] # just grab the class token for each image in batch
    ln = tf.keras.layers.LayerNormalization()(fl)
    clas = tf.keras.layers.Dense(num_classes,activation='softmax')(ln) # probability that the image is in each category
    mod = tf.keras.models.Model(inp,clas)
    mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return mod


# In[ ]:


# Model hyperparameters
n = 6
m = 7
block_size = 2
hidden_dim = 32
num_layers = 4
num_heads = 4
key_dim = hidden_dim // num_heads  # Good practice for key_dim to be hidden_dim//num_heads
value_dim = key_dim * 2
mlp_dim = hidden_dim
dropout_rate = 0.1
num_classes = 7  # Output classes for classification

# Build the Transformer model
trans = build_ViT(n, m, block_size, hidden_dim, num_layers, num_heads,
                  key_dim, value_dim, mlp_dim, dropout_rate, num_classes)

# Display model summary
trans.summary()


# In[ ]:


trans.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# In[ ]:





# In[26]:




early_stopping = EarlyStopping(
    monitor='val_loss',  # Stop if validation loss doesn't improve
    patience=5,          # Number of epochs to wait before stopping
    restore_best_weights=True  # Restore model to the best state
)

history = trans.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,  # Adjust based on performance
                    batch_size=128,
                    callbacks=[early_stopping])  # Tune based on memory and dataset size


# ## Test

# In[27]:


# plot results
trans.evaluate(X_val, y_val)


# In[28]:


# export model results
trans.save('transformer_model.h5')


# In[30]:


import matplotlib.pyplot as plt

# Extract loss and accuracy values
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Get number of epochs
epochs = range(1, len(train_loss) + 1)

# Create a figure
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')  # Blue circles
plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')  # Red stars
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')  # Blue circles
plt.plot(epochs, val_acc, 'r*-', label='Validation Accuracy')  # Red stars
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy Over Epochs')
plt.legend()

# Show plots
plt.show()

