import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
#from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import os
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.callbacks import TensorBoard


AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SIZE=32
PATCH_SIZE=4
NUM_LAYERS=8
NUM_HEADS=16
MLP_DIM=256
lr=0.001
WEIGHT_DECAY=1e-4
BATCH_SIZE=8
epochs=30


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        #attention takes three inputs: queries, keys, and values,
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        #use the product between the queries and the keys 
        #to know "how much" each element is the sequence is important with the rest
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        #resulting vector, score is divided by a scaling factor based on the size of the embedding
        #scaling fcator is square root of the embeding dimension
        scaled_score = score / tf.math.sqrt(dim_key)
        #the attention scaled_score is then softmaxed
        weights = tf.nn.softmax(scaled_score, axis=-1)
        #Attention(Q, K, V ) = softmax[(QK)/√dim_key]V
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
         
        batch_size = tf.shape(inputs)[0]
        #MSA takes the queries, keys, and values  as input from the previous layer 
        #and projects them using the three linear layers.
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        #print('attention :', attention.shape)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        #self attention of different heads are concatenated  
        output = self.combine_heads(concat_attention)
        return output, weights

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Transfromer block multi-head Self Attention
        self.multiheadselfattention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        out1 = self.layernorm1(inputs)       
        attention_output, weights = self.multiheadselfattention(out1)

        #print("attention output ",attention_output.shape)

        attention_output = self.dropout1(attention_output, training=training)       
        out2 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output)

class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        # create patches based on patch_size
        # image_size/patch_size==0
        num_patches=self.create_patch(image_size,patch_size, channels)
        self.d_model = d_model
        #self.rescale = Rescaling(1./255)
        self.patch_proj= self.create_postional_embedding(num_patches, d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation='relu'),
                Dense(mlp_dim, activation='relu'),#tfa.activations.gelu
                Dropout(dropout),
                Dense(1, activation='linear'),
            ]
        )

    def create_patch(self, image_size, patch_size, channels):
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        return num_patches
    def create_postional_embedding(self,num_patches, d_model):
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches + 1, d_model))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        #print(self.class_emb.shape)
        return Dense(d_model)
   
        
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        #rescale 
        #x = self.rescale(x)
        #x = x/255.
        #print(type(x), x, x/255.)
        # extract the patches from the image
        patches = self.extract_patches(x)
        # Apply the postio embedding
        x = self.patch_proj(patches)        
        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )              
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb        
        for layer in self.enc_layers:
            x = layer(x, training)
        x = self.mlp_head(x[:, 0])
        return x

class TransformerBlock2(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock2, self).__init__()
        # Transfromer block multi-head Self Attention
        self.multiheadselfattention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        out1 = self.layernorm1(inputs)       
        attention_output, weights = self.multiheadselfattention(out1)

        #print("weights output ",weights.shape)

        attention_output = self.dropout1(attention_output, training=training)       
        out2 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output), weights

class VisionTransformer_attn(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
    ):
        super(VisionTransformer_attn, self).__init__()
        # create patches based on patch_size
        # image_size/patch_size==0
        num_patches=self.create_patch(image_size,patch_size, channels)
        self.d_model = d_model
        #self.rescale = Rescaling(1./255)
        self.patch_proj= self.create_postional_embedding(num_patches, d_model)
        self.enc_layers = [
            TransformerBlock2(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        # self.mlp_head = tf.keras.Sequential(
        #     [
        #         Dense(mlp_dim, activation='relu'),
        #         Dense(mlp_dim, activation='relu'),#tfa.activations.gelu
        #         Dropout(dropout),
        #         Dense(1, activation='linear'),
        #     ]
        # )

    def create_patch(self, image_size, patch_size, channels):
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        return num_patches
    def create_postional_embedding(self,num_patches, d_model):
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches + 1, d_model))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        #print(self.class_emb.shape)
        return Dense(d_model)
   
        
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        #rescale 
        #x = self.rescale(x)
        #x = x/255.
        #print(type(x), x, x/255.)
        # extract the patches from the image
        patches = self.extract_patches(x)
        # Apply the postio embedding
        x = self.patch_proj(patches)        
        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )              
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb        
        for layer in self.enc_layers:
            x, attn_map = layer(x, training)
        #x = self.mlp_head(x[:, 0])
        #print(attn_map.shape)
        return attn_map

class VisionTransformer2(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
    ):
        super(VisionTransformer2, self).__init__()
        # create patches based on patch_size
        # image_size/patch_size==0
        num_patches=self.create_patch(image_size,patch_size, channels)
        self.d_model = d_model
        #self.rescale = Rescaling(1./255)
        self.patch_proj= self.create_postional_embedding(num_patches, d_model)
        self.enc_layers = [
            TransformerBlock2(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        # self.mlp_head = tf.keras.Sequential(
        #     [
        #         Dense(mlp_dim, activation='relu'),
        #         Dense(mlp_dim, activation='relu'),#tfa.activations.gelu
        #         Dropout(dropout),
        #         Dense(1, activation='linear'),
        #     ]
        # )

    def create_patch(self, image_size, patch_size, channels):
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        return num_patches
    def create_postional_embedding(self,num_patches, d_model):
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches + 1, d_model))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        #print(self.class_emb.shape)
        return Dense(d_model)
   
        
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        #rescale 
        #x = self.rescale(x)
        x = x/255.
        #print(type(x), x, x/255.)
        # extract the patches from the image
        patches = self.extract_patches(x)
        # Apply the postio embedding
        x = self.patch_proj(patches)        
        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )              
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb        
        for layer in self.enc_layers:
            x, attn_map = layer(x, training)
        #x = self.mlp_head(x[:, 0])
        #print(attn_map.shape)
        return x



if __name__ == '__main__':
    inputs = tf.Variable(tf.random.uniform(shape=(1, 96, 128, 256)))
    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_layers=NUM_LAYERS, 
        num_classes=10,
        d_model=64,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        channels=3,
        dropout=0.1,
    )
    print(model(inputs).shape)