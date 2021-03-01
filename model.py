from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Input, BatchNormalization
from keras.models import Model

def conv2d_block( input_tensor, n_filters, kernel_size = (3,3), name="contraction"):
  "Add 2 conv layer"
  x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal', 
             padding='same',activation="relu", name=name+'_1')(input_tensor)
  
  x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal', 
             padding='same',activation="relu",name=name+'_2')(x)
  return x

def baseModel(input_shape):
    inp = Input( shape=input_shape )

    d1 = conv2d_block( inp, 64, name="contraction_1")
    p1 = MaxPooling2D( pool_size=(2,2), strides=(2,2))(d1)
    p1 = BatchNormalization(momentum=0.8)(p1)
    p1 = Dropout(0.1)(p1)

    d2 = conv2d_block( p1, 128, name="contraction_2_1" )
    p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d2)
    p2 = BatchNormalization(momentum=0.8)(p2)
    p2 = Dropout(0.1)(p2)

    d3 = conv2d_block( p2, 256, name="contraction_3_1")
    p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d3)
    p3 = BatchNormalization(momentum=0.8)(p3)
    p3 = Dropout(0.1)(p3)

    d4 = conv2d_block(p3,512, name="contraction_4_1")
    p4 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d4)
    p4 = BatchNormalization(momentum=0.8)(p4)
    p4 = Dropout(0.1)(p4)

    d5 = conv2d_block(p4,512, name="contraction_5_1")

    u1 = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same')(d5)
    u1 = concatenate([u1,d4])
    u1 = Dropout(0.1)(u1)
    c1 = conv2d_block(u1, 512, name="expansion_1")

    u2 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(c1)
    u2 = concatenate([u2,d3])
    u2 = Dropout(0.1)(u2)
    c2 = conv2d_block(u2, 256, name="expansion_2")

    u3 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(c2)
    u3 = concatenate([u3,d2])
    u3 = Dropout(0.1)(u3)
    c3 = conv2d_block(u3, 128, name="expansion_3")

    u4 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(c3)
    u4 = concatenate([u4,d1])
    u4 = Dropout(0.1)(u4)
    c4 = conv2d_block(u4,64, name="expansion_4")

    out = Conv2D(1, (1,1), name="output", activation='sigmoid')(c4)

    unet = Model( inp, out  ,name="unet_base")

    return unet;

def unet_v4(input_shape):
    inp = Input( shape=input_shape )

    d1 = conv2d_block( inp, 64, name="contraction_1")
    p1 = MaxPooling2D( pool_size=(2,2), strides=(2,2))(d1)
    p1 = BatchNormalization(momentum=0.8)(p1)
    p1 = Dropout(0.1)(p1)

    d2 = conv2d_block( p1, 128, name="contraction_2_1" )
    p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d2)
    p2 = BatchNormalization(momentum=0.8)(p2)
    p2 = Dropout(0.1)(p2)

    d3 = conv2d_block( p2, 256, name="contraction_3_1")
    p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d3)
    p3 = BatchNormalization(momentum=0.8)(p3)
    p3 = Dropout(0.1)(p3)

    d4 = conv2d_block(p3,512, name="contraction_4_1")
    p4 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d4)
    p4 = BatchNormalization(momentum=0.8)(p4)
    p4 = Dropout(0.1)(p4)

    d5 = conv2d_block(p4,512, name="contraction_5_1")

    u1 = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same')(d5)
    u1 = concatenate([u1,d4])
    u1 = Dropout(0.1)(u1)
    c1 = conv2d_block(u1, 512, name="expansion_1")

    u2 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(c1)
    u2 = concatenate([u2,d3])
    u2 = Dropout(0.1)(u2)
    c2 = conv2d_block(u2, 256, name="expansion_2")

    u3 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(c2)
    u3 = concatenate([u3,d2])
    u3 = Dropout(0.1)(u3)
    c3 = conv2d_block(u3, 128, name="expansion_3")

    u4 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(c3)
    u4 = concatenate([u4,d1])
    u4 = Dropout(0.1)(u4)
    c4 = conv2d_block(u4,64, name="expansion_4")

    out = Conv2D(1, (1,1), name="output", activation='sigmoid')(c4)

    unet = Model( inp, out  ,name="unet_base")

    return unet;


def unet_v1(input_shape):
    inp = Input( shape=input_shape )

    d1 = conv2d_block( inp, 64, name="contraction_1", kernel_size=(5,5))
    p1 = MaxPooling2D( pool_size=(2,2), strides=(2,2))(d1)
    p1 = BatchNormalization(momentum=0.8)(p1)
    p1 = Dropout(0.1)(p1)

    d2 = conv2d_block( p1, 128, name="contraction_2_1", kernel_size=(5,5) )
    p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d2)
    p2 = BatchNormalization(momentum=0.8)(p2)
    p2 = Dropout(0.1)(p2)

    d3 = conv2d_block( p2, 256, name="contraction_3_1", kernel_size=(5,5))
    p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d3)
    p3 = BatchNormalization(momentum=0.8)(p3)
    p3 = Dropout(0.1)(p3)

    d4 = conv2d_block(p3,512, name="contraction_4_1", kernel_size=(5,5))
    p4 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d4)
    p4 = BatchNormalization(momentum=0.8)(p4)
    p4 = Dropout(0.1)(p4)

    d5 = conv2d_block(p4,512, name="contraction_5_1")

    u1 = Conv2DTranspose(512, kernel_size=(5,5), strides = (2, 2), padding = 'same')(d5)
    u1 = concatenate([u1,d4])
    u1 = Dropout(0.1)(u1)
    c1 = conv2d_block(u1, 512, name="expansion_1")

    u2 = Conv2DTranspose(256, kernel_size=(5,5), strides = (2, 2), padding = 'same')(c1)
    u2 = concatenate([u2,d3])
    u2 = Dropout(0.1)(u2)
    c2 = conv2d_block(u2, 256, name="expansion_2")

    u3 = Conv2DTranspose(128, kernel_size=(5,5), strides = (2, 2), padding = 'same')(c2)
    u3 = concatenate([u3,d2])
    u3 = Dropout(0.1)(u3)
    c3 = conv2d_block(u3, 128, name="expansion_3")

    u4 = Conv2DTranspose(64, kernel_size=(5,5), strides = (2, 2), padding = 'same')(c3)
    u4 = concatenate([u4,d1])
    u4 = Dropout(0.1)(u4)
    c4 = conv2d_block(u4,64, name="expansion_4")

    out = Conv2D(1, (1,1), name="output", activation='sigmoid')(c4)

    unet = Model( inp, out, name="unet_v1")

    return unet;

def unet_v2(input_shape):
    inp = Input( shape=input_shape )

    d1 = conv2d_block( inp, 64, name="contraction_1")
    p1 = MaxPooling2D( pool_size=(2,2), strides=(2,2))(d1)
    p1 = BatchNormalization(momentum=0.8)(p1)
    p1 = Dropout(0.1)(p1)

    d2 = conv2d_block( p1, 64, name="contraction_2_1" )
    p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d2)
    p2 = BatchNormalization(momentum=0.8)(p2)
    p2 = Dropout(0.1)(p2)

    d3 = conv2d_block( p2, 128, name="contraction_3_1")
    p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d3)
    p3 = BatchNormalization(momentum=0.8)(p3)
    p3 = Dropout(0.1)(p3)

    d4 = conv2d_block(p3,128, name="contraction_4_1")
    p4 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d4)
    p4 = BatchNormalization(momentum=0.8)(p4)
    p4 = Dropout(0.1)(p4)

    d5 = conv2d_block( p4, 256, name="contraction_5_1")
    p5 = MaxPooling2D( pool_size=(2,2), strides=(2,2))(d5)
    p5 = BatchNormalization(momentum=0.8)(p5)
    p5 = Dropout(0.1)(p5)

    d6 = conv2d_block( p5, 256, name="contraction_6_1" )
    p6 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d6)
    p6 = BatchNormalization(momentum=0.8)(p6)
    p6 = Dropout(0.1)(p6)

    d7 = conv2d_block( p6, 512, name="contraction_7_1")
    p7 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d7)
    p7 = BatchNormalization(momentum=0.8)(p7)
    p7 = Dropout(0.1)(p7)

    d8 = conv2d_block(p7,512, name="contraction_8_1")
    p8 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d8)
    p8 = BatchNormalization(momentum=0.8)(p8)
    p8 = Dropout(0.1)(p8)

    d9 = conv2d_block(p8,512, name="contraction_9_1")

    u1 = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same')(d9)
    u1 = concatenate([u1,d8])
    u1 = Dropout(0.1)(u1)
    c1 = conv2d_block(u1, 512, name="expansion_1")

    u2 = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same')(c1)
    u2 = concatenate([u2,d7])
    u2 = Dropout(0.1)(u2)
    c2 = conv2d_block(u2, 512, name="expansion_2")

    u3 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(c2)
    u3 = concatenate([u3,d6])
    u3 = Dropout(0.1)(u3)
    c3 = conv2d_block(u3, 256, name="expansion_3")

    u4 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(c3)
    u4 = concatenate([u4,d5])
    u4 = Dropout(0.1)(u4)
    c4 = conv2d_block(u4,256, name="expansion_4")

    u5 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(c4)
    u5 = concatenate([u5,d4])
    u5 = Dropout(0.1)(u5)
    c5 = conv2d_block(u5, 128, name="expansion_5")

    u6 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6,d3])
    u6 = Dropout(0.1)(u6)
    c6 = conv2d_block(u6, 128, name="expansion_6")

    u7 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7,d2])
    u7 = Dropout(0.1)(u7)
    c7 = conv2d_block(u7, 64, name="expansion_7")

    u8 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8,d1])
    u8 = Dropout(0.1)(u8)
    c8 = conv2d_block(u8,64, name="expansion_8")

    out = Conv2D(1, (1,1), name="output", activation='sigmoid')(c8)

    unet = Model( inp, out  ,name="unet_v2")

    return unet;

def unet_v3(input_shape):
    inp = Input( shape=input_shape )

    d1 = conv2d_block( inp, 16, name="contraction_1")
    p1 = MaxPooling2D( pool_size=(2,2), strides=(2,2))(d1)
    p1 = BatchNormalization(momentum=0.8)(p1)
    p1 = Dropout(0.1)(p1)

    d2 = conv2d_block( p1, 32, name="contraction_2_1" )
    p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d2)
    p2 = BatchNormalization(momentum=0.8)(p2)
    p2 = Dropout(0.1)(p2)

    d3 = conv2d_block( p2, 64, name="contraction_3_1")
    p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d3)
    p3 = BatchNormalization(momentum=0.8)(p3)
    p3 = Dropout(0.1)(p3)

    d4 = conv2d_block(p3,128, name="contraction_4_1")
    p4 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d4)
    p4 = BatchNormalization(momentum=0.8)(p4)
    p4 = Dropout(0.1)(p4)

    d5 = conv2d_block( p4, 256, name="contraction_5_1")
    p5 = MaxPooling2D( pool_size=(2,2), strides=(2,2))(d5)
    p5 = BatchNormalization(momentum=0.8)(p5)
    p5 = Dropout(0.1)(p5)

    d6 = conv2d_block( p5, 512, name="contraction_6_1" )
    p6 = MaxPooling2D(pool_size=(2,2), strides=(2,2) )(d6)
    p6 = BatchNormalization(momentum=0.8)(p6)
    p6 = Dropout(0.1)(p6)

    d7 = conv2d_block(p6,512, name="contraction_7_1")

    u1 = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same')(d7)
    u1 = concatenate([u1,d6])
    u1 = Dropout(0.1)(u1)
    c1 = conv2d_block(u1, 512, name="expansion_1")

    u2 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(c1)
    u2 = concatenate([u2,d5])
    u2 = Dropout(0.1)(u2)
    c2 = conv2d_block(u2,256, name="expansion_2")

    u3 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same')(c2)
    u3 = concatenate([u3,d4])
    u3 = Dropout(0.1)(u3)
    c3 = conv2d_block(u3, 128, name="expansion_3")

    u4 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same')(c3)
    u4 = concatenate([u4,d3])
    u4 = Dropout(0.1)(u4)
    c4 = conv2d_block(u4, 64, name="expansion_4")

    u5 = Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = 'same')(c4)
    u5 = concatenate([u5,d2])
    u5 = Dropout(0.1)(u5)
    c5 = conv2d_block(u5, 32, name="expansion_5")

    u6 = Conv2DTranspose(16, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6,d1])
    u6 = Dropout(0.1)(u6)
    c6 = conv2d_block(u6,16, name="expansion_6")

    out = Conv2D(1, (1,1), name="output", activation='sigmoid')(c6)

    unet = Model( inp, out  ,name="unet_v3")

    return unet;