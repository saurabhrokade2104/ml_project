import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset", help="root dataset dir (contains train/ and test/ or val/)")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--save_path", type=str, default="models/attention_model.h5")
args = parser.parse_args()

IMG_SIZE = args.img_size
BATCH = args.batch_size
DATA_DIR = args.data_dir

# Try common folder names
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")
if not os.path.exists(train_dir):
    # maybe dataset has Engaged/ Not Engaged directly - use DATA_DIR
    train_dir = DATA_DIR
if not os.path.exists(val_dir) and os.path.exists(test_dir):
    val_dir = test_dir

print("Using train dir:", train_dir)
print("Using val dir:", val_dir)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=(0.8,1.2),
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='binary',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='binary',
    shuffle=False
)

base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                         include_top=False,
                                         weights='imagenet', pooling='avg')
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
checkpoint = ModelCheckpoint(args.save_path, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    train_gen,
    epochs=args.epochs,
    validation_data=val_gen,
    callbacks=[checkpoint, reduce_lr]
)

# Optional fine-tune
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen, epochs=3, validation_data=val_gen, callbacks=[checkpoint, reduce_lr])
model.save(args.save_path)
print("Saved model to", args.save_path)