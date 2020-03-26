from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow as tf
import logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

tf.flags.DEFINE_float('learning_rate', 1e-4, 'learning rate to fine tune the last layer')
tf.flags.DEFINE_integer('img_width', 360, 'resized image width')
tf.flags.DEFINE_integer('img_height', 360, 'resized image height')
#tf.flags.DEFINE_boolean('horizontal_flip', False, 'whether to flip images horizontally')
tf.flags.DEFINE_boolean('dropout_flag', True, 'whether to dropout during training')
tf.flags.DEFINE_boolean('finetuning_last', False, 'whether to finetuning the last layer first')
tf.flags.DEFINE_integer('epoch1', 50, 'the number of epochs to train the last layer')
tf.flags.DEFINE_integer('epoch2', 40, 'the number of epochs to train the whole network')
tf.flags.DEFINE_string('saved_model_name', 'keras_inception.h5', 'the saved model file name')
tf.flags.DEFINE_integer('batch_size', 12, 'batch size')
FLAGS = tf.flags.FLAGS

logger.info('learning_rate: %f', FLAGS.learning_rate)
logger.info('resized img with: %d', FLAGS.img_width)
logger.info('resized img height: %d', FLAGS.img_height)
logger.info('whether to dropout with a ratio of 0.5: %d', FLAGS.dropout_flag)
logger.info('whether to finetuning the last layer first: %d', FLAGS.finetuning_last)
logger.info('training the last layer for %d epoch', FLAGS.epoch1)
logger.info('training the whole networ for %d epoch', FLAGS.epoch2)
logger.info('saved model file name: %s', FLAGS.saved_model_name)
logger.info('batch size: %d', FLAGS.batch_size)
#train_data_dir = "./dataset_agg/sample_dataset/training_dataset"
#validation_data_dir = "./dataset_agg/sample_dataset/validation_dataset"
#train_data_dir = "./dataset_agg/sample_dataset/training_dataset"
#validation_data_dir = "./dataset_agg/sample_dataset/validation_dataset"
#test_data_dir = "./dataset_agg/sample_dataset/test_dataset"
train_data_dir = "./training_1000_2000"
validation_data_dir = "./validation_1000_2000"
test_data_dir = './test_1000_2000'
nb_train_samples = 24000
nb_validation_samples = 3000
nb_test_samples = 2794

#config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                        allow_soft_placement=True, device_count = {'CPU': 16})
#session = tf.Session(config=config)
#K.set_session(session)

base_model = applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(FLAGS.img_width, FLAGS.img_height, 3))
x = base_model.output
#use dropout of ratio 0.5
if FLAGS.dropout_flag == True:
    print('hello world')
    x = Dropout(0.5)(x)
x = GlobalAveragePooling2D(name='avg_pool')(x)
#print(x.shape)
# x = Dense(1024, activation='relu', name='ff')(x)
predictions = Dense(6, activation='softmax', name='predictions')(x)

# this is the model we will train
model_final = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
if FLAGS.finetuning_last == True:
    print('freezing pretrained network')
    for layer in base_model.layers:
        layer.trainable = False
   
# compile the model (should be done *after* setting layers to non-trainable)
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=FLAGS.learning_rate), metrics=["accuracy"])
#model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=FLAGS.learning_rate), metrics=["accuracy"])

#print(model_final.summary())

train_datagen = ImageDataGenerator(fill_mode="nearest", zoom_range=0.1, \
                                   width_shift_range=0.1, height_shift_range=0.1, rotation_range=0)

vali_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

#train_datagen.flow(x, y, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset)
#train_datagen = ImageDataGenerator()

#test_datagen = ImageDataGenerator()

#train_datagen.filenames()


#test_datagen.flow()

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(FLAGS.img_height, FLAGS.img_width), \
                                                    batch_size=FLAGS.batch_size, class_mode="categorical", shuffle=True)

'''for i in train_generator:
    print(train_generator.batch_index)
    idx = (train_generator.batch_index - 1) * train_generator.batch_size
    print(train_generator.filenames[idx : idx + train_generator.batch_size])'''
    
validation_generator = vali_datagen.flow_from_directory(validation_data_dir,target_size=(FLAGS.img_height, FLAGS.img_width), \
                                                        batch_size=FLAGS.batch_size, class_mode="categorical", shuffle=False)
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(FLAGS.img_height, FLAGS.img_width), \
                                                  batch_size=FLAGS.batch_size, class_mode='categorical', shuffle=False)
'''for i, j, filename in validation_generator:
    print(j)
    print(filename)'''
#test_datagen.flow()

#print(train_generator.class_indices)
#print(train_generator.classes)
#print(train_generator.filenames)

# Save the model according to the conditions  
checkpoint = ModelCheckpoint(FLAGS.saved_model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=2, mode='auto')

#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)))

# Train the model 
model_final.fit_generator(train_generator, steps_per_epoch=nb_train_samples/float(FLAGS.batch_size), epochs=FLAGS.epoch1, \
                          validation_data=validation_generator, validation_steps=np.ceil(nb_validation_samples/float(FLAGS.batch_size)), callbacks=[checkpoint, early])


#for layer in base_model.layers:
#    layer.trainable = True

#model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=FLAGS.learning_rate), metrics=["accuracy"])

#model_final.fit_generator(train_generator, samples_per_epoch=nb_train_samples, epochs=FLAGS.epoch2, \
#                         validation_data=validation_generator, validation_steps=np.ceil(nb_validation_samples/float(batch_size)), callbacks=[checkpoint, early])

del model_final
print('***************************************')
print('start to predict on the test set using saved model best on the validation set')
model_final = load_model(FLAGS.saved_model_name)
result1 = model_final.predict_generator(test_generator, steps=np.ceil(nb_test_samples/float(FLAGS.batch_size)), use_multiprocessing=False, verbose=1)
result2 = model_final.evaluate_generator(test_generator, steps=np.ceil(nb_test_samples/float(FLAGS.batch_size)), use_multiprocessing=False)

print('result2')
y_pred = np.argmax(result1, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['0', '1', '2', '3', '4', '5']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
#final_model.evaluate(np.expand_dims(X, axis=3), y, batch_size=32)
#model_final.train_on_batch(x, y, sample_weight, class_weight)