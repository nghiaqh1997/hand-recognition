def cnn_model():
    num_of_classes = get_num_of_classes()
    model = keras.Sequential()
    model.add(layers.Conv2D(16,(2,2), input_shape=(image_x,image_y,1), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(layers.Conv2D(32,(3,3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(3,3),padding='same'))
    model.add(layers.Conv2D(64,(5,5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_of_classes,activation="softmax"))
    #model.summary()
    model.compile(
        optimizer = keras.optimizers.SGD(learning_rate=0.01),
        loss = keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    filepath="cnn_model_keras2.h5"
    checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='val_accuracy',
        verbose=1, save_best_only=True, mode='max'
    )
    callbacks_list = [checkpoint1]
    tf.keras.utils.plot_model(model=model, to_file="model.png", show_shapes=True)
    return model,callbacks_list