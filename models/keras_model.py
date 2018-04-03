import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, \
    AveragePooling2D, Flatten


class Model:
    def __init__(self):
        self.model = Sequential([
            # first block
            Conv2D(6, 3, input_shape=(224, 224, 3)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(9, 3),
            BatchNormalization(),
            Activation('relu'),
            AveragePooling2D(2, 2),

            # second block
            Conv2D(12, 3),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(16, 3),
            BatchNormalization(),
            Activation('relu'),
            AveragePooling2D(2, 2),

            # third block
            Conv2D(32, 3),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, 3),
            BatchNormalization(),
            Activation('relu'),
            AveragePooling2D(2, 2),

            # forth block
            Conv2D(128, 3),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, 3),
            BatchNormalization(),
            Activation('relu'),

            # transition to classes
            AveragePooling2D(20),
            Flatten(),
            Dense(512),
            BatchNormalization(),
            Activation('relu'),
            Dense(100),
            Activation('sigmoid')
        ])

        self.model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def predict(self, inputs):
        return self.model.predict(inputs)

    def summary(self):
        return self.model.summary()


if __name__ == '__main__':
    sample_image = np.random.random((1, 224, 224, 3))
    model = Model()
    preds = model.predict(sample_image)
    print(model.summary())
