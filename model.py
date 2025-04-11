from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from pathlib import Path
import cv2
import numpy as np


class CNN_Model(object):
    def __init__(self, weight_path=None):
        self.weight_path = weight_path
        self.model = None

    def build_model(self, rt=False):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

        if self.weight_path is not None:
            self.model.load_weights(self.weight_path)

        if rt:
            return self.model

    @staticmethod
    
    def load_data():
        dataset_dir = './dataCNN/'
        images = []
        labels = []
    
        unchoice_paths = list(Path(dataset_dir + 'unchoice/').glob("*.jpg"))
        choice_paths = list(Path(dataset_dir + 'choice/').glob("*.jpg"))
        
        print(f"Số ảnh unchoice: {len(unchoice_paths)}")
        print(f"Số ảnh choice: {len(choice_paths)}")
    
        for img_path in unchoice_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = to_categorical(0, num_classes=2)
            images.append(img / 255.0)
            labels.append(label)
    
        for img_path in choice_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = to_categorical(1, num_classes=2)
            images.append(img / 255.0)
            labels.append(label)
    
        if not images:
            raise ValueError("❌ Không tìm thấy dữ liệu để train. Kiểm tra lại thư mục dataCNN/.")
    
        datasets = list(zip(images, labels))
        np.random.shuffle(datasets)
        images, labels = zip(*datasets)
        images = np.array(images)
        labels = np.array(labels)
    
        return images, labels


    def train(self):
        images, labels = self.load_data()

        # Build model
        self.build_model(rt=False)

        # Compile model
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizers.Adam(1e-3),
            metrics=['acc']
        )

        # Callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor='val_acc', factor=0.2, patience=5, verbose=1
        )
        cpt_save = ModelCheckpoint(
            './weight.h5', save_best_only=True, monitor='val_acc', mode='max'
        )

        print("Training......")
        self.model.fit(
            images, labels,
            callbacks=[cpt_save, reduce_lr],
            verbose=1,
            epochs=10000,  # Có thể tăng số epoch lên nếu muốn
            validation_split=0.15,
            batch_size=32,
            shuffle=True
        )


if __name__ == "__main__":
    cnn = CNN_Model()
    cnn.train()
