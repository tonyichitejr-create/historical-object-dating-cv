from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224,224)

BATCH_SIZE = 32


def create_generators(train_dir,val_dir,test_dir):

    train_gen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=20,

        zoom_range=0.2,

        brightness_range=[0.8,1.2],

        horizontal_flip=True

    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(

        train_dir,

        target_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="categorical"

    )

    validation = test_gen.flow_from_directory(

        val_dir,

        target_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        class_mode="categorical"

    )

    test = test_gen.flow_from_directory(

        test_dir,

        target_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=False,

        class_mode="categorical"

    )

    return train,validation,test
