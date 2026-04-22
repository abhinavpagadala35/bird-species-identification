import os
import tensorflow as tf

def load_bird_dataset(batch_size=32, image_size=(224, 224), debug_fast_run=False):
    print("Downloading Caltech Birds 2011 dataset from official canonical source...")
    
    dataset_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    data_dir = tf.keras.utils.get_file(
        origin=dataset_url,
        extract=True,
        cache_dir='.',
        cache_subdir='.'
    )
    
    base_dir = os.path.join(os.path.dirname(data_dir), 'CUB_200_2011', 'images')
    if not os.path.exists(base_dir):
        # Handle keras default extraction making a directory named after tarball
        base_dir = os.path.join(os.path.dirname(data_dir), 'CUB_200_2011.tgz', 'CUB_200_2011', 'images')
    
    print(f"Loading images from {base_dir}...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int'
    )
    
    class_names = train_ds.class_names
    print(f"Num classes loaded natively: {len(class_names)}")
    
    if debug_fast_run:
        print("DEBUG MODE: Taking a tiny fraction of the dataset for testing.")
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(2)
        
    def preprocess(image, label):
        # Apply standard DenseNet preprocessing function map over casted raw floats!
        return tf.keras.applications.densenet.preprocess_input(tf.cast(image, tf.float32)), label

    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal"),
      tf.keras.layers.RandomRotation(0.2),
      tf.keras.layers.RandomZoom(0.2),
    ])

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
