import os
import urllib.request
import tensorflow as tf

def construct_pretrained_model():
    print("Downloading pre-trained DenseNet121 from ImageNet...")
    
    # Ensure data dir exists
    os.makedirs('../data', exist_ok=True)
    
    # 1. Load the pre-trained DenseNet121 with ImageNet weights, INCLUDING the classification top
    model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=True)
    
    # Save the model
    model_path = '../data/elitedensenet_model.h5'
    model.save(model_path)
    print(f"Pre-trained DenseNet121 saved to {model_path}!")

    # 2. Download standard ImageNet labels mapping
    print("Downloading ImageNet class labels...")
    labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    
    import json
    try:
        response = urllib.request.urlopen(labels_url)
        labels = json.loads(response.read())
        
        with open('../data/class_names.txt', 'w') as f:
            for label in labels:
                f.write(f"{label}\n")
        print("ImageNet class names saved successfully!")
    except Exception as e:
        print(f"Error downloading labels: {e}")

if __name__ == '__main__':
    construct_pretrained_model()
