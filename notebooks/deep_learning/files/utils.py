import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def test_marginal_perm(test_data_dir, model, image_size):
    """
    tests the model by permuting the value of each of the three channels and seeing what happens to the (probability of the correct class?)
    
    - for the model trained on the red channel only, only the red channel should make a difference (the rest are the same!)
    - for the model trained on the noisy data, the effect of permuting the non-red channels should be almost zero.
    
    For small samples, repeat for all permutations except the original data.
    """

    # read the entire dataset into an array
    # Perform a prediction on all examples, summing the loss (= 1-p for the correct class)
    # Loop over n:
    #   predict sample n and compute the loss (1-p)
    #   loop over m:
    #       For each channel, swap n for m unless n=m, get the prediction and add the loss to the total for that channel
    #   compute the average of the two losses
    # Compute importance = loss(permuted)/loss(original) [note: just a normalising value]
    # Show visually?
    
    test_images = os.listdir(test_data_dir)
    image_data = []
    image_classes = []
    class_names = ['X', '0']
    tot_correct = 0
    num_images = len(test_images)

    results = []
    
    # Load and score the original images
    loss = 0.
    for image_file in test_images:
        img = tf.keras.utils.load_img(os.path.join(test_data_dir, image_file), target_size=(image_size, image_size))
        img_array = tf.keras.utils.img_to_array(img)

        image_data.append(img_array)
        image_classes.append(image_file[0])

        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = model.predict(img_array, verbose=0)
        scores = tf.nn.softmax(predictions[0])
        score = np.max(scores)
        ans = class_names[np.argmax(scores)]
        correct = (ans == image_file[0]) # Filename is prefixed with class
        tot_correct += correct
        if correct:
            loss += 1-score
        else:
            loss += score
        # print(f"{image_file}: {ans} {score} {correct}")
    loss = loss/num_images
    print(f"Unpermuted loss:{loss}, correct:{tot_correct}/{num_images} {tot_correct/num_images}")
    results.append(('All', loss))
    
    # print(f"image_data shapes: {image_data[0].shape}")

    num_bands = image_data[0].shape[2]

    rng = np.random.default_rng()
    figure, axis = plt.subplots(num_images, num_bands+1)
    for band in range(num_bands):
        # print(f"Permuting band {band} of {num_bands}")
        band_loss = 0
        processed = 0
        for i in range(num_images):
            for j in range(num_images):
                # if i != j: # all permutations
                if image_classes[i] != image_classes[j]: # Permute the class
                    processed += 1
                    perm = image_data[i].copy()
                    perm[:,:,band] = image_data[j][:,:,band]   # Permute the band
                    #print(f"band:{band} swap:{swaps[i]}")
                    #axis[i,0].imshow(image_data[i])
                    #axis[i,band+1].imshow(perm)
                    
                    # Compute the loss
                    img_array = tf.expand_dims(perm, 0)  # Create a batch
                    predictions = model.predict(img_array, verbose=0)
                    scores = tf.nn.softmax(predictions[0])
                    score = np.max(scores)
                    ans = class_names[np.argmax(scores)]
                    correct = (ans == image_classes[i])
                    tot_correct += correct
                    if correct:
                        band_loss += 1-score
                    else:
                        band_loss += score
                    # print(f"{image_file}: {ans} {score} {correct} ")
        band_loss = band_loss/processed
        print(f"Band {band+1}: loss:{band_loss} gain:{band_loss-loss} correct:{tot_correct}/{processed} ({tot_correct/processed})")
        results.append((f'Band {band+1}', band_loss - loss))

    # plot the original loss and the gain from each band
    fig, ax = plt.subplots()
    ax.bar([r[0] for r in results], [r[1] for r in results])
    ax.set_ylabel('Importance')
    ax.set_xlabel('Band')
    ax.set_title('Attribute importance')
    plt.show()

def visualise_L2_weights(model):
    """
    Generate a mosaic of L1 (red channel) weights, weighted by each L2 filter to give the overall pattern the L2 filter is looking for.
    
    Loop over layers:
        if layer name = L1_NAME, store all the filter values ready for later use
        if layer name = L2_NAME, for each filter:
            create a 9x9 array (TODO: make 7x7?)
            for each L1 filter:
                for each (x,y) in the L2 filter:
                    add to x*2,y*2 * L2 filter values
    show the filter
    """

    # Weights are shape (Y, X, bands, filters)
    # L1: (3,3,3,8) (3X3, RGB, 8 filters)
    # L2: (3,3,8,8) (3X3, 8 feature maps, 8 filters)

    print("==================================")
    print("===== VISUALISING L2 WEIGHTS =====")
    print("==================================")
  
    l1_weights = None

    for layer in model.layers:
        if layer.name == 'conv2d': # Layer 1
            l1_weights = layer.weights[0].numpy() # 1 is the bias - do we need to care? generally ~0

        elif layer.name == 'conv2d_1': # layer 2
            weights = layer.weights[0].numpy() # 1 is the bias - do we need to care? generally ~0
            height, width, bands, filters = weights.shape
            for f in range(filters):
                f_weights = weights[:,:,:,f]
                # First cut: no overlap (not correct!)
                f_out = np.zeros((height*height, width*width)) # FUDGE! Assumes all filters the same size (they currently are...)
                #f_out = np.zeros((height+(height-1), width+(width-1))) # Stride of 1 + hangover
                for l1_f in range(f_weights.shape[2]): # Loop over the inputs (L1 filters)
                    for y in range(height):
                        for x in range(width):
                            print(f"Current slice: {(f_out[y*height:(y+1)*height, x*width:(x+1)*width]).shape}")
                            print(f_out[y*height:(y+1)*height, x*width:(x+1)*width])
                            #print(f"Current slice: {(f_out[y:y+height, x:x+width]).shape}")
                            #print(f_out[y:y+height, x:x+width])
                            print("New filter weight value:")
                            print(f_weights[y,x,l1_f])
                            print(f"L1 pattern: {(l1_weights[:,:,:,l1_f]).shape}")
                            print(l1_weights[:,:,0,l1_f]) # Just the red channel...
                            f_out[y*height:(y+1)*height, x*width:(x+1)*width] = (f_out[y*height:(y+1)*height,x*width:(x+1)*width] +
                                                                                   f_weights[y,x,l1_f] * l1_weights[:,:,0,l1_f])
                            #f_out[y:y+height, x:x+width] = (f_out[y:y+height, x:x+width] +
                            #                                                       f_weights[y,x,l1_f] * l1_weights[:,:,0,l1_f])
                print("Inflated filter:")
                print(f_out)
                f_min = np.min(f_out)
                f_max = np.max(f_out)
                f_out_NORM = f_out / (f_max - f_min)  # scale spread to 1
                f_out_NORM = f_out_NORM - (f_min/(f_max-f_min))  # shift to 0..1
                print("Inflated normalised:")
                print(f_out_NORM)
                plt.imshow(f_out_NORM, cmap='gray', vmin=0, vmax=1)
                plt.show()


def report_weights(model):
    for i in range(len(model.layers)):
        print("--------------------------- LAYER ----------------------------")
        weights = model.layers[i].weights
        print(f"{model.layers[i].name}: {len(weights)}")
        if len(weights) > 0:
            #print(weights[0].shape) # Check if the Output shape matches the shape of Model.summary()
            #print(weights[1].shape) # Check if the Output shape matches the shape of Model.summary()
            #print(weights[1])
            if len(weights[0].shape) == 4 : # Y x X x bands x filters
                num_bands = weights[0].shape[2]
                num_filters = weights[0].shape[3]
                figure, axis = plt.subplots(num_filters,num_bands+1)
                axis[0,0].set_title(f"WEIGHTS FOR {model.layers[i].name}")
                ff_weights = weights[0].numpy()
                f_min = np.min(ff_weights)
                f_max = np.max(ff_weights)
                ff_weights_NORM = ff_weights / (f_max - f_min)  # scale spread to 1
                ff_weights_NORM = ff_weights_NORM - (f_min/(f_max-f_min))  # shift to 0..1
                for f in range(num_filters):
                    #print(f"Filter {f+1}/{weights[0].shape[3]}")
                    f_weights = ff_weights[:,:,:,f]
                    f_weights_NORM = ff_weights_NORM[:,:,:,f]
                    
                    #f_weights_1 = weights[0][:,:,0,f].numpy() # One channel only - grayscale images
                    #print(f_weights)
                    #print(f"NORMALIZED: min={f_min}, max={f_max}")
                    #print(f_weights_NORM)

                    # Plot the filter weights
                    f_weights_RGB = f_weights_NORM[:,:,:3]  # First three channels/layers as an RGB image - will fail for later layers...
                    axis[f, 0].imshow(f_weights_RGB)  # Plots the first three bands as RGB
                    for b in range(num_bands):
                        # plt.imshow(f_weights_1, cmap='gray')  # Plots the output of Conv2D and MaxPooling
                        axis[f, b+1].imshow(f_weights_NORM[:,:,b], cmap='gray', vmin=0, vmax=1)  # Plots the output of Conv2D and MaxPooling. Retains the same scale across bands
                        # plt.imshow(Image.fromarray(f_weights/(np.max(f_weights)/255.0),'RGB'))  # Plots the output of Conv2D and MaxPooling
                plt.show()
            else:
                print(weights[0])

def report_outputs(model, image_file, image_size):
    """
    Reports and plots the outputs for all layers in response to a particular input file
    Output tensors are (X, Y, batch, channels)
    """
    layer_input = img = tf.keras.utils.load_img(image_file, target_size=(image_size, image_size))
    layer_input = tf.expand_dims(layer_input,0)   # Add prefix of Batch Size 
    for i in range(len(model.layers)):
        print(f"++++++++++++ OUTPUTS FOR {model.layers[i].name} +++++++++++++++++++")
        # get_layer_output = K.function(inputs = model.layers[0].input, outputs = model.layers[i].output)
        get_layer_output = Model(inputs = model.layers[0].input, outputs = model.layers[i].output)
        outputs = get_layer_output(layer_input)
        #print(outputs.shape) # Check if the Output shape matches the shape of Model.summary()
        #print(outputs)   # If not Image, ie. Array, print the Values

        if outputs.ndim == 4:             # Check for Dimensionality: FMs are 1,y,x,FMs)
            # Normalise the weights so we can compare the strength across different filters
            num_outputs = outputs.shape[3]
            figure, axis = plt.subplots(1, num_outputs)
            axis[0].set_title(f"OUTPUTS FOR {model.layers[i].name}")
            o_min = np.min(outputs)
            o_max = np.max(outputs)
            outputs_NORM = outputs / (o_max - o_min)  # scale spread to 1
            outputs_NORM = outputs_NORM - (o_min/(o_max-o_min))  # shift to 0..1
            for o in range(num_outputs):
                #print(f"OUTPUT {o+1}/{num_outputs}")
                output = outputs[0,:,:,o]
                #print(output)
                output_NORM = outputs_NORM[0,:,:,o]
                axis[o].imshow(output_NORM, cmap='gray', vmin=0, vmax=1)  # Plots the output of Conv2D and MaxPooling. Retains the same stretch across outputs
            plt.show()

