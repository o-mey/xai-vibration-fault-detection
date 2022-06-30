import scipy as sc
import tensorflow as tf
import numpy as np

def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    #return (x) / (tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(x))) + 1e-16)
    return tf.math.divide(x, tf.sqrt(tf.reduce_mean(tf.square(x))) + tf.constant(1e-5))

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

class ActivationMapper():
    def __init__(self, model, cam_layer_name = 'cam_layer'):
        self.model=model
        self.last_conv = self.model.get_layer(cam_layer_name)
        self.grad_model = tf.keras.models.Model([self.model.inputs], [self.last_conv.output, self.model.output])
        
    def grad_cam(self, input_data, cls_idx):
        with tf.GradientTape() as tape:
            #get activations maps + predictions from last conv layer
            last_conv_output, predictions = self.grad_model(input_data) 
            # the variable loss gets the probability of belonging to the defined class (the predicted class on the model output)
            y_c = predictions[:, cls_idx] 

        # calculating the gradients from the last convolutional layer to the output node of the winning class
        grads = tape.gradient(y_c,last_conv_output)[0]

        # Normalize if necessary
        grads = normalize(grads)

        # Global average pooling over the gradients values
        pooled_grads = tf.reduce_mean(grads, axis=0)

        # remove batch dimension
        last_conv_output = last_conv_output[0, :, :]
        # calculating CAM
        cam_output = np.dot(last_conv_output, pooled_grads)

        length_zoomout = input_data.shape[1] / cam_output.shape[0]
        cam_output = sc.ndimage.zoom(cam_output, length_zoomout, order=2)

        #cam_output = np.maximum(cam_output, 0)

        return cam_output

    def grad_cam_plus_plus(self, input_data, cls_idx):
        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    last_conv_output, predictions = self.grad_model(input_data) 

                    y_c = predictions[:, cls_idx]
                    first = gtape3.gradient(y_c, last_conv_output)
                second = gtape2.gradient(first, last_conv_output)
            third = gtape1.gradient(second, last_conv_output)

        n_filters = last_conv_output[0].shape[1]
        global_sum = np.sum(last_conv_output[0], axis=0)

        alpha_num = second[0]
        alpha_denom = second[0]*2.0 + third[0]*global_sum.reshape((1,n_filters))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)
        alphas = alpha_num/alpha_denom

        weights = np.maximum(first[0], 0.0)
        alpha_normalization_constant = np.sum(alphas, axis=0)

        alphas /= alpha_normalization_constant.reshape((1,n_filters))

        deep_linearization_weights = np.sum(weights*alphas,axis=0)
        cam = np.sum(deep_linearization_weights*last_conv_output[0], axis=1)

        length_zoomout = input_data.shape[1] / cam.shape[0]
        cam = sc.ndimage.zoom(cam, length_zoomout, order=2)

        #cam = np.maximum(cam, 0)  # Passing through ReLU

        max_heat = np.max(cam)
        if max_heat == 0:
            max_heat = 1e-10
        cam /= max_heat # scale 0 to 1.0  

        return cam

    def score_cam(self, input_data, cls_idx, max_N=-1):

    #     cls = np.argmax(model.predict(img_array))
        act_map_array,_ = self.grad_model.predict(input_data)

        # extract effective maps
        if max_N != -1:
            act_map_std_list = [np.std(act_map_array[0,:,k]) for k in range(act_map_array.shape[2])]
            unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
            max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
            act_map_array = act_map_array[:,:,max_N_indices]

        input_shape = self.model.layers[0].output_shape[0][1:]  # get input shape

        # 1. upsampled to original input size
        length_zoomout = input_data.shape[1] / act_map_array.shape[1]    
        act_map_resized_list = [sc.ndimage.zoom(act_map_array[0,:,k], length_zoomout, order=2) for k in range(act_map_array.shape[2])]

        # 2. normalize the raw activation value in each activation map into [0, 1]
        act_map_normalized_list = []
        for act_map_resized in act_map_resized_list:
            if np.max(act_map_resized) - np.min(act_map_resized) != 0:
                act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
            else:
                act_map_normalized = act_map_resized
            act_map_normalized_list.append(act_map_normalized)

        # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
        masked_input_list = []
        for act_map_normalized in act_map_normalized_list:
            masked_input = np.copy(input_data)
            for k in range(input_data.shape[2]):
                masked_input[0,:,k] *= act_map_normalized
            masked_input_list.append(masked_input)
        masked_input_array = np.concatenate(masked_input_list, axis=0)
        # 4. feed masked inputs into CNN model and softmax
        pred_from_masked_input_array = softmax(self.model.predict(masked_input_array))
        # 5. define weight as the score of target class
        weights = pred_from_masked_input_array[:,cls_idx]
        # 6. get final class discriminative localization map as linear weighted combination of all activation maps
        cam = np.dot(act_map_array[0,:,:], weights)
        cam = np.maximum(0, cam)  # Passing through ReLU
        cam /= np.max(cam)  # scale 0 to 1.0
        cam = sc.ndimage.zoom(cam, length_zoomout, order=2)

        return cam
