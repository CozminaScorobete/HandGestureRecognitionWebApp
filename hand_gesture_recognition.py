import numpy as np
import tensorflow as tf

class HandGestureRecognition:
    def __init__(self, model_path='hand_recognition.tflite', num_threads=1):
        # Initializam TensorFlow Lite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        #Alocam memorie       
        self.interpreter.allocate_tensors()
        # Luam input and output 
        self.input_details, self.output_details = self._get_model_details()

    def __call__(self, landmark_list):
        # Set input tensor
        self._set_input_tensor(landmark_list)
        # Run inference
        self.interpreter.invoke()
        # Get output tensor
        result = self._get_output_tensor()
        # Get predicted gesture index
        result_index = self._get_predicted_index(result)
        return result_index

    def _get_model_details(self):
        # Retrieve input and output details from interpreter
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        return input_details, output_details

    def _set_input_tensor(self, landmark_list):
        # Get input tensor index
        input_index = self.input_details[0]['index']
        # Set input tensor with landmark list
        self.interpreter.set_tensor(input_index, np.array([landmark_list], dtype=np.float32))

    def _get_output_tensor(self):
        # Get output tensor index
        output_index = self.output_details[0]['index']
        # Get output tensor from interpreter
        result = self.interpreter.get_tensor(output_index)
        return result

    def _get_predicted_index(self, result):
        # Find index of maximum value in output tensor
        result_index = np.argmax(np.squeeze(result))
        return result_index
