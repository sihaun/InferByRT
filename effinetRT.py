
import ctypes
import random
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit 
import pycuda.driver as cuda
import tensorrt as trt
import logging
import argparse

# 에러 로그를 파일로 저장
logging.basicConfig(filename='error.log', level=logging.ERROR)

def softmax(x):
    exp_x = np.exp(x - np.max(x)) 
    return exp_x / np.sum(exp_x) 

class EffinetRT(object):
    def __init__(self, engine_file_path, num_classes, input_size, batch_size):
        try:
            stream = cuda.Stream()
            # TensorRT 로깅 설정
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            # TensorRT 런타임 생성
            runtime = trt.Runtime(TRT_LOGGER)

            # Deserialize the engine from file
            with open(engine_file_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(serialized_engine=f.read())
            context = engine.create_execution_context()
            host_inputs = []
            cuda_inputs = []
            host_outputs = []
            cuda_outputs = []
            bindings = []

            for i in engine:
                print('bingding:', i, engine.get_tensor_shape(i))
                size = trt.volume(engine.get_tensor_shape(i)) * batch_size
                dtype = trt.nptype(engine.get_tensor_dtype(i))
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                # Append the device buffer to device bindings.
                bindings.append(int(cuda_mem))
                # Append to the appropriate list.
                if i == 'input':
                    self.input_w = engine.get_tensor_shape(i)[-1]
                    self.input_h = engine.get_tensor_shape(i)[-2]
                    host_inputs.append(host_mem)
                    cuda_inputs.append(cuda_mem)
                else:
                    host_outputs.append(host_mem)
                    cuda_outputs.append(cuda_mem)

            # Store
            self.stream = stream
            self.context = context
            self.engine = engine
            self.host_inputs = host_inputs
            self.cuda_inputs = cuda_inputs
            self.host_outputs = host_outputs
            self.cuda_outputs = cuda_outputs
            self.bindings = bindings
            self.input_size = input_size
            self.batch_size = batch_size
            self.num_classes = num_classes
        except Exception as e:
            logging.error("Error in __init__: %s", str(e))
            raise

    def infer(self, raw_image):
        threading.Thread.__init__(self)
        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        # Do image preprocess
        batch_image_raw = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_size, self.input_size])
        image_raw = raw_image

        input_image, image_raw = self.preprocess_image(image_raw)
        batch_image_raw.append(image_raw)
        for i in range(self.batch_size):
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Set host in/output to the context
        context.set_tensor_address('input', int(cuda_inputs[0]))
        context.set_tensor_address('output', int(cuda_outputs[0]))
        # Run inference.
        context.execute_async_v3(stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Here we use the first row of output in that batch_size = 1
        output = np.array(host_outputs[0]).reshape(self.num_classes)
        # Do postprocess
        for i in range(self.batch_size):
            output_slice = output[i * self.num_classes: (i + 1) * self.num_classes]
            softmax_output = softmax(output_slice)
            max_value = np.max(softmax_output)
            max_index = np.argmax(softmax_output)
            class_name = categories[max_index]
        print('max_value:', max_value, 'max_index:', max_index, 'class_name:', class_name) 
               
        return image_raw

    def destroy(self):
        # Synchronize the stream before popping the context
        self.stream.synchronize()

    def __del__(self):
        self.destroy()

    def preprocess_image(self, raw_bgr_image):

        image_raw = raw_bgr_image

        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", 
            required=False,
            type=str, 
            default="Image", 
            help="Select Image or Video mode")
    parser.add_argument("-e", "--engine", 
            required=False,
            type=str, 
            default="recycle.trt",
            help="TensorRT engine file path")
    parser.add_argument("-s", "--size", 
            required=False,
            type=int, 
            default=224, 
            help="Input image size")
    parser.add_argument("-b", "--batch", 
            required=False,
            type=int, 
            default=1, 
            help="Batch size")
    parser.add_argument("-src", "--source", 
            required=False,
            type=str, 
            default="test_image_4.jpg", 
            help="Image Or Video source")
    args = parser.parse_args()
    categories = ["can", "glass", "paper", "plastic"]
    output_size = len(categories)
    try:
        er = EffinetRT(args.engine, output_size, args.size, args.batch)
        if args.mode == "Image":

            img = cv2.imread(args.source, cv2.IMREAD_COLOR)
            batch_image_raw = er.infer(img)
            cv2.imshow("Output", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(args.source)
                
            cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    batch_image_raw = er.infer(frame)
                    cv2.imshow("Output", frame)
                    cv2.waitKey(1)
                else:
                    break
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    except Exception as e:
        logging.error("Error in main: %s", str(e))
        raise

    finally:
        er.destroy()
        del er
