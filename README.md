# InferByRT
Infer with NVIDIA's TensorRT

Model List : EfficientNet

## How to Infer By TensorRT
### Download TensorRT
Download it from the [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) and check if **trtexec** is available
### modelTOonnx 
Convert a model file in the **weight.pt** format to a model file in the **weight.onnx** format.

**weight.pt -> weight.onnx**

If you want to change the form of dummy_input, use the **set_input**.
```Python
    def set_input(self, 
                  dummy_input : torch.Tensor) -> None:
        '''
        Set dummy input for the model
        format: [batch_size, channel, height, width]
        '''
        self.dummy_input = dummy_input
```
### onnxTOtrt
Convert a model file in the **weight.onnx** format to an engine file in the **engine.trt** format.
You can also use other options such as **--fp16**
```Bash
trtexec --onnx=recycle_weight.onnx --saveEngine=recycle.trt --fp16
```
### effinetRT
The inference about the EfficientNet engine can be carried out here.
```Bash
python3 effinetRT.py --mode Image --engine engine.trt --label label_map.txt --size 224 --source test_image.jpg
```
If you want to use a Raspberry Pi camera as a video source in a Jetson Nano environment, type **csi://0** in the --source. The default setting is as follows
```Python
gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224,\
            format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw,\
            width=(int)224, height=(int)224, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
```
### result
After the effinetRT.py is executed, the following format results appear in the image and terminal.
<p align="center">
    <img src="source/result.png" width="50%" height="50%">
</p>

```Bash
[02/02/2025-21:44:08] [TRT] [I] Loaded engine size: 126 MiB
[02/02/2025-21:44:09] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +2, GPU +9, now: CPU 2, GPU 130 (MiB)      
bingding: input (1, 3, 224, 224)
bingding: output (1, 5)
image_raw: (1413, 1624, 3)
max_value: 0.6667156 max_index: 1 class_name: fubao
```
### The script for the model weight in the example above is in [RuiHui](https://github.com/sihaun/RuiHui)



