import torch
import argparse
import models


class ModelToOnnx():
    def __init__(self, 
                 arch : str,
                 weight : str,
                 output_size : int):
        self.model = models.__dict__[arch](output_size)
        self.model.load_state_dict(torch.load(weight, weights_only=True))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

    def to_onnx(self, 
                 onnx_path : str) -> None:
        torch.onnx.export(
            self.model,
            self.dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )
    
    def set_input(self, 
                  dummy_input : torch.Tensor) -> None:
        '''
        Set dummy input for the model
        format: [batch_size, channel, height, width]
        '''
        self.dummy_input = dummy_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', 
                        required=False,
                        type=str, 
                        default='tf_efficientnet_b7',
                        metavar='ARCH',
                        help='model architecture')
    parser.add_argument("-w", "--weight", 
                        required=False,
                        type=str,
                        default="weight.pt",
                        metavar="WEIGHT",
                        help="weight path")
    parser.add_argument("--output-size", 
                        required=False,
                        type=int, 
                        default=5, 
                        metavar="OUTPUT_SIZE",
                        help="Output size of the model")
    parser.add_argument("-s", "--save-path", 
                        required=False,
                        type=str,
                        default="weight.onnx",
                        metavar="SAVE_PATH",
                        help="Save path of the onnx file")
    args = parser.parse_args()
    effi_to_onnx = ModelToOnnx(args.arch, args.weight, args.output_size)
    effi_to_onnx.to_onnx(args.save_path)