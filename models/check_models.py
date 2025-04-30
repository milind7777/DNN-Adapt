import onnx

def get_input_output_names(onnx_file_path):
    model = onnx.load(onnx_file_path)
    input_names = [input.name for input in model.graph.input]
    output_names = [output.name for output in model.graph.output]

    print(onnx_file_path)
    print(input_names)
    print(output_names)
    print("*" * 30)

get_input_output_names("efficientnetb0.onnx")
get_input_output_names("resnet18.onnx")
get_input_output_names("vit16.onnx")