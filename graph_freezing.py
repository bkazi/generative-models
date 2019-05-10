import os
from tensorflow.python.tools import freeze_graph

_MODEL_DIR = "./model"
_CHECKPOINT_DIR = "./checkpoints"
_GRAPH_FILE = "frozen_graph.pb"
input_node_names = "origin,incident,normal,image_rgb,image_depth,image_position"
output_node_names = "out"

def main():
    input_graph_name = "model.pb"
    output_graph_name = _GRAPH_FILE

    input_graph_path = os.path.join(_MODEL_DIR, input_graph_name)
    input_saver_def_path = ""
    input_binary = True
    input_checkpoint_path = os.path.join(_CHECKPOINT_DIR, 'model.ckpt')

    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(_MODEL_DIR, output_graph_name)
    clear_devices = False

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices, "")

if __name__ == "__main__":
    main()