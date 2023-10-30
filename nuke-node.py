import nuke
import run


def create_depth_node():
    # Create a new node
    depth_node = nuke.createNode("NoOp")
    
    # Set its name
    depth_node.setName("DepthEstimation")
    
    # Add knobs (controls) to the node
    depth_node.addKnob(nuke.String_Knob("input_file", "Input File"))
    depth_node.addKnob(nuke.String_Knob("output_path", "Output Path"))
    depth_node.addKnob(nuke.String_Knob("model_path", "Model Path"))
    depth_node.addKnob(nuke.Enumeration_Knob("model_type", "Model Type", ["dpt_beit_large_512", "dpt_beit_large_384", "dpt_beit_base_384", "dpt_swin2_large_384", "dpt_swin2_base_384", "dpt_swin2_tiny_256", "dpt_swin_large_384", "dpt_next_vit_large_384", "dpt_levit_224", "dpt_large_384", "dpt_hybrid_384", "midas_v21_384", "midas_v21_small_256" , "openvino_midas_v21_small_256"]))
    depth_node.addKnob(nuke.Boolean_Knob("optimize", "Optimize"))
    depth_node.addKnob(nuke.Boolean_Knob("side", "Side"))
    depth_node.addKnob(nuke.String_Knob("height", "Height"))
    depth_node.addKnob(nuke.Boolean_Knob("square", "Square"))
    depth_node.addKnob(nuke.Boolean_Knob("grayscale", "Grayscale"))
    
    # Add a button to run the depth estimation
    run_button = nuke.PyScript_Knob("run", "Get Depth")
    run_button.setCommand("run_depth_estimation(nuke.thisNode())")
    depth_node.addKnob(run_button)

def run_depth_estimation(node):
    # Get the values from the knobs
    input_file = node.knob("input_file").value()
    output_path = node.knob("output_path").value()
    model_path = node.knob("model_path").value()
    model_type = node.knob("model_type").value()
    optimize = node.knob("optimize").value()
    side = node.knob("side").value()
    height = node.knob("height").value()
    square = node.knob("square").value()
    grayscale = node.knob("grayscale").value()
    
    # Run the depth estimation
    run.run(input_file, output_path, model_path, model_type, optimize, side, height, square, grayscale)

create_depth_node()