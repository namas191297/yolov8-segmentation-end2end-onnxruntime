import argparse
import os
from pathlib import Path
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort
import onnxruntime_extensions
import cv2
import time

# ==============================
# Functions from the first file
# ==============================
import sys
# We'll re-implement or copy the classes and functions defined in File 1 here:

from onnxruntime_extensions.tools.pre_post_processing import PrePostProcessor, utils, Identity, Resize, LetterBox, ChannelsLastToChannelsFirst, ImageBytesToFloat, Unsqueeze, Squeeze, Transpose, Split, SelectBestBoundingBoxesByNMS, Step, create_named_value
import onnx.parser
from typing import List, Optional
from PIL import Image

class ScaleNMSBoundingBoxesAndKeyPoints(Step):
    """
    Scale bounding box and mask coordinates back to the original image size.
    """
    def __init__(self, layout: Optional[str] = "HWC", name: Optional[str] = None):
        super().__init__(["nms_step_output", "original_image", "resized_image", "letter_boxed_image"],
                         ["nms_output_with_scaled_boxes_and_masks"], name)
        self.layout_ = layout

        if layout not in ["HWC", "CHW"]:
            raise ValueError("Invalid layout. Only HWC and CHW are supported")

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        graph_input_params = []
        for idx, input_name in enumerate(self.input_names):
            input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, idx)
            graph_input_params.append(f"{input_type_str}[{input_shape_str}] {input_name}")

        graph_input_params = ', '.join(graph_input_params)

        if self.layout_ == "HWC":
            orig_image_h_w_c = "oh, ow, oc"
            scaled_image_h_w_c = "sh, sw, sc"
            letterboxed_image_h_w_c = "lh, lw, lc"
        else:
            orig_image_h_w_c = "oc, oh, ow"
            scaled_image_h_w_c = "sc, sh, sw"
            letterboxed_image_h_w_c = "lc, lh, lw"

        def split_num_outputs(num_outputs: int):
            split_input_shape_attr = ''
            if onnx_opset >= 18:
                split_input_shape_attr = f", num_outputs = {num_outputs}"
            return split_input_shape_attr

        nms_output_type_str, nms_output_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        nms_output_shape = nms_output_shape_str.split(',')
        data_size_per_result = int(nms_output_shape[-1])
        if not isinstance(data_size_per_result, int):
            raise ValueError("Shape of input must have a numeric value for the mask data size")

        data_num_splits = 4
        data_split_sizes = f"2, 2, 2, {data_size_per_result - 6}"

        graph_text = f"""\
            ScaleNMSBoundingBoxesAndKeyPoints 
            ({graph_input_params}) => ({nms_output_type_str}[{nms_output_shape_str}] {self.output_names[0]})
            {{
                i64_2 = Constant <value = int64[1] {{2}}>()
                data_split_sizes = Constant <value = int64[{data_num_splits}] {{{data_split_sizes}}}>()
                
                boxes_xy, boxes_wh_or_xy, score_class, masks = Split <axis=-1>({self.input_names[0]}, data_split_sizes)
                    
                ori_shape = Shape ({self.input_names[1]})
                scaled_shape = Shape ({self.input_names[2]})
                lettered_shape = Shape ({self.input_names[3]})
                {orig_image_h_w_c} = Split <axis = 0 {split_num_outputs(3)}> (ori_shape)
                {scaled_image_h_w_c} = Split <axis = 0 {split_num_outputs(3)}> (scaled_shape)
                {letterboxed_image_h_w_c} = Split <axis = 0 {split_num_outputs(3)}> (lettered_shape)
                swh = Concat <axis = -1> (sw, sh)
                lwh = Concat <axis = -1> (lw, lh)
                
                f_oh = Cast <to = 1> (oh)
                f_sh = Cast <to = 1> (sh)
                ratios = Div (f_oh, f_sh)
                
                pad_wh = Sub (lwh, swh)
                half_pad_wh = Div (pad_wh, i64_2)
                f_half_pad_wh = Cast <to = 1> (half_pad_wh)

                offset_boxes_xy = Sub (boxes_xy, f_half_pad_wh)
                restored_boxes = Concat <axis=-1> (offset_boxes_xy, boxes_wh_or_xy)
                scaled_boxes = Mul (restored_boxes, ratios)
                
                scaled_masks = Identity(masks)

                {self.output_names[0]} = Concat <axis=-1> (scaled_boxes, score_class, scaled_masks)
            }}
            """

        converter_graph = onnx.parser.parse_graph(graph_text)
        return converter_graph


def yolo_detection(model_file: Path, output_file: Path, output_format: str = 'jpg',
                   onnx_opset: int = 16, num_classes: int = 80, input_shape: List[int] = [640,640,3]):
    model = onnx.load(str(model_file.resolve(strict=True)))
    model_with_shape_info = onnx.shape_inference.infer_shapes(model)
    model_input_shape = model_with_shape_info.graph.input[0].type.tensor_type.shape
    model_output_shape = model_with_shape_info.graph.output[0].type.tensor_type.shape

    h_in = model_input_shape.dim[-2].dim_value
    w_in = model_input_shape.dim[-1].dim_value
    inputs = [create_named_value("rgb_data", onnx.TensorProto.UINT8, [h_in, w_in, 3])]

    pipeline = PrePostProcessor(inputs, onnx_opset)
    pipeline.add_pre_processing(
        [
            Identity(name='RGBInput'),
            Resize((h_in, w_in), policy='not_larger'),
            LetterBox(target_shape=(h_in, w_in)),
            ChannelsLastToChannelsFirst(),
            ImageBytesToFloat(),
            Unsqueeze([0]),
        ]
    )

    post_processing_steps = [
        Squeeze([0]),
        Transpose([1, 0]),
        Split(num_outputs=3, axis=-1, splits=[4, num_classes, 32]),
        SelectBestBoundingBoxesByNMS(has_mask_data=True, iou_threshold=0.5, score_threshold=0.67),
        (ScaleNMSBoundingBoxesAndKeyPoints(name='ScaleBoundingBoxes'),
         [
            utils.IoMapEntry("RGBInput", producer_idx=0, consumer_idx=1),
            utils.IoMapEntry("Resize", producer_idx=0, consumer_idx=2),
            utils.IoMapEntry("LetterBox", producer_idx=0, consumer_idx=3),
        ]),
    ]

    pipeline.add_post_processing(post_processing_steps)

    new_model = pipeline.run(model)
    _ = onnx.shape_inference.infer_shapes(new_model, strict_mode=True)
    onnx.save_model(new_model, str(output_file.resolve()))


def get_yolo_model(version: int, onnx_model_name: str, model_input_height=640, model_input_width=640):
    from pip._internal import main as pipmain
    try:
        import ultralytics
    except ImportError:
        pipmain(['install', 'ultralytics'])
        import ultralytics
    pt_model = Path(f"yolov{version}n-seg.pt")
    model = ultralytics.YOLO(str(pt_model))
    exported_filename = model.export(format="onnx", opset=15, imgsz=(model_input_height, model_input_width))
    assert exported_filename, f"Failed to export yolov{version}n.pt to onnx"
    import shutil
    shutil.move(exported_filename, onnx_model_name)


def add_pre_post_processing_to_yolo(input_model_file: Path, output_model_file: Path):
    yolo_detection(input_model_file, output_model_file, "jpg", onnx_opset=16)

# For initial inference test (if needed)
def run_inference(onnx_model_file: Path, model_input_height=640, model_input_width=640):
    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    session = ort.InferenceSession(str(onnx_model_file), providers=providers, sess_options=session_options)

    inname = [i.name for i in session.get_inputs()]
    rgb_image = np.array(Image.new('RGB', (model_input_width, model_input_height)), dtype=np.uint8)
    inp = {inname[0]: rgb_image}
    outputs = session.run(None, inp)
    print("Inference test successful. Outputs:", [o.shape for o in outputs])


def add_resize_node_to_mask_protos(input_model: str, model_size: int) -> str:
    model = onnx.load(input_model)
    graph = model.graph

    # Create target_size initializer
    target_size = helper.make_tensor('target_size', TensorProto.INT64, [4], [1, 32, model_size, model_size])
    graph.initializer.append(target_size)

    # Create Resize node
    resize_node = helper.make_node(
        'Resize',
        inputs=['output1', '', '', 'target_size'],
        outputs=['mask_protos'],
        mode='linear'
    )
    graph.node.append(resize_node)

    new_output = helper.make_tensor_value_info('mask_protos', TensorProto.FLOAT, [1, 32, model_size, model_size])
    graph.output.append(new_output)

    # Remove old output1 from graph outputs if present
    output1_index = None
    for i, output in enumerate(graph.output):
        if output.name == "output1":
            output1_index = i
            break
    if output1_index is not None:
        graph.output.remove(graph.output[output1_index])

    resized_model_name = f"yolov8n-seg_end2end_{model_size}_resizedprotos.onnx"
    onnx.save(model, resized_model_name)
    return resized_model_name


def finalize_mask_processing(input_model: str, model_size: int) -> str:
    model = onnx.load(input_model)
    graph = model.graph

    reshape_mask_shape = numpy_helper.from_array(np.array([32, model_size*model_size], dtype=np.int64), name="reshape_mask_shape")
    final_reshape_shape = numpy_helper.from_array(np.array([-1, model_size, model_size], dtype=np.int64), name="final_reshape_shape")
    start = numpy_helper.from_array(np.array([6], dtype=np.int64), name="slice_start")
    end = numpy_helper.from_array(np.array([38], dtype=np.int64), name="slice_end")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_steps")

    graph.initializer.extend([start, end, axes, steps, final_reshape_shape, reshape_mask_shape])

    # Constant threshold
    threshold = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['threshold_value'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=onnx.TensorProto.FLOAT,
            dims=[1],
            vals=[0.5],
        )
    )
    graph.node.append(threshold)

    # Slice node
    slice_node = helper.make_node(
        'Slice',
        inputs=['nms_output_with_scaled_boxes_and_masks', 'slice_start', 'slice_end', 'slice_axes', 'slice_steps'],
        outputs=['sliced_nms_output'],
        name='SliceNMSOutput'
    )
    graph.node.append(slice_node)

    # Reshape mask_protos
    reshape_mask_protos_node = helper.make_node(
        'Reshape',
        inputs=['mask_protos', 'reshape_mask_shape'],
        outputs=['reshaped_mask_protos'],
        name='ReshapeMaskProtos'
    )
    graph.node.append(reshape_mask_protos_node)

    # MatMul
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['sliced_nms_output', 'reshaped_mask_protos'],
        outputs=['matmul_output'],
        name='MatMulMasks'
    )
    graph.node.append(matmul_node)

    # Reshape final output
    reshape_final_output_node = helper.make_node(
        'Reshape',
        inputs=['matmul_output', 'final_reshape_shape'],
        outputs=['final_masks'],
        name='ReshapeFinalOutput'
    )
    graph.node.append(reshape_final_output_node)

    # Greater node for threshold
    binary_masks = helper.make_node(
        'Greater',
        inputs=['final_masks', 'threshold_value'],
        outputs=['binary_masks']
    )
    graph.node.append(binary_masks)

    # Cast bool to float
    cast_node = helper.make_node(
        'Cast',
        inputs=['binary_masks'],
        outputs=['cast_binary_masks'],
        to=TensorProto.FLOAT,
        name='CastToInt'
    )
    graph.node.append(cast_node)

    # ReduceMax to combine masks
    reduced_mask = helper.make_node(
        'ReduceMax',
        inputs=['cast_binary_masks'],
        outputs=['input_image_mask'],
        axes=[0],
        keepdims=1
    )
    graph.node.append(reduced_mask)

    final_masks_output = helper.make_tensor_value_info(
        'final_masks', TensorProto.FLOAT, [None, model_size, model_size]
    )

    input_image_mask_output = helper.make_tensor_value_info(
        'input_image_mask', TensorProto.FLOAT, [1, model_size, model_size]
    )

    graph.output.append(final_masks_output)
    graph.output.append(input_image_mask_output)

    # Remove mask_protos from outputs if present
    for i, output in enumerate(graph.output):
        if output.name == 'mask_protos':
            del graph.output[i]
            break

    final_model_name = f"yolov8n-seg-end2end_{model_size}_finalmasks.onnx"
    onnx.save(model, final_model_name)
    return final_model_name


def apply_masks_and_draw(nms_output, individual_masks, image, fps):
    original_height, original_width = image.shape[:2]

    for i, detection in enumerate(nms_output):
        xc, yc, w, h = map(int, detection[:4])
        cls = detection[5]
        conf = detection[4]

        xmin = max(0, xc - w // 2)
        ymin = max(0, yc - h // 2)
        xmax = min(original_width, xc + w // 2)
        ymax = min(original_height, yc + h // 2)

        # Resize the individual mask to the original image size
        individual_mask_resized = cv2.resize(individual_masks[i], (original_width, original_height))

        # Apply a threshold to create a binary mask
        _, mask_binary = cv2.threshold(individual_mask_resized, 0.5, 1, cv2.THRESH_BINARY)

        # Convert binary mask to uint8 for visualization
        mask_binary = (mask_binary * 255).astype(np.uint8)
        mask_region = mask_binary[ymin:ymax, xmin:xmax]

        # Create an overlay for the mask
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        mask_overlay[ymin:ymax, xmin:xmax, 1] = mask_region

        # Blend the original image with the mask overlay
        image = cv2.addWeighted(image, 1.0, mask_overlay, 0.5, 0)

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2)
        cv2.putText(image, f'Class {int(cls)} - {conf:.2f}', (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display FPS
    cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Result', image)


def run_webcam_inference(final_model_path: str, model_input_width=640, model_input_height=640):
    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    session = ort.InferenceSession(final_model_path, providers=providers, sess_options=session_options)

    inname = [i.name for i in session.get_inputs()]

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        resized_frame = cv2.resize(frame, (model_input_width, model_input_height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        inp = {inname[0]: rgb_frame}
        outputs = session.run(['nms_output_with_scaled_boxes_and_masks', 'final_masks', 'input_image_mask'], inp)

        nms_output_with_scaled_boxes_and_masks = outputs[0]
        final_masks = outputs[1]

        if nms_output_with_scaled_boxes_and_masks.shape[0] != 0:
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            apply_masks_and_draw(nms_output_with_scaled_boxes_and_masks, final_masks, frame, fps)
        else:
            cv2.imshow('Result', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine and run YOLO model post-processing pipeline with ONNX.")
    parser.add_argument('--model-size', type=int, default=640, help='Model input size (both width and height).')
    parser.add_argument('--yolo-version', type=int, default=8, help='YOLO version (e.g. 5 or 8).')
    parser.add_argument('--model', type=str, default='yolov8n-seg.onnx', help='Base YOLO ONNX model file.')
    parser.add_argument('--final-model', type=str, default='', help='Output final model name. If empty, a default name is used.')
    parser.add_argument('--download-model', action='store_true', help='Download YOLO model if not present.')
    parser.add_argument('--run-inference', action='store_true', help='Run webcam inference after model is generated.')
    parser.add_argument('--input-width', type=int, default=640, help="Input width for the model")
    parser.add_argument('--input-height', type=int, default=640, help="Input height for the model")

    args = parser.parse_args()

    model_input_height = args.input_height
    model_input_width = args.input_width
    model_size = args.model_size
    yolo_version = args.yolo_version
    base_model_path = Path(args.model)

    # Step 1: If requested, download the YOLO model
    if args.download_model and not base_model_path.exists():
        print("Fetching original model...")
        get_yolo_model(yolo_version, str(base_model_path), model_input_height, model_input_width)

    # Step 2: Add pre/post-processing to YOLO model if not done
    e2e_model_name = base_model_path.with_suffix(".with_pre_post_processing.onnx")
    if not e2e_model_name.exists():
        print("Adding pre/post processing...")
        add_pre_post_processing_to_yolo(base_model_path, e2e_model_name)

    # Step 3: Add resize node for mask protos
    resized_protos_model = add_resize_node_to_mask_protos(str(e2e_model_name), model_size)

    # Step 4: Finalize the mask processing
    final_model_name = finalize_mask_processing(resized_protos_model, model_size)
    if args.final_model:
        # rename final model if requested
        os.rename(final_model_name, args.final_model)
        final_model_name = args.final_model

    print("Model processing complete. Final model:", final_model_name)

    # Optional: Run inference using the webcam
    if args.run_inference:
        print("Running webcam inference. Press 'q' to quit.")
        run_webcam_inference(final_model_name, model_input_width, model_input_height)
