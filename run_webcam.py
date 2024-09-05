import onnxruntime as ort
import numpy as np
import cv2
import time
import argparse

class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def apply_masks_and_draw(nms_output, individual_masks, image, fps, original_width, original_height, model_input_width, model_input_height):
    """
    Applies masks to the input image, draws bounding boxes, and displays the FPS, while scaling the results back
    to the original image size.
    
    Args:
        nms_output (np.array): Array of NMS outputs containing bounding boxes and class scores.
        individual_masks (np.array): Array of final masks for detected objects.
        image (np.array): Original image from the video frame.
        fps (float): Frames per second for display.
        original_width (int): Original width of the image/frame.
        original_height (int): Original height of the image/frame.
        model_input_width (int): Width used for model input.
        model_input_height (int): Height used for model input.
    """
    scale_x = original_width / model_input_width
    scale_y = original_height / model_input_height

    # Iterate over each detection
    for i, detection in enumerate(nms_output):
        # Parse the bounding box and scale it back to the original size
        xc, yc, w, h = map(int, detection[:4])
        xc = int(xc * scale_x)
        yc = int(yc * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        
        xmin = max(0, xc - w // 2)
        ymin = max(0, yc - h // 2)
        xmax = min(original_width, xc + w // 2)
        ymax = min(original_height, yc + h // 2)

        score = detection[4]
        class_id = int(detection[5])

        # Add class name using cls
        class_name = class_names.get(class_id, 'Unknown')  # Get class name from the dictionary

        # Resize the individual mask to the original image size
        individual_mask_resized = cv2.resize(individual_masks[i], (original_width, original_height))

        # Apply a threshold to create a binary mask
        _, mask_binary = cv2.threshold(individual_mask_resized, 0.5, 1, cv2.THRESH_BINARY)

        # Convert binary mask to uint8 for visualization
        mask_binary = (mask_binary * 255).astype(np.uint8)

        # Extract the mask region for the current detection
        mask_region = mask_binary[ymin:ymax, xmin:xmax]

        # Create an overlay for the mask
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        mask_overlay[ymin:ymax, xmin:xmax, 1] = mask_region  # Apply mask to the green channel

        # Blend the original image with the mask overlay
        image = cv2.addWeighted(image, 1.0, mask_overlay, 0.5, 0)

        # Draw the bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2)
        cv2.putText(image, f'Class:{class_name} - {score:.2f}', (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display FPS on the top-left corner
    cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Result', image)



def main(args):
    """
    Main function to handle video capture, ONNX model inference, and applying masks.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    model_input_width = args.input_width
    model_input_height = args.input_height

    # Load ONNX model
    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 4
    session = ort.InferenceSession(str(args.model), providers=providers, sess_options=session_options)

    inname = [i.name for i in session.get_inputs()]

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start time to calculate FPS
        start_time = time.time()

        # Resize frame to model input size
        resized_frame = cv2.resize(frame, (model_input_width, model_input_height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Run inference to get both outputs
        inp = {inname[0]: rgb_frame}
        try:
            outputs = session.run(['nms_output_with_scaled_boxes_and_masks', 'final_masks', 'input_image_mask'], inp)

            # Unpack the outputs
            nms_output_with_scaled_boxes_and_masks = outputs[0]
            final_masks = outputs[1]

            if nms_output_with_scaled_boxes_and_masks.shape[0] != 0:

                # Calculate the FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)

                # Apply masks and draw on the original frame
                apply_masks_and_draw(nms_output_with_scaled_boxes_and_masks, final_masks, frame, fps, frame.shape[1], frame.shape[0], model_input_width, model_input_height)
                
            else:
                cv2.imshow('Result', frame)
        except Exception as e:
            cv2.imshow('Result', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Real-time object detection with ONNX model and webcam feed.")
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Path to the ONNX model file."
    )
    
    parser.add_argument(
        '--input-width',
        type=int,
        default=640,
        help="Width of the model input. Default is 640."
    )
    
    parser.add_argument(
        '--input-height',
        type=int,
        default=640,
        help="Height of the model input. Default is 640."
    )

    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)

