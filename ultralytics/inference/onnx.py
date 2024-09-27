import argparse
from pathlib import Path
import sys
import time
import os
import cv2
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import onnxruntime as ort

class ImageReader:
    def __init__(self, resize=640):
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.pil_img = None

    def __call__(self, image_path):
        self.pil_img = Image.open(image_path).convert('RGB').resize((self.resize, self.resize))
        return self.transform(self.pil_img).unsqueeze(0)
    
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def main(args):
    print("Using ONNX Runtime on device:", ort.get_device())
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    session = ort.InferenceSession(args.model, sess_options=sess_options, providers=providers)
    
    reader = ImageReader(resize=args.imgsize)
    img_path_list = []
    all_inf_time = []

    if args.imgpath is not None:
        possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png']
        for (root, dirs, files) in os.walk(args.imgpath):
            if len(files) > 0:
                for file_name in files:
                    if os.path.splitext(file_name)[1] in possible_img_extension:
                        img_path = root + '/' + file_name
                        img_path_list.append(img_path)
    else:
        img_path_list.append(args.image)
    
    for path in img_path_list:
        img_path = Path(path)
        img = reader(img_path)
        # size = torch.tensor([[img.shape[2], img.shape[3]]])
        
        start_time = time.time()
        output = session.run(
            output_names=None,
            input_feed={'images': img.numpy(), 
                        # "orig_target_sizes": size.numpy()
                        }
        )
        inf_time = time.time() - start_time
        fps = float(1/inf_time)
        print(f"Inferece time = {inf_time:.4f} s")
        print(f"FPS = {fps:.2f}")
        all_inf_time.append(inf_time)
        
        boxes = output[0][0, :, :4]  # First 4 columns
        scores = output[0][0, :, 4]  # 5th column
        labels = output[0][0, :, 5]  # 6th column

        class_name_list = ['class1', 'class10', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']

        # Random color
        # color_palette = np.random.uniform(0, 255, size=(len(class_name_list), 3)).astype(int)
        # color_map = {i: tuple(color_palette[i]) for i in range(len(class_name_list))}

        color_map = {
            0: 'red',
            1: 'blue',
            2: 'green',
            3: 'purple',
            4: 'orange',
            5: 'pink',
            6: 'yellow',
            7: 'cyan',
            8: 'magenta',
            9: 'lime'
        }

        im = reader.pil_img
        draw = ImageDraw.Draw(im)

        indices = cv2.dnn.NMSBoxes(boxes, scores, args.threshold, args.iou)

        for i in indices:
            scr = scores[i]
            lab = labels[i][scr > args.threshold]
            box = boxes[i][scr > args.threshold]

            if len(lab) == 0:  # Check if lab is empty
                continue  # Skip this iteration if no objects are detected

            # Iterate over each box and corresponding label
            for j in range(box.shape[0]):
                b = box[j]
                l = lab[j]  # Get the corresponding label for this box
                color = color_map.get(int(l), (255, 255, 255))  # Default is white if not found
                draw.rectangle(list(b), outline=color)
                draw.text((b[0], b[1]), text=str(class_name_list[int(l)]), fill='yellow')
                
        file_dir = Path(img_path).parent.parent / 'onnx_output'
        createDirectory(file_dir)
        new_file_name = os.path.basename(img_path).split('.')[0] + '_onnx' + os.path.splitext(img_path)[1]
        new_file_path = file_dir / new_file_name
        print('New File Path:', new_file_path)
        print("================================================================================")
        im.save(new_file_path)

    avr_time = sum(all_inf_time) / len(img_path_list)
    avr_fps = float(1/avr_time)
    print('Total images processed: {}'.format(len(img_path_list)))
    print(f"Average Inference time = {avr_time:.4f} s")
    print(f"Average FPS = {avr_fps:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", '-m', type=str, ) # ONNX model path
    parser.add_argument("--image", '-i', type=str, ) # Single image path
    parser.add_argument("--imgpath", '-ipth', type=str, default=None) # Directory with images
    parser.add_argument("--imgsize", '-isize', type=int, default=640) # Single image path
    parser.add_argument("--threshold", '-t', type=float, default=0.6)
    parser.add_argument("--iou", '-iou', type=float, default=0.8)
    args = parser.parse_args()

    main(args)