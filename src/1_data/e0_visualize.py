import random
import cv2
import os

def visualize_yolo_annotation(image_path, annotation_path, output_path=None):
    """
    Visualize YOLO format annotations (xywh) on an image.
    
    :param image_path: Path to the input image
    :param annotation_path: Path to the YOLO annotation .txt file
    :param output_path: Path to save the output image with bounding boxes (optional)
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    img_height, img_width, _ = image.shape
    image = cv2.resize(image, (img_width//3, img_height//3))
    img_height, img_width, _ = image.shape
    
    if os.path.isfile(annotation_path):
        # Read the YOLO annotation file (class_id, x_center, y_center, width, height)
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        # Iterate over each annotation and draw the bounding boxes
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.split())
            
            # Convert normalized YOLO values to pixel values
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            
            # Calculate top-left and bottom-right corners of the bounding box
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            # Draw the bounding box on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Optionally, you can add the class_id as text above the bounding box
            cv2.putText(image, f"Class {int(class_id)}", (x_min, y_min + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else: 
        cv2.putText(image, "Background", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Display the image with bounding boxes
    cv2.imshow('YOLO Annotations', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the image with bounding boxes if output path is provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")

if __name__ == '__main__':
    # folder  = '/home/shared/FPT/projects/z_10_CBB_detection_new/data/interim/data_gen/e5/'
    folder = '/home/shared/FPT/projects/z_16_screen_detection/data/processed/final_screen_detection_v1/train'
    image_names = os.listdir(os.path.join(folder, 'images'))
    number_of_choice = 20
    image_choices = random.choices(image_names, k=number_of_choice)
    for file in image_choices:
        image_path = os.path.join(folder, 'images', file)
        annotation_path = os.path.join(folder, 'labels', file.rsplit('.', 1)[0] + '.txt')
        visualize_yolo_annotation(image_path, annotation_path)
