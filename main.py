import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

#Paths
charlie_path = "Charlie.jpg"  # Your uploaded image
output_path = "charlie_no_background.png"
forest_bg_path = "./forest.jpg"  # Replace with a forest image of your choice
output_video = "charlie_running.mp4"

def remove_bg_2(image_path):
    # Load image
    image = cv2.imread("Charlie.jpg")
    mask = np.zeros(image.shape[:2], np.uint8)

    # Create models for grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    user_mask = cv2.imread("charlie_mask.jpg", 0)

    # Convert user mask to grabCut-compatible codes
    # Background (0) → 0, Foreground (255) → 1
    mask[user_mask == 0] = 0  # Sure background
    mask[user_mask == 255] = 1  # Sure foreground

    # Run grabCut using mask
    cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    output_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * output_mask[:, :, np.newaxis]

    # Convert to transparent PNG
    rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = output_mask * 255

    cv2.imwrite("charlie_tail_preserved.png", rgba)

def remove_bg(image_path):
    from rembg import remove  # imported only when needed
    input_image = Image.open(image_path)
    output_image = remove(
        input_image,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=5
    )

    # Save the result
    output_image.save(output_path)

    input_img = Image.open(charlie_path).convert("RGBA")
    charlie_no_bg = remove(input_img)
    charlie_resized = charlie_no_bg.resize((350, 350), Image.LANCZOS)
    return charlie_resized

def Loading_background():
    # Load forest background
    bg = Image.open(forest_bg_path).convert("RGBA")
    return bg

def Loading_font_for_text():
    # Load font for text
    font_path = "arial.ttf"  # Optional: replace with a path to a .ttf file
    try:
        font = ImageFont.truetype(font_path, 40)
    except:
        font = ImageFont.load_default()
    return font
def print_title(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Processing your image: {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def main():
    # Load Charlie and remove background
    # Resize Charlie smaller
    charlie_resized = remove_bg()
    bg = Loading_background()
    frame_width, frame_height = bg.size

    # Video settings
    fps = 30
    duration_sec = 5
    frame_count = fps * duration_sec
    output = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    font = Loading_font_for_text()

    for frame in range(frame_count):
        x = int((frame / frame_count) * frame_width)
        frame_img = bg.copy()
        frame_img.paste(charlie_resized, (x, frame_height - 350), charlie_resized)

        # Add "Go Charlie!" text
        draw = ImageDraw.Draw(frame_img)
        draw.text((10, 10), "Go Charlie!", font=font, fill=(255, 255, 255, 255))

        # Convert to BGR for OpenCV and write frame
        frame_bgr = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGBA2BGR)
        output.write(frame_bgr)

    output.release()
    print("✅ Video created:", output_video)

print_title('Charlie')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
remove_bg(charlie_path)