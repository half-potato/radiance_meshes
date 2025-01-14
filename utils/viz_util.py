import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
import IPython.display as ipd
import numpy as np

def create_image_viewer(image_list, start_index=0, skip_value=1):
    """
    Creates an interactive viewer to flip through a list of loaded images.

    Parameters:
        image_list (list): A list of PIL images or file paths.
    """
    if not image_list:
        print("No images to display.")
        return
    
    index = {'value': start_index}  # Mutable dictionary to track image index
    output = widgets.Output()  # Output widget for updating the display

    def show_image():
        """Updates the displayed image."""
        with output:
            clear_output(wait=True)  # Clear previous image before updating
            img = image_list[index['value']]
            if isinstance(img, str):  # If path, open the image
                img = Image.open(img)
            if isinstance(img, np.ndarray):  # If path, open the image
                img = Image.fromarray((img*255).astype(np.uint8))
            display(img)
            print(f"Image {index['value'] + 1} / {len(image_list)}")  # Display image number

    def next_image(b):
        """Moves to the next image."""
        index['value'] = (index['value'] + skip_value) % len(image_list)
        show_image()

    def prev_image(b):
        """Moves to the previous image."""
        index['value'] = (index['value'] - skip_value) % len(image_list)
        show_image()

    # Create buttons
    prev_button = widgets.Button(description="Previous")
    next_button = widgets.Button(description="Next")

    prev_button.on_click(prev_image)
    next_button.on_click(next_image)

    # Display UI
    display(widgets.HBox([prev_button, next_button]))
    display(output)

    # Show first image
    show_image()
