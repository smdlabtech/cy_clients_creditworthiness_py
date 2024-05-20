# styles_app.py
import os
import streamlit as st


### Chargement de style css
def load_css(css_file):
    """
    Loads and applies a CSS file to a Streamlit application.
    
    Args:
    css_file (str): Name of CSS file to load.
    """
    # Construire le chemin complet vers le fichier CSS en utilisant le nom du fichier passé en argument
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "assets", "css", css_file)
    with open(css_path) as f:
        css = f.read()
    return css

### Chargemement d'image dans l'application
def load_img(image_file, caption=None, use_column_width=True):
    """
    Loads and displays an image in a Streamlit application.
    
    Args:
    image_file (str): Name of the image file to load.
    caption (str): Optional caption to display below the image.
    use_column_width (bool): Whether to use the column width for the image.
    """
    # Construire le chemin complet vers le fichier image en utilisant le nom du fichier passé en argument
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "assets", "img", image_file)
    return image_path


### styles_app.py
def load_html(file_name):
    """
    Load the HTML content from the given file name.
    
    Args:
    file_name (str): The name of the HTML file to load.

    Returns:
    str: The HTML content as a string.
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "assets", "html", file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found at '{file_path}'")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content


### styles_app.py
def load_js(file_name):
    """
    Load the HTML content from the given file name.
    
    Args:
    file_name (str): The name of the HTML file to load.

    Returns:
    str: The HTML content as a string.
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "assets", "js", file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found at '{file_path}'")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return content
