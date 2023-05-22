from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
from traitlets import Integer
from traitlets.config import Config
from nbconvert.preprocessors import Preprocessor
from temp import dl


# Load the notebook file
notebook_path = './002-estimating-parameters.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as file:
    jake_notebook = nbformat.read(file, as_version=4)

# Process the notebook

print (len(jake_notebook) , [k for k,v in jake_notebook.items()])

jake_notebook["cells"] =  jake_notebook["cells"][:3] + jake_notebook["cells"][19:20]

print('Notebook loaded!')
# Execute the notebook
# #  jupyter kernelspec list

executor = ExecutePreprocessor(timeout=-1,kernel_name='python3')  # No timeout
## Execute/Run (preprocess): To actually run the notebook we call the method 
executor.preprocess(jake_notebook, {})
print('Notebook successfully run to the end')


# Configure the HTML exporter
## C:\Users\ZJin\anaconda3\envs\dscc\share\jupyter\nbconvert\templates\compatibility\full.tpl
# https://nbconvert.readthedocs.io/en/latest/customizing.html
# https://github.com/dunovank/jupyter-themes
# https://blog.jupyter.org/the-templating-system-of-nbconvert-6-47ea781eacd2
# https://github.com/SylvainCorlay/nbconvert-acme
# html_exporter = HTMLExporter(extra_loaders=[dl], template_file='footer')
# html_exporter = HTMLExporter(extra_loaders=[dl], template_file='matrix',theme="dark")
# html_exporter = HTMLExporter(template_name='lab',theme="dark") #reveal
# html_exporter = HTMLExporter(template_name='matrix',theme="blue") #lab
# nbconvert\exporters\html.py
#pip install jupyterlab_theme_solarized_dark
# html_exporter = HTMLExporter(template_name='matrix',theme="jupyterlab-theme-solarized-dark") #lab
html_exporter = HTMLExporter(extra_loaders=[dl],template_name='matrix',theme="jupyterlab-theme-solarized-dark") #lab
template_paths = html_exporter.template_paths


html_exporter.exclude_input = True  # Exclude code input cells
html_exporter.exclude_output_prompt = True  # Exclude code input cells
html_exporter.exclude_raw = True

# Convert the notebook to HTML
html_output, _ = html_exporter.from_notebook_node(jake_notebook)

# Save the HTML output to a file
output_path = './output.html'
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(html_output)

print('Notebook exported successfully as HTML without code input cells.')
