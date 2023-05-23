from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
from nbconvert.preprocessors import Preprocessor
import sys
sys.path.insert(0,"/uos/github/python-for-transportation-modeling/endpoints/choice")
from temp import dl
# conda activate arboretum

# Load the notebook file
notebook_path = './005-nested-logit.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as file:
    jake_notebook = nbformat.read(file, as_version=4)

# Process the notebook
# jake_notebook["cells"] =  jake_notebook["cells"][:3] + jake_notebook["cells"][18:23]
rawnb = jake_notebook["cells"]
jake_notebook["cells"] =  rawnb[:6] + rawnb[21:23]

print('Notebook loaded!')
# Execute the notebook
# #  jupyter kernelspec list

executor = ExecutePreprocessor(timeout=-1,kernel_name='python3')  # No timeout
## Execute/Run (preprocess): To actually run the notebook we call the method 
executor.preprocess(jake_notebook, {})
print('Notebook successfully run to the end')


# Configure the HTML exporter
html_exporter = HTMLExporter(extra_loaders=[dl],template_file='matrix',theme="jupyterlab-theme-solarized-dark") #lab
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
