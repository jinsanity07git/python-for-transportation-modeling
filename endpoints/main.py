from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

# Load the notebook file
notebook_path = './001_fund.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as file:
    notebook = nbformat.read(file, as_version=4)

# Execute the notebook
# #  jupyter kernelspec list
executor = ExecutePreprocessor(timeout=-1,kernel_name='python3')  # No timeout
## Execute/Run (preprocess): To actually run the notebook we call the method 
#executor.preprocess(notebook, {})
print('Notebook successfully run to the end')


# Configure the HTML exporter
html_exporter = HTMLExporter()
html_exporter.exclude_input = True  # Exclude code input cells

# Convert the notebook to HTML
html_output, _ = html_exporter.from_notebook_node(notebook)

# Save the HTML output to a file
output_path = './output.html'
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(html_output)

print('Notebook exported successfully as HTML without code input cells.')
