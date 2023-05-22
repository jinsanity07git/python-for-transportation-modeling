

from jinja2 import DictLoader
# C:\Users\ZJin\anaconda3\envs\dscc\share\jupyter\nbconvert\templates

dl = DictLoader(
    {
        'matrix': """
{%- extends 'lab/index.html.j2' -%}

{%- block html_head_css -%}
  {{ super() }}
  <style>
    @font-face {
      font-family: 'Acme';
      font-style: normal;
      font-weight: 400;
      src: url(http://themes.googleusercontent.com/static/fonts/acme/v2/h0STFiiHJJuefGZJAxrSiA.ttf) format('truetype');
    }
    h1, h2, h3 {
      font-family: "Acme", sans-serif
    }
    .header {
      overflow: hidden;
      background-color: #f1f1f1;
      padding: 20px 10px;
    }
    .logo-container {
      margin: auto;
      width: fit-content;
    }
  </style>

{%- endblock html_head_css -%}

{% block body_header %}


  <body class="jp-Notebook theme-blue" style="padding: 0; margin: 0;">
  <div class="header">
    <div class="logo-container">
      {% include 'matrix_logo.svg' %}
    </div>
  </div>
  <div style="padding: 20px; margin: 20px;">
{% endblock body_header %}

{% block body_footer %}
  </div>
  </body>
  <script>
    // Function to remove cells with a specific mime type
    function removeCellsByMimeType(mimeType) {
        var cells = document.querySelectorAll('[data-mime-type="' + mimeType + '"]');
        cells.forEach(function(cell) {
            cell.parentNode.removeChild(cell);
        });
    }

    // Call the function to remove stderr cells
    removeCellsByMimeType('application/vnd.jupyter.stderr');
</script>
{% endblock body_footer %}


"""
    }
)

