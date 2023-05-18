
# Python for Transportation Modeling

This course provides a basic introduction on the use of Python for
transportation planning and modeling.  It includes a brief review
of the fundamentals of writing code in Python, as well as modules
on tabular data analysis, visualizations, and geographic analysis.
The course assumes that students are already somewhat familiar with
the mathematical tools of transportation modeling in general, and
focuses exclusively on the how these models are constructed and
implemented within Python.

<img src="_static/fdot-logo.png" alt="FDOT" width="100px" align="right" >

The [first version](http://www.fsutmsonline.net/fdot-python/fdot-python-html/index.html) 
of this course was developed with funding provided by the Florida 
Department of Transportation.

This version of the course is hosted on Github, and thus can be run 
from inside [Binder](https://mybinder.org),
a free online server for Jupyter and Python.  The resources available
on this service are limited, and you likely will not find them satisfactory
for production-level transportation planning and analysis work for
nearly any purpose. However, they are sufficient to run the code demonstrated
in these training exercises, and **you will not need to install anything
on your local machine** beyond a standard web browser, which you 
undoubtedly already have.

Click here to open these tutorials online in Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jpn--/python-for-transportation-modeling.git/v1.0.1?filepath=course-content)

## update envrioment yml
```
conda env export --name dscc > dscc.yml

conda env export --name arboretum > conda-environments/arboretum_linux.yml
```

### setup
* create a 'tmp' folder under 'course-content/choice-modeling'
```
mkdir course-content/choice-modeling/tmp
```

##### installation(optional)
raise ImportError("larch cannot be installed with pip, try installing using conda-forge instead.\nSee https://larch.newman.me/v5.7.0/intro.html for instructions.")

```
- pandas >=1.2,<1.5
conda install -c conda-forge larch==5.3

conda install -c conda-forge larch==5.7.0
pip install kaleido==0.2.1
```

https://anaconda.org/conda-forge/larch/files?version=5.3.0&page=4

### git config
[github  username setting](https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git?platform=linux)
```
git config --global user.email "aaa"
git config --global user.email "@qq.com"
```
* `conda install gh --channel conda-forge`
* [Github CLI login](https://docs.github.com/en/get-started/getting-started-with-git/caching-your-github-credentials-in-git#github-cli)
* * `gh auth login`

```
import sys
import os
sys.path.insert(0, os.path.abspath('../../example-package/'))
import transportation_tutorials as tt
```

* https://pypi.org/project/osmnx/
```
pip install osmnx
```

https://pypi.org/project/contextily/
```
contextily
```
