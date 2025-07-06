---
date: 2025-07-05
title: Example Post Title!
draft: false
toc: false 
---

# Blog with Jupyter Notebooks!

Create a directory for the blog post somewhere like:

```
content/post/test-post/
```


Within this directory create an ipynb named:
``` 
index.ipynb
```


The first cell of the notebook to have the following content:
```
---
date: 2024-02-02T04:14:54-08:00
draft: false
title: Example
---
```

You then run the following code, assuming your working directory is your sites home directory:

```
jupyter nbconvert --to markdown content/post/blog-with-jupyter/index.ipynb
```

If you want a featured image, then add an image in the same directory named

```
featured.png
```

Here is an example of running some python code:


```python
from IPython.core.display import Image
Image('https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png')
```




    
![png](index_files/index_6_0.png)
    




```python
print("Welcome to Academic Blogging!")
```

    Welcome to Academic Blogging!
    
