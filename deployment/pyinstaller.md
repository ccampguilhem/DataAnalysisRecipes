# Deploying a pandas script with pyinstaller

Let's say you have a very simple script using pandas and you want to make an application out of it. The users may not have Python and pandas installed.

You can do this pretty simply with [pyinstaller](https://www.pyinstaller.org/).

## Installing pyinstaller

Two options whether you use anaconda or not:

```sh
conda install -c conda-forge pyinstaller
```

```sh
pip install pyinstaller
```

## A example of script using pandas

test.py:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({"A": np.random.random(100),
                   "B": np.random.random(100)})

print(df.A.describe())
```

## A first attempt to build a one-file executable

The pyinstaller states that the following has to be made:

```sh
pyinstaller test.py -n foobar
```

This will create:
- a `foobar.spec` file
- a dist directory holding a foobar folder:
    - multiples shared objects required to use the application
    - a foobar executable
    
Another option you have is to pack enverything in a single file:

```sh
pyinstaller test.py -n foobar --onefile
```

This will create:
- a `foobar.spec` file
- a dist directory holding a foobar executable

## Fixing pandas error message

This is the classical way to create stand-alone application with pyinstaller. The problem is that it does not work completly and if you try to execute the application you will have error messages related to pandas library.

But there is a workaround documented [here](https://stackoverflow.com/a/36146649).

Simply edit the `foobar.spec` file to add the following lines:

```python
# Add the following
def get_pandas_path():
    import pandas
    pandas_path = pandas.__path__[0]
    return pandas_path


dict_tree = Tree(get_pandas_path(), prefix='pandas', excludes=["*.pyc"])
a.datas += dict_tree
a.binaries = filter(lambda x: 'pandas' not in x[0], a.binaries)
```

right after this:

```python
a = Analysis(['test.py'],
             pathex=['/tmp'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
```

You then need to regenerate the application using this command:

```sh
pyinstaller foobar.spec --onefile
```