import shutil
import errno
from os
from shutil import ignore_patterns

def copy(src, dest):
#Joey Payne : http://pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/
    try:
        shutil.copytree(src, dest, ignore=ignore_patterns('*.py', '*.sh', 'specificfile.file'))
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)
