from multiprocessing.sharedctypes import Value
import cantera

def check(qual, major, minor, patch=None):
  """Check cantera version, specify qualifier as one of ["pre", "at", "atleast", "post"] and major/minor/patch version numbers, using None to ignore a level of precision (default for patch is None to ignore)"""

  patch = 0 if patch is None else patch
  minor = 0 if minor is None else minor
  if major is None:
    raise ValueError('Invalid major version, None, for cantera version check.')
  
  check_version = major * 100 + minor * 10 + patch
  
  cv_split = [int(a) for a in cantera.__version__.split('.') if a.isdigit()]
  cv_patch = 0 if patch is None else cv_split[2]
  cv_minor = 0 if minor is None else cv_split[1]
  cv = 100 * cv_split[0] + 10 * cv_minor + cv_patch

  if qual == 'pre':
    return cv < check_version
  elif qual == 'at':
    return cv == check_version
  elif qual == 'atleast':
    return cv >= check_version
  elif qual == 'post':
    return cv > check_version
  else:
    raise ValueError(f'Invalid qualification {qual} for cantera version check, must be one of ["pre", "at", "atleast", "post"].')
