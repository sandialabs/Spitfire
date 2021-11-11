try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


def datafile(filename):
  """Obtain the complete path to the installed file in the src/spitfire/data directory."""
  with pkg_resources.path('spitfire.data', filename) as p:
    path = str(p)
  return path
