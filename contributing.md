# Contributing to Spitfire

## Found a bug?
First, check if we may already be aware of the problem by looking/searching through the list of [Spitfire issues and development tasks](TODO).
If you find a relevant issue let us know by commenting on the existing issue tracker.
Otherwise, the best path forward is to simplify your problem as much as you can so we can reproduce it easily with a [minimum working example (MWE)](https://en.wikipedia.org/wiki/Minimal_working_example).
A good MWE should require us to learn as little as possible about your specific problem to reproduce and verify it.
In writing your MWE, focus on the basic skeleton of your script and simplify away as many details as possible.

Once you have a good minimum working example ready, go to [our issue list](TBD) and make a new issue with the `spitfire-bug-report` template (TODO).

## Have an idea?
If you have an idea for spitfire development - a feature, not a bug fix (see above) - go to [our issue list](TBD)
and make a new issue with the `spitfire-feature-request` template (TODO).

## Have code ready to push?
If you've already been developing some code for Spitfire and want it in the master branch,
please take the following steps:
- make a branch with your changes and ask for a code review from other Spitfire developers
- ensure that any new or modified public methods have up-to-date docstrings and appropriate unit tests
- please check that all of the unit and regression tests pass with your new code
- ensure that the documentation builds successfully into HTML format and is accurate in describing new code and capabilities

### Documentation tips
To rebuild the API docs skeleton, which shouldn't be needed from here on out as we can manually add module names and files, navigate to `spitfire_docs` and run `sphinx-apidoc -e -f -o source ../spitfire/ ../setup.py`.

To build the HTML documentation, navigate to `spitfire_docs` and run `make html`.

To open the HTML documentation, point your favorite web browser to the file (you'll need the full path), `spitfire_docs/build/html/index.html`.

Here are example `NumPy`-style docstrings: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html, and here's a manual: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
Don't worry too much about consistency in writing docstrings, just be complete and "not too inconsistent" ;).
