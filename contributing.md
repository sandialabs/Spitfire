# Contributing to Spitfire

## Found a bug?
So something isn't working and you suspect Spitfire is doing something wrong.
First, check if we may already be aware of the problem by looking/searching through the list of [Spitfire issues and development tasks](https://gitlab.multiscale.utah.edu/common/spitfire/issues).
If you find a relevant issue let us know with a comment on that particular issue - if not please make a new one as follows.

The best way to help us fix your issue is to simplify it as much as you can so we can reproduce it easily with a [minimum working example (MWE)](https://en.wikipedia.org/wiki/Minimal_working_example).
A good MWE should require us to learn as little as possible about your specific problem.
In writing your MWE you should focus on the basic skeleton of your script and simplify _as much as possible_.
The more you-specific knowledge we have to learn, the harder it will be for us to fix the problem.

Once you have a good minimum working example ready, go to [our issue list](https://gitlab.multiscale.utah.edu/common/spitfire/issues)
and make a new issue with the `spitfire-bug-report` template.

## Have an idea?
If you have an idea for spitfire development - a feature, not a bug fix (see above) - go to [our issue list](https://gitlab.multiscale.utah.edu/common/spitfire/issues)
and make a new issue with the `spitfire-feature-request` template.

## Have code ready to push?!
If you've already been developing some code for Spitfire and want it in the master branch,
please take the following steps:
- make a branch with your changes, ideally squashed into a single commit just prior to review
- ensure that the any new or modified public methods are included in the documentation
- add testing for sufficiently complex code
- please check that 100% of the tests pass with your new code
- ensure that the documentation builds successfully into HTML format and is accurate in describing new code and capabilities
- make a merge request and add at least Mike Hansen (mahanse) as a reviewer

### Documentation tips
To rebuild the API docs skeleton, which shouldn't be needed from here on out as we can manually add module names and files, navigate to `docs` and run `sphinx-apidoc -e -f -o source ../spitfire/ ../setup.py`.

To build the HTML documentation, navigate to `docs` and run `make html`.

To open the HTML documentation, point your favorite web browser to the file (you'll need the full path), `docs/build/html/index.html`.

Here are example `numpy`-style docstrings: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html, and here's a manual: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
