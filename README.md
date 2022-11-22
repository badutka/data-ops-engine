## Generating documentation

Create html documentation auto-generated from code:
> pdoc --html --force src/mlengine

## Install this package locally elsewhere:

Installing package:
> pip install <path_to_this_project>

## Running tests

Run tests:
> python -m unittest discover tests

Run Coverage (--source is optional in this situation):
> coverage run -m --source=./src unittest discover -s tests

Get coverage report:
> coverage report -m

Get coverage report in html:
> coverage html

Optional:
> python -m unittest discover -s <directory> -p '*_test.py'



