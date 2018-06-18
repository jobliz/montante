Montante
===

Montante is a library that maps data processing, analysis and prediction 
tasks to a dict/jsonable payload that can be offered by a web framework

It currently allows you to run R the caret machine learning from R on a 
pandas dataframe.


Installation under Ubuntu Linux
-------------------------------

Montante depends on many different packages (R, Spark, etc) that must be
installed separately. Assuming a recent Ubuntu install, do the following:

1. Install MySQL and PosgreSQL drivers.

2. Install the requirements in requirements.txt inside a virtual environment
or with the --user pip flag.

3. Install R: `sudo apt install r-base`

4. Install R dependencies: install.packages base64enc, caret, C50, e1071.


Testing in a development environment
------------------------------------

Execute `python3 -m unittest discover -s montante/tests/ -p 'test_*.py'`