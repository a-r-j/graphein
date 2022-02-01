#### 1.0.10 - 23/12/2021

* [Bug] Adds a fix for #74. Adding a disulfide bond to a protein with no disulphide bonds would fail. This was fixed by adding a check for the presence of a minimum of two CYS residues.

* [Improvement] - #79 Replaces `Literal` references with `typing_extensions.Literal` for Python 3.7 support.
