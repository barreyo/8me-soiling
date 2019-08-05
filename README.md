# 8ME Soiling Calculator

Example usage:
```bash
$ cd backend
$ python3.6 -m soiling.calculator data/prism_1.csv -y .3 .4 .5 .9
```

Where a precipitation data file path has to be supplied and the average number
of washes per year as a space separated list with `-y` argument.

For more usage details:
```bash
$ python3.6 -m soiling.calculator --help
```
