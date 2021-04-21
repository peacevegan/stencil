### Resouces for Package creation

- [Real Python](https://realpython.com/pypi-publish-python-package/)
- [Making Python package](https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html)
- [Free Code camp](https://www.freecodecamp.org/news/build-your-first-python-package/)

### For updating a package
- [Update pypi package](https://widdowquinn.github.io/coding/update-pypi-package/)
- [Stack Overflow](https://stackoverflow.com/questions/52700692/a-guide-for-updating-packages-on-pypi)

a. Delete all files in the dist folder.

b. Update the version number in the setup.py file.

c. Re-create the wheels:
```
python3 setup.py sdist bdist_wheel
```

d. Re-upload the new files:
```
twine upload dist/*
```