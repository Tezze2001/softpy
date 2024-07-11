VENV := .venv/bin/activate
NMODULE := softpy
TEST_MODULE := tests

verify:
	pylint $(NMODULE)

tests:
	pytest -q $(TEST_MODULE)