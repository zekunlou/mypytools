# reinstall: Reinstall the package
PACKAGE_NAME="mypytools"
CONDA_ENV_NAME="mypytools_dev"

reinstall:
	@echo "Reinstalling the package $(PACKAGE_NAME)"
	@pip uninstall -y $(PACKAGE_NAME)
	@pip install -e .

clean_tmp:
	@echo "Cleaning up ./tmp/"
	@rm -rf ./tmp/

clean_conda_env:
	@echo "Cleaning up conda environment"
	@conda env remove -n $(CONDA_ENV_NAME) -y
	@conda create -n $(CONDA_ENV_NAME) python=3.10 ipython ipykernel matplotlib numpy tqdm -y
	@python -m pip install shapely==2.0.3 ase==3.22.1
	@python -m pip install -e .
