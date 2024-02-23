# reinstall: Reinstall the package
PACKAGE_NAME="mypytools"
CONDA_ENV_NAME="mptdev"
mkf_fpath=$(abspath $(lastword $(MAKEFILE_LIST)))
mkf_dpath=$(dir $(mkf_fpath))

reinstall:
	@echo "Reinstalling the package $(PACKAGE_NAME)"
	@pip uninstall -y $(PACKAGE_NAME)
	@pip install -e .

clean_tmp:
	@echo "Cleaning up ./tmp/"
	@rm -rf ./tmp/

reinstall_env:
	@echo "Cleaning up conda environment"
	@source $$(conda info --base)/etc/profile.d/conda.sh
	@conda deactivate
	@conda env remove -n $(CONDA_ENV_NAME) -y
	@rm -r $(mkf_dpath)/mypytools/mypytools.egg-info
	@rm -r $(mkf_dpath)/mypytools.egg-info
	@conda create -n $(CONDA_ENV_NAME) python=3.10 ipython ipykernel matplotlib numpy tqdm scipy -y
	@conda activate $(CONDA_ENV_NAME)
	@python -m pip install shapely==2.0.3 ase==3.22.1 clims==0.4.4
	@cd $(mkf_dpath) && python -m pip install -e .

check_path:
	@echo "Current file path: $(mkf_fpath)"
	@echo "Current directory path: $(mkf_dpath)"
