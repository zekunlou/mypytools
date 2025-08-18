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

# Testing and Coverage targets
test:
	@echo "Running tests with pytest"
	@python -m pytest

test-coverage:
	@echo "Running tests with coverage report"
	@python -m pytest --cov=mypytools

test-coverage-html:
	@echo "Running tests with HTML coverage report"
	@python -m pytest --cov=mypytools --cov-report=html

coverage-serve:
	@echo "Starting coverage report server at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"
	@cd htmlcov && python -m http.server 8000 --bind 127.0.0.1

coverage-open:
	@echo "Opening coverage report in browser"
	@if command -v xdg-open > /dev/null; then \
		xdg-open htmlcov/index.html; \
	elif command -v open > /dev/null; then \
		open htmlcov/index.html; \
	else \
		echo "Coverage report available at: file://$(abspath htmlcov/index.html)"; \
		echo "Copy this path to your browser to view the report"; \
	fi

clean-coverage:
	@echo "Cleaning coverage reports"
	@rm -rf htmlcov/
	@rm -f .coverage
