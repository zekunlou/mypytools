# reinstall: Reinstall the package
PACKAGE_NAME="mypytools"

reinstall:
	@echo "Reinstalling the package $(PACKAGE_NAME)"
	@pip uninstall -y $(PACKAGE_NAME)
	@pip install -e .

clean_tmp:
	@echo "Cleaning up ./tmp/"
	@rm -rf ./tmp/

