.PHONY: clean-%
clean-%:
	$(MAKE) -C $* clean

.PHONY: clean
clean: clean-frontend
	@echo Done

.PHONY: get-deps
get-deps:
	pip install -r ./requirements.txt

.PHONY: build-%
build-%:
	$(MAKE) -C $* build

.PHONY: build
build: build-frontend
	@echo Done Building


.PHONY: serve
serve: build
	python3 ./launch_backend.py

.PHONY: check
check:
	@echo It always just works
