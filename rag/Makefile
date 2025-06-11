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

.PHONY: docker
docker: build
	docker build --progress=plain -t enabler:latest .

.PHONY: serve
serve: build
	python3 ./launch_backend.py

.PHONY: serve-docker
serve-docker:
	docker run -p 127.0.0.1:8080:5000 enabler

.PHONY: check
check:
	@echo It always just works
