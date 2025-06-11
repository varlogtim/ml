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
	docker build --progress=plain -t varlogtim/enabler:latest .

.PHONY: docker-push
docker-push: docker
	docker push varlogtim/enabler:latest

.PHONY: docker-serve
docker-serve:
	docker run -p 127.0.0.1:8080:5000 varlogtim/enabler:latest

.PHONY: serve
serve: build
	python3 ./launch_backend.py

.PHONY: check
check:
	@echo It always just works
