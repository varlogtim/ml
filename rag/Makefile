VERSION := $(shell cat ./VERSION)

.PHONY: clean-%
clean-%:
	$(MAKE) -C $* clean

.PHONY: clean
clean: clean-frontend
	@echo Done

.PHONY: version
version:
	@echo $(VERSION)

.PHONY: get-deps
get-deps:
	pip install -r ./requirements.txt

.PHONY: build-%
build-%:
	$(MAKE) -C $* build

.PHONY: build
build: build-frontend
	@echo Done Building

.PHONY: dev-frontend
dev-frontend:
	$(MAKE) -C frontend dev

.PHONY: docker
docker: build
	docker build -t varlogtim/enabler:latest .

.PHONY: docker-push
docker-push: docker
	docker push varlogtim/enabler:latest

.PHONY: docker-serve
docker-serve:
	docker run -p 127.0.0.1:8080:5000 varlogtim/enabler:latest

.PHONY: serve
serve: build
	python3 ./launch_backend.py

CHART_PATH := "./helm/build/enabler-chart-$(VERSION).tar.gz"
.PHONY: helm
helm:
	$(info building helm chart)
	rm -rvf ./helm/build
	mkdir ./helm/build
	# Hacky... make this better.
	sed -i 's/version: .*$$/version: $(VERSION)/' ./helm/enabler/Chart.yaml
	tar zcvf $(CHART_PATH) -C ./helm/enabler ./
	@echo $(CHART_PATH)

.PHONY: check
check:
	@echo It always just works
