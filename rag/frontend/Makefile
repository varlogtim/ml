.PHONY: clean
clean:
	rm -rf dist src/generated node_modules

.PHONY: get-deps
get-deps:
	npm ci

.PHONY: build
build:
	@rm -rf dist
	npm ci
	npm run build

.PHONY: check
check:
	npm run lint
