.PHONY: test test-fast lint fmt protocol
test:        ; python -m pytest -q
test-fast:   ; python -m pytest -q -m "not slow"
lint:        ; ruff check sgta tests scripts && ruff format --check sgta tests scripts && vulture sgta --min-confidence 80
fmt:         ; ruff format sgta tests scripts
protocol:    ; sha256sum -c HYPOTHESES.sha256
