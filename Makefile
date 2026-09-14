.PHONY: up down test lint seed-db

up:
	docker compose up -d

down:
	docker compose down

test:
	pytest ml/tests

lint:
	ruff check ml

seed-db:
	@echo "seed-db ยังใช้ไม่ได้จนกว่าจะถึง Issue #5 (Postgres schema + DB layer)"
