# SentinelAI — Final Verification Checklist

Run these **on your Mac**, from the `sentinel-ai` project root. The files in this bundle do not connect to your local containers and cannot perform the checks for you.

## A. Confirm services

```bash
docker ps --filter name=sentinel
curl -fsS http://127.0.0.1:8000/health
```

PostgreSQL and Kafka should be listed as running, and the API should report a healthy status. The dashboard should open at http://localhost:8501 when Streamlit is running.

## B. Check database counts

```bash
docker exec sentinel-postgres psql -U sentinel -d sentinel_ai -c "SELECT (SELECT COUNT(*) FROM events) AS events, (SELECT COUNT(*) FROM predictions) AS predictions, (SELECT COUNT(*) FROM alerts) AS alerts;"
```

For the demonstrated local database, expect **7, 7, 3**. Do not confuse test-fixture counts with production data.

## C. Run the known integration test

```bash
export DATABASE_URL='postgresql://sentinel:<LOCAL_PASSWORD>@127.0.0.1:5432/sentinel_ai'
python -m tests.test_system_integration
```

Use the actual local password. The test is tied to the current local seeded database.

## D. Check repository safety **before any Git push**

```bash
git status --short
git ls-files .env mlflow.db backups
```

The second command should show **no tracked secret/database/backup files**. Also review tracked and staged source for literal API keys/passwords. `git status` can fail if Git has not been initialized; in that case, do not initialize/publish until you've reviewed the directory.

If any sensitive file appears in `git ls-files`, **stop**; adding `.gitignore` alone does not untrack it or erase history.

## E. Review README and diagram

Open `README.md` in VS Code Preview (Command+Shift+V). Mermaid diagram preview support depends on your Markdown viewer; if it does not render, the source diagram in README is still present.

## Definition of done

- [ ] README reflects the actual local setup
- [ ] Architecture diagram visible or Mermaid source verified
- [ ] Existing API, database and dashboard running
- [ ] Integration test passes again
- [ ] Git tracked files reviewed for secrets

Once these are checked, the core SentinelAI portfolio project is finalized. Do not claim a production-grade deployment or real-world model validation.
