# Contributing to City Pulse

Thank you for helping improve this teaching resource! Contributions of all kinds are welcome — bug fixes, new analyses, better visualisations, or additional datasets.

## Getting started

```bash
git clone https://github.com/sophie-nguyenthuthuy/city-pulse.git
cd city-pulse
pip install -r requirements.txt
pytest tests/ -v   # all 31 tests should pass
```

## What to contribute

| Type | Where |
|------|-------|
| Bug fix | Open an issue first, then a PR |
| New city data | Add rows to `data/cities.csv` + update tests |
| New ML module | Add to `src/models/`, write tests in `tests/` |
| New notebook section | Add cells to the walkthrough notebook |
| Streamlit page | Add a page in `streamlit_app/app.py` |
| Documentation | Edit `README.md` or inline docstrings |

## Code style

- Follow PEP 8
- All public functions must have a docstring
- Every new function needs at least one test in `tests/test_pipeline.py`
- Run `pytest tests/ -v` before opening a PR

## Pull request checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] New code has docstrings
- [ ] `README.md` updated if behaviour changed
- [ ] PR description explains *why* the change is needed

## Questions?

Open a GitHub Issue with the `question` label.
