# Regression Test Suite

This directory contains "golden" images for regression testing.

## How to use

1.  **Add Images**: Place your difficult/failing images in this directory (e.g., `blazemire_verge.jpg`).
2.  **Run Baseline**: Run the regression script with `--update` to generate the initial `expected.json`.
    ```bash
    uv run python scripts/run_regression.py --update
    ```
3.  **Verify**: Check `expected.json` and manually correct any wrong matches. This file is your "source of truth".
4.  **Test**: After making code changes, run the script to verify no regressions.
    ```bash
    uv run python scripts/run_regression.py
    ```

## File Structure

*   `*.jpg/png`: Test images.
*   `expected.json`: Ground truth data.
