# Payroll ETL Automation – Short Overview

This project automates the CPA and PUA payroll adjustment process using a single Python script that connects directly to Box, processes all input files, and saves finalized outputs back into Box. No installation is required by the client—everything can be run through GitHub Actions.

## What the Script Does
- Connects securely to Box using GitHub Secrets (no credentials stored in code).
- Automatically finds the latest CPA and PUA files in Box.
- Loads all lookup tables needed for processing.
- Runs full ETL for:
  - **PUA** (TS-Org lookups, E-Class, Time Entry, Department, Pay Event, Job Number, etc.)
  - **CPA** (Merge BW/MN, fiscal year filter, E-Class, Time Entry, Overtime, TS-Org fields)
- Produces both **CSV** and **Excel** outputs.
- Saves finished files into the designated Box output folder with a date-stamped filename.

## Automation
The workflow is designed to run via GitHub Actions:
1. Client uploads new CPA/PUA files to Box.
2. You click “Run Workflow” in GitHub.
3. GitHub executes the ETL script in a clean environment.
4. Results are uploaded back into Box.

## Benefits
- Zero installation required for the client.
- Secure (Box credentials stored only in GitHub Secrets).
- Free (no hosting costs, no subscriptions required).
- Repeatable and fully automated.

