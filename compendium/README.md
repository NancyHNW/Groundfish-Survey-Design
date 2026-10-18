# Groundfish Survey Design Research Compendium

This folder is the submission and navigation layer for the Groundfish Survey
Design project. The working source code and research data remain in their
existing locations in the repository; this compendium explains how those
materials fit together and how the work can be reproduced.

## Project

The project studies route planning for an Icelandic groundfish survey when
station catch is uncertain. A deterministic route is evaluated against
simulated catch realisations, including capacity overflows, unscheduled port
returns, fish-time limits, and route-repair policies.

## Contents

| Folder | Contents |
|---|---|
| `01_project_records/` | Decision log, meeting minutes, and shared contribution record |
| `02_methods_and_data/` | Research questions, methods, experimental design, data provenance, and system diagrams |
| `03_experiments_and_outputs/` | Experiment register, report mapping, and curated report outputs |
| `04_reproduction_and_verification/` | Installation, tests, environment, limitations, and future work |

The core method, data, experiment, reproduction, and verification documents
are included. Add the meeting-minutes PDFs to
`01_project_records/meeting_minutes/` when they are ready. The compendium is
a joint record: shared code, data, experiments, diagrams, and report outputs
are documented once rather than duplicated for each student.

## Canonical project materials

| Material | Location |
|---|---|
| Active unified code | [`../unified/`](../unified/) |
| Experiment runners | [`../experiments/`](../experiments/) |
| Automated tests | [`../tests/`](../tests/) |
| Historical catch data and fitting | [`../simulations/`](../simulations/) |
| Inherited heuristic planner | [`../final_code/`](../final_code/) |
| Inherited GFSP solver | [`../gfsp_code/`](../gfsp_code/) |
| Reports | [`../Reports/`](../Reports/) |

## Reproduction starting points

Read
[`04_reproduction_and_verification/reproduction.md`](04_reproduction_and_verification/reproduction.md)
before running experiments. The shortest initial check is:

```powershell
python -m pytest
```

The technical details, experiment flags, saved-plan behaviour, and matrix
roles are documented in the repository's main
[`README.md`](../README.md) and [`experiments/README.md`](../experiments/README.md).

## Joint contribution record

Student 1 and Student 2 are recorded in
[`01_project_records/contribution_record.md`](01_project_records/contribution_record.md).
Replace the placeholder student labels and IDs before submission, and record
each person's individual contributions alongside the shared project work.
