# SmartHireAI

SmartHireAI is an AI-driven hiring system that runs the full recruitment lifecycle:

1. Job intake
2. AI job description (JD) creation with HR approval
3. Multi-platform ad generation with HR approval
4. Ad posting
5. Candidate application intake
6. Resume prescreening
7. Shortlist-1 and interview dispatch
8. Interview completion
9. Shortlist-2 and offer/rejection communication

The project supports React UI, Streamlit UI, and CLI execution paths.

## Table of Contents

- [What This Repository Does](#what-this-repository-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Run Modes](#run-modes)
- [Command Cookbook](#command-cookbook)
- [Expected Outputs](#expected-outputs)
- [Data Model and Persistence](#data-model-and-persistence)
- [API and UI Endpoints](#api-and-ui-endpoints)
- [Troubleshooting](#troubleshooting)
- [Developer Notes](#developer-notes)

## What This Repository Does

SmartHireAI combines LLM agents, workflow orchestration, and persistence into one hiring pipeline.

### Core Functional Capabilities

- JD generation from structured intake + policy context.
- Multi-candidate JD drafting and judge-based selection.
- HR-in-the-loop approvals for JD and ad packs.
- Ad generation for social platforms with A/B variants.
- Social posting automation to Meta and X.
- Application threshold gating before prescreen progression.
- Resume prescreen scoring and ranking.
- Interview token generation and completion ingestion.
- Final shortlisting and offer/rejection actions.
- Event, conversation, audit, and learning persistence in Postgres.

## Architecture

```text
Intake
  -> JD Agent (Create_JD)
  -> HR approval
  -> Ad Agent (Generate_Ad)
  -> HR approval
  -> Post Agent (Post_Ad)
  -> Application collection (threshold gate)
  -> Prescreen
  -> Shortlist-1
  -> Interview link dispatch
  -> Interview completion
  -> Finalize (Shortlist-2 + offer/rejections)
```

### Primary Orchestration Paths

- Unified launcher: `main_v2.py`
  - UI modes: React, Streamlit, CLI.
  - Status-only mode for existing jobs.
- Operational CLI: `hiring_pipeline_main.py`
  - End-to-end run, phase commands, single-command watcher, interview completion, finalize.
- Core workflow engine: `Pipeline/hiring_pipeline.py`
  - Handles transitions and stage services.

## Project Structure

High-value modules and what they own:

- `main_v2.py`: Unified entrypoint and launcher (React/Streamlit/CLI/status).
- `hiring_pipeline_main.py`: Operational command runner for full flow and stage-level controls.
- `Pipeline/`: Core hiring workflow, audit, learning, interview token logic.
- `Create_JD/`: JD generation pipeline implementations.
- `Generate_Ad/`: Ad generation pipeline implementations.
- `Post_Ad/`: Posting agents and platform integrations.
- `Frontend/`: Streamlit and React API/UI integration.
- `Interview_Agent/`: Browser and media interview utilities.
- `database/`: SQLAlchemy models, session, repositories.
- `services/`: Integrations such as Google Forms.
- `readme_docs/`: Additional deep docs and quickstarts.

## Requirements

- Python 3.11+ recommended (3.12 works well).
- PostgreSQL for pipeline persistence (`DATABASE_URL`).
- OpenAI API key(s) for JD/ad/prescreen LLM operations.
- Optional but commonly used:
  - Qdrant (`QDRANT_URL`, `QDRANT_API_KEY`) for policy/vector flows.
  - Google credentials for forms.
  - Meta/X credentials for live social posting.
- Node.js + npm required only for React UI mode.

## Environment Setup

### 1) Clone and install

```bash
git clone https://github.com/SyedAli127/SmartHireAI.git
cd SmartHireAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

Create a `.env` at repo root (or export vars in shell).

Use values from `readme_docs/env.example` as a base.

Minimum keys for main pipeline run:

```bash
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/smarthire
OPENAI_API_KEY=...
```

Commonly required for JD + ads + posting:

```bash
QDRANT_URL=...
QDRANT_API_KEY=...

JD_CREATION_EMB_MODEL=text-embedding-3-small
JD_CREATION_OUTLINE_LLM=gpt-4o-mini
JD_CREATION_LLM_1=gpt-5-mini
JD_CREATION_LLM_2=gpt-5-nano
JD_CREATION_LLM_3=gpt-4o-mini
JD_CREATION_JUDGE_LLM=gpt-4o-mini

AD_GENERATION_LLM_EXTRACT=gpt-4o-mini
AD_GENERATION_LLM_BRIEF=gpt-4o-mini
AD_GENERATION_LLM=gpt-4o-mini
```

### 3) Database initialization

Schema is auto-created when pipeline boots (`ensure_schema()` path). No separate migration command is required for first run in the current flow.

## Run Modes

### One-command local QA stack

Start backend + frontenddev together from repo root:

```bash
./start_local_qa.sh
```

Optional port overrides:

```bash
BACKEND_PORT=8001 FRONTEND_PORT=3000 ./start_local_qa.sh
```

This script:

- Activates `.venv`
- Starts `Frontend/react_api.py` if backend is not already healthy
- Waits for `/api/health` before launching the Next.js app in `Frontenddev`
- Stops the backend process it started when you exit with `Ctrl+C`

## 1) Unified launcher (recommended)

```bash
python main_v2.py
```

Default mode is React (`--ui react`).

Important flags:

- `--ui react|streamlit|cli`
- `--react-api-port 8001`
- `--react-ui-port 5173`
- `--status-job-id <uuid>`
- `--watch-status`
- `--status-poll-seconds 10`
- `--status-max-polls 0`

### 2) Streamlit dashboard

```bash
python main_v2.py --ui streamlit
```

### 3) Classic CLI run

```bash
python main_v2.py --ui cli
```

### 4) Status-only monitor for existing job

```bash
python main_v2.py --status-job-id 11111111-2222-3333-4444-555555555555 --watch-status --status-poll-seconds 10
```

## Command Cookbook

All commands below use `hiring_pipeline_main.py`.

### A) Full flow through interview dispatch

```bash
python hiring_pipeline_main.py \
  --resume CV/2-col-pdfcv.pdf \
  --email candidate@example.com \
  --name "Candidate One" \
  --title "AI Engineer" \
  --min-applications-threshold 5 \
  --shortlist-top-n 3
```

### B) Full flow with HR auto-approval

```bash
python hiring_pipeline_main.py \
  --resume CV/2-col-pdfcv.pdf \
  --email ua74800@gmail.com \
  --name Ali \
  --title "AI Engineer" \
  --min-applications-threshold 1 \
  --shortlist-top-n 3 \
  --hr-auto-approve-jd \
  --hr-auto-approve-ads
```

### C) Auto-drive mode (dev/test single-run automation)

```bash
python hiring_pipeline_main.py \
  --resume CV/2-col-pdfcv.pdf \
  --email ua74800@gmail.com \
  --name Ali \
  --title "AI Engineer" \
  --min-applications-threshold 1 \
  --shortlist-top-n 3 \
  --hr-auto-approve-jd \
  --hr-auto-approve-ads \
  --auto-drive
```

This mode can auto-complete interviews and run finalize in one command for local validation.

### D) Production-style phased flow

1. Create and post job:

```bash
python hiring_pipeline_main.py \
  --title "AI Engineer" \
  --min-applications-threshold 5 \
  --shortlist-top-n 3 \
  --hr-auto-approve-jd \
  --hr-auto-approve-ads
```

2. Submit applications as candidates arrive:

```bash
python hiring_pipeline_main.py \
  --submit-application-job-id 11111111-2222-3333-4444-555555555555 \
  --email candidate1@example.com \
  --name "Candidate One" \
  --resume CV/2-col-pdfcv.pdf \
  --shortlist-top-n 3
```

3. Process pending queue later (if threshold was not met earlier):

```bash
python hiring_pipeline_main.py --process-pending-job-id 11111111-2222-3333-4444-555555555555 --shortlist-top-n 3
```

4. Complete interview by token:

```bash
python hiring_pipeline_main.py \
  --complete-interview-token RAW_INTERVIEW_TOKEN \
  --score-overall 0.82 \
  --score-communication 0.78 \
  --score-technical 0.85
```

5. Finalize shortlist-2 and send offer/rejections:

```bash
python hiring_pipeline_main.py --finalize-job-id 11111111-2222-3333-4444-555555555555 --shortlist-top-n 3
```

### E) True single-command intake watcher

```bash
python hiring_pipeline_main.py \
  --single-command \
  --applications-jsonl ./applications.jsonl \
  --title "AI Engineer" \
  --min-applications-threshold 5 \
  --shortlist-top-n 3 \
  --hr-auto-approve-jd \
  --hr-auto-approve-ads
```

`applications.jsonl` example:

```json
{"email":"a@x.com","name":"Ali","resume_path":"CV/ali.pdf"}
{"email":"b@x.com","name":"Sara","resume_path":"CV/sara.pdf"}
```

### F) Browser interview server

```bash
python Interview_Agent/browser_interview_app.py
```

Default local URL:

- `http://localhost:8000/interview`

## Expected Outputs

Below are representative outputs (format may vary by stage and log level).

### 1) `main_v2.py` launch output

```text
[Interview] Starting browser interview server on port 8000...
[Interview] Browser interview server is ready.
[React API] Starting backend on port 8001...
[Frontend] Launching React control deck...
  UI URL   : http://localhost:5173
  API URL  : http://localhost:8001/api/health
```

### 2) Live status snapshot output

```text
[STATUS] Manual status command
  Job ID            : 11111111-2222-3333-4444-555555555555
  Job status        : collecting
  Applications      : 3/5 (received/threshold)
  Application states:
    - submitted: 3
  Interview states  : none
  Pending interviews: 0
  Offer states      : none
```

### 3) `hiring_pipeline_main.py` outputs

```text
INFO:__main__:Pipeline result: {'status': 'awaiting_interviews', 'job_id': '...'}
INFO:__main__:Submit application result: {'ok': True, 'processing': {...}}
INFO:__main__:Interview completion status: True
INFO:__main__:Finalize result: {'winner_application_id': '...', 'offer_sent': True}
```

## Data Model and Persistence

Primary tables in `database/models.py`:

- `jobs`
- `jd_versions`
- `ad_posts`
- `candidates`
- `applications`
- `prescreening_runs`
- `interviews`
- `interview_results`
- `shortlist_entries`
- `offers`
- `pipeline_events`
- `hr_conversations`
- `audit_findings`
- `learning_feedback`

What this gives you:

- Full lifecycle auditability
- Idempotency protections for repeated actions
- Searchable stage-by-stage history
- Traceability via `correlation_id`

## API and UI Endpoints

Typical local endpoints during React mode:

- React UI: `http://localhost:5173`
- React API health: `http://localhost:8001/api/health`
- Interview server health: `http://localhost:8000/health`
- Interview page: `http://localhost:8000/interview`

## Frontend Deployment (Vercel)

### Environment Configuration

When deploying [Frontenddev](Frontenddev) to Vercel, set these environment variables in the Vercel project dashboard:

| Variable Name | Value | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `https://<backend-url>/api` | Public backend API URL. Must be reachable from browser. |
| `BACKEND_URL` | `https://<backend-url>` | Internal URL for server-side rewrites (optional; defaults to localhost for local dev). |

### Recommended Deployment Flow

1. **Deploy Backend First**
   - Expose your [Frontend/react_api.py](Frontend/react_api.py) backend to a public URL
   - Options: ngrok tunnel, AWS/GCP/Railway, or Docker container with public endpoint
   - Verify health at `https://<backend-url>/api/health`

2. **Set Vercel Environment Variables**
   - Go to Vercel project settings → Environment Variables
   - Add `NEXT_PUBLIC_API_BASE_URL=https://<backend-url>/api`
   - Redeploy or trigger a rebuild

3. **Verify**
   - Check browser console for fetch errors
   - Verify `/api/health` endpoint from the deployed frontend:
     ```
     curl https://your-vercel-app.vercel.app/api/health
     ```

### Common Issues

**API 404: The page could not be found**

- Root cause: `NEXT_PUBLIC_API_BASE_URL` is not set or points to an unreachable backend
- Fix: Verify backend is running and reachable, then set the correct `NEXT_PUBLIC_API_BASE_URL` in Vercel

**DNS_HOSTNAME_RESOLVED_PRIVATE**

- Root cause: rewrite destination is resolving to a private IP address
- Fix: Use publicly accessible backend URL for `NEXT_PUBLIC_API_BASE_URL`

## Troubleshooting

### 1) React mode fails with npm missing

Symptom:

- Startup fails in React mode with npm not found.

Fix:

- Install Node.js/npm and verify:

```bash
node -v
npm -v
```

- Or run Streamlit mode:

```bash
python main_v2.py --ui streamlit
```

### 2) `DATABASE_URL` missing

Symptom:

- CLI exits with missing database configuration error.

Fix:

```bash
export DATABASE_URL='postgresql+psycopg2://user:pass@localhost:5432/smarthire'
```

### 3) Threshold reached but no progression

Check:

- You submitted applications to the correct `job_id`.
- Resume paths are valid and readable.
- Run pending processing manually:

```bash
python hiring_pipeline_main.py --process-pending-job-id <job_uuid> --shortlist-top-n 3
```

### 4) Social posting not happening

Check:

- Meta/X credentials are present and valid.
- Platform permissions/scopes are granted.
- Inspect `ad_posts` and `pipeline_events` records for failures.

## Developer Notes

- Existing deep docs:
  - `readme_docs/README.md`
  - `readme_docs/PIPELINE_QUICKSTART.md`
  - `readme_docs/PROJECT_DOCUMENTATION.md`
- React frontend details:
  - `Frontend/README_REACT.md`

