# Industrial Bearing PER-DDPG Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and deliver a polished Chinese Word report that integrates the two source documents with the paper's PER-DDPG method in a fog-free industrial bearing monitoring scenario.

**Architecture:** A single deterministic Python builder will encode the approved report content and Word style system. The generated DOCX will be checked structurally with `python-docx`, rendered to page PNGs with the bundled document renderer, visually inspected page by page, and then copied to the requested destination.

**Tech Stack:** Bundled Python 3, `python-docx`, OOXML helpers, bundled `render_docx.py`, LibreOffice/Poppler when available.

## Global Constraints

- The final system topology is sensor/acquisition sender -> MQTT Broker/scheduler -> heterogeneous edge node pool -> cloud when required.
- Fog nodes and the fog layer must not appear as actual components of the proposed system.
- Preserve the source timing parameters: 16 kHz, 4 ms/64 samples, 50 ms/800 samples, and 200 ms/3200 samples.
- Preserve the paper's two-stage idea: Algorithm 3.1 creates `List_seq`; adapted PER-DDPG selects an edge/cloud node and a continuous CPU ratio.
- The report must distinguish paper-faithful content from the fog-free bearing-scene adaptation.
- Final output is a standalone Chinese `.docx`; the two source documents remain unchanged.

---

### Task 1: Build The Report Content And Word Styles

**Files:**
- Create: `tools/build_bearing_per_ddpg_report.py`
- Create: `output/docs/industrial_bearing_per_ddpg_solution.docx`

**Interfaces:**
- Consumes: the approved design specification and extracted source-document facts.
- Produces: `build_report(output_path: Path) -> None` and the generated DOCX.

- [ ] **Step 1: Encode the complete Chinese report content**

Write explicit sections for source-document analysis, PER-DDPG mapping, fog-free architecture, acquisition and MQTT flow, DAG decomposition, node/resource decisions, end-to-end workflow, issue/solution matrix, validation metrics, and conclusion.

- [ ] **Step 2: Encode the standard_business_brief style tokens**

Set Letter portrait, 1-inch margins, Calibri/Chinese fallback body at 11 pt, exact heading spacing, fixed-width tables, real numbered and bulleted lists, quiet header/footer, and a memo-style first page.

- [ ] **Step 3: Generate the DOCX**

Run:

```powershell
& 'C:\Users\dafen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\build_bearing_per_ddpg_report.py --output output\docs\industrial_bearing_per_ddpg_solution.docx
```

Expected: the DOCX exists, is non-empty, and opens with `python-docx`.

### Task 2: Perform Structural And Content Audits

**Files:**
- Inspect: `output/docs/industrial_bearing_per_ddpg_solution.docx`

**Interfaces:**
- Consumes: generated DOCX.
- Produces: a passing audit result with document statistics and required-term checks.

- [ ] **Step 1: Verify required technical content**

Assert the document contains `16 kHz`, `4 ms`, `64`, `50 ms`, `800`, `200 ms`, `3200`, `List_seq`, `PER-DDPG`, `MQTT`, `边缘节点`, and `云端`.

- [ ] **Step 2: Verify fog-free proposal wording**

Inspect every occurrence of `雾` and permit it only in historical comparison statements explaining that the paper originally used a cloud-fog-edge topology and that this proposal removes it. Reject any occurrence that presents a fog node as part of the proposed architecture, action candidates, workflow, or experiments.

- [ ] **Step 3: Verify document structure**

Assert the file has a title, at least eight first-level sections, non-empty tables, headers/footers, and no empty required section.

### Task 3: Render, Inspect, And Deliver

**Files:**
- Create: `output/docs/rendered_bearing_report/page-<N>.png`
- Create: `D:\揭榜挂帅\DDPG_v2\工业轴承检测场景_PER-DDPG技术方案.docx`

**Interfaces:**
- Consumes: structurally valid DOCX.
- Produces: visually verified final DOCX at the requested destination.

- [ ] **Step 1: Render every page**

Run:

```powershell
& 'C:\Users\dafen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'C:\Users\dafen\.codex\plugins\cache\openai-primary-runtime\documents\26.709.11516\skills\documents\render_docx.py' output\docs\industrial_bearing_per_ddpg_solution.docx --output_dir output\docs\rendered_bearing_report
```

Expected: one `page-<N>.png` per rendered page.

- [ ] **Step 2: Inspect every rendered page**

Review all page PNGs for clipped text, overlaps, broken tables, missing Chinese glyphs, isolated headings, excessive blank space, and inconsistent page furniture. Revise and re-render until all pages pass.

- [ ] **Step 3: Copy the verified DOCX to the requested directory**

Create `D:\揭榜挂帅\DDPG_v2` if absent and copy the verified file as `工业轴承检测场景_PER-DDPG技术方案.docx` without modifying either source DOCX.

- [ ] **Step 4: Verify the delivered file**

Confirm destination file size and SHA-256 match the verified workspace copy.


